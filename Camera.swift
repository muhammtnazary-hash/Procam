import AVFoundation
import CoreImage
import CoreImage.CIFilterBuiltins
import CoreLocation
import CoreVideo
import ImageIO
import Metal
import os
import Photos
import UIKit
import Vision

// MARK: - Look: Blackmagic tarzı renk; karanlık sahnede daha çok gürültü azaltma + gölge açma
enum Look {
    static let context = CIContext()

    /// level: 0 kapalı, 1 doğal, 2 sinema. ev: sahne parlaklığı (düşük = karanlık)
    static func apply(_ input: CIImage, level: Int, ev: Double, denoise: Bool = true) -> CIImage {
        guard level > 0 else { return input }
        let k: CGFloat = level == 2 ? 1 : 0.5
        let dark = CGFloat(min(max((9 - ev) / 7, 0), 1)) // 0 aydınlık ... 1 karanlık
        var img = input
        if denoise {
            let nr = CIFilter.noiseReduction()
            nr.inputImage = img
            nr.noiseLevel = Float(0.006 + 0.03 * dark)
            nr.sharpness = 0.6
            img = nr.outputImage ?? img
        }
        let hs = CIFilter.highlightShadowAdjust() // karanlık bölgeleri aç, patlayan ışığı kıs
        hs.inputImage = img
        hs.shadowAmount = Float((0.2 + 0.5 * dark) * k + 0.1)
        hs.highlightAmount = Float(1 - 0.2 * k)
        img = hs.outputImage ?? img
        let c = CIFilter.toneCurve()
        c.inputImage = img
        c.point0 = CGPoint(x: 0, y: 0.02 * k)
        c.point1 = CGPoint(x: 0.25, y: 0.25 - 0.03 * k + 0.04 * dark)
        c.point2 = CGPoint(x: 0.5, y: 0.5)
        c.point3 = CGPoint(x: 0.75, y: 0.75 + 0.04 * k)
        c.point4 = CGPoint(x: 1, y: 1 - 0.02 * k)
        img = c.outputImage ?? img
        let cc = CIFilter.colorControls()
        cc.inputImage = img
        cc.saturation = Float(1 + 0.08 * k)
        cc.contrast = Float(1 + 0.04 * k)
        img = cc.outputImage ?? img
        return img
    }
}

// MARK: - Birleştirme (Metal): 4 kare detay birleştirme veya 3 pozlama HDR
final class Fusion {
    static let shared = Fusion()
    private let device = MTLCreateSystemDefaultDevice()
    private lazy var queue = device?.makeCommandQueue()
    private lazy var library: MTLLibrary? = device.flatMap { try? $0.makeLibrary(source: Fusion.source, options: nil) }
    private var pipes: [String: MTLComputePipelineState] = [:]
    private let ci = CIContext(options: [.cacheIntermediates: false])
    private let cs = CGColorSpace(name: CGColorSpace.sRGB)!

    private func pipe(_ name: String) -> MTLComputePipelineState? {
        if let p = pipes[name] { return p }
        guard let d = device, let fn = library?.makeFunction(name: name),
              let p = try? d.makeComputePipelineState(function: fn) else { return nil }
        pipes[name] = p
        return p
    }

    /// Kare genel keskinlik puanı (küçük kopyada Laplace toplamı)
    private func score(_ img: CIImage) -> Double {
        let s = img.transformed(by: CGAffineTransform(scaleX: 0.25, y: 0.25))
        let w = Int(s.extent.width), h = Int(s.extent.height)
        guard w > 4, h > 4 else { return 0 }
        var px = [UInt8](repeating: 0, count: w * h * 4)
        ci.render(s, toBitmap: &px, rowBytes: w * 4, bounds: s.extent, format: .RGBA8, colorSpace: cs)
        var sum = 0
        for y in 1..<(h - 1) {
            for x in 1..<(w - 1) {
                let i = (y * w + x) * 4
                let c = Int(px[i + 1]), l = Int(px[i - 3]), r = Int(px[i + 5])
                let u = Int(px[i - w * 4 + 1]), d = Int(px[i + w * 4 + 1])
                sum += abs(4 * c - l - r - u - d)
            }
        }
        return Double(sum)
    }

    /// frames[0] referans; diğerleri küçük kopyada bulunan kaymaya göre hizalanır
    private func align(_ frames: [CIImage]) -> [CIImage] {
        let ref = frames[0], rect = ref.extent
        let s: CGFloat = 0.25
        let small: (CIImage) -> CIImage = { $0.transformed(by: CGAffineTransform(scaleX: s, y: s)) }
        let handler = VNImageRequestHandler(ciImage: small(ref), options: [:])
        var aligned = [ref]
        for f in frames.dropFirst() {
            let req = VNTranslationalImageRegistrationRequest(targetedCIImage: small(f), options: [:])
            try? handler.perform([req])
            var out = f
            if let o = req.results?.first as? VNImageTranslationAlignmentObservation {
                let t = o.alignmentTransform
                out = f.clampedToExtent()
                    .transformed(by: CGAffineTransform(translationX: t.tx / s, y: t.ty / s))
                    .cropped(to: rect)
            }
            aligned.append(out)
        }
        return aligned
    }

    // Bayt dizisi üzerinden aktarım: yön karışıklığı yok (hep üstten alta)
    private func upload(_ imgs: [CIImage], rect: CGRect, _ dev: MTLDevice) -> [MTLTexture]? {
        let w = Int(rect.width), h = Int(rect.height)
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: w, height: h, mipmapped: false)
        d.usage = [.shaderRead, .shaderWrite]
        var buf = [UInt8](repeating: 0, count: w * h * 4)
        var out: [MTLTexture] = []
        for img in imgs {
            guard let t = dev.makeTexture(descriptor: d) else { return nil }
            ci.render(img, toBitmap: &buf, rowBytes: w * 4, bounds: rect, format: .RGBA8, colorSpace: cs)
            t.replace(region: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0, withBytes: buf, bytesPerRow: w * 4)
            out.append(t)
        }
        return out
    }

    private func download(_ t: MTLTexture) -> CIImage {
        let w = t.width, h = t.height
        var buf = [UInt8](repeating: 0, count: w * h * 4)
        t.getBytes(&buf, bytesPerRow: w * 4, from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0)
        return CIImage(bitmapData: Data(buf), bytesPerRow: w * 4, size: CGSize(width: w, height: h),
                       format: .RGBA8, colorSpace: cs)
    }

    func fuse(_ input: [CIImage]) -> CIImage { autoreleasepool { run(input, hdr: false) } }
    func fuseHDR(_ input: [CIImage]) -> CIImage { autoreleasepool { run(input, hdr: true) } }

    private func run(_ input: [CIImage], hdr: Bool) -> CIImage {
        let n = hdr ? 3 : 4
        let frames: [CIImage]
        if hdr { frames = input.count == 3 ? [input[1], input[0], input[2]] : input } // orta pozlama referans
        else { frames = input.map { ($0, score($0)) }.sorted { $0.1 > $1.1 }.map { $0.0 } } // en keskin referans
        let ref = frames[0]
        let mem = os_proc_available_memory() // bellek azsa güvenli çıkış: tek kare
        guard frames.count == n, mem == 0 || mem > 450_000_000, let dev = device, let queue = queue,
              let cb = queue.makeCommandBuffer(),
              let textures = upload(align(frames), rect: ref.extent, dev) else { return ref }
        let w = Int(ref.extent.width), h = Int(ref.extent.height)
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: w, height: h, mipmapped: false)
        d.usage = [.shaderRead, .shaderWrite]
        guard let outTex = dev.makeTexture(descriptor: d) else { return ref }
        let tg = MTLSize(width: 16, height: 16, depth: 1)
        let grid = MTLSize(width: (w + 15) / 16, height: (h + 15) / 16, depth: 1)

        if hdr {
            guard let pw = pipe("hdrWeights"), let pb = pipe("hdrBlend") else { return ref }
            let cw = (w + 7) / 8, ch = (h + 7) / 8 // ağırlık haritası 1/8 çözünürlükte (yumuşak geçiş)
            let wd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: cw, height: ch, mipmapped: false)
            wd.usage = [.shaderRead, .shaderWrite]
            guard let wt = dev.makeTexture(descriptor: wd), let e1 = cb.makeComputeCommandEncoder() else { return ref }
            e1.setComputePipelineState(pw)
            for (i, t) in textures.enumerated() { e1.setTexture(t, index: i) }
            e1.setTexture(wt, index: 3)
            e1.dispatchThreadgroups(MTLSize(width: (cw + 15) / 16, height: (ch + 15) / 16, depth: 1), threadsPerThreadgroup: tg)
            e1.endEncoding()
            guard let e2 = cb.makeComputeCommandEncoder() else { return ref }
            e2.setComputePipelineState(pb)
            for (i, t) in textures.enumerated() { e2.setTexture(t, index: i) }
            e2.setTexture(wt, index: 3)
            e2.setTexture(outTex, index: 4)
            e2.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
            e2.endEncoding()
        } else {
            guard let pf = pipe("fuse"), let e = cb.makeComputeCommandEncoder() else { return ref }
            e.setComputePipelineState(pf)
            for (i, t) in textures.enumerated() { e.setTexture(t, index: i) }
            e.setTexture(outTex, index: 4)
            e.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
            e.endEncoding()
        }
        cb.commit()
        cb.waitUntilCompleted()
        return download(outTex)
    }

    /// Gece: kareleri hizala, hareketi reddederek ortala (gürültü ~1/√N), sonra parlaklığı sahneye göre yükselt
    func night(_ input: [CIImage]) -> CIImage { autoreleasepool { runNight(input) } }

    private func runNight(_ input: [CIImage]) -> CIImage {
        let frames = input.map { ($0, score($0)) }.sorted { $0.1 > $1.1 }.map { $0.0 } // en keskin referans
        guard let ref = frames.first else { return CIImage.empty() }
        let (black, power) = nightTone(ref)
        let mem = os_proc_available_memory() // bellek azsa güvenli çıkış: tek kare
        guard frames.count >= 2, mem == 0 || mem > 700_000_000,
              let merged = accumulate(frames, black: black, power: power) else { return ref }
        return merged
    }

    /// Referans karenin küçük kopyasından kara seviyesi ve ortalama parlaklığı hedefe çeken gama gücü
    private func nightTone(_ img: CIImage) -> (Float, Float) {
        let s = img.transformed(by: CGAffineTransform(scaleX: 0.125, y: 0.125))
        let w = Int(s.extent.width), h = Int(s.extent.height)
        guard w > 4, h > 4 else { return (0, 1) }
        var px = [UInt8](repeating: 0, count: w * h * 4)
        ci.render(s, toBitmap: &px, rowBytes: w * 4, bounds: s.extent, format: .RGBA8, colorSpace: cs)
        var hist = [Int](repeating: 0, count: 256)
        for i in stride(from: 0, to: px.count, by: 4) {
            hist[(77 * Int(px[i]) + 150 * Int(px[i + 1]) + 29 * Int(px[i + 2])) >> 8] += 1
        }
        func pct(_ p: Double) -> Double {
            let target = Int(Double(w * h) * p)
            var acc = 0
            for (v, c) in hist.enumerated() { acc += c; if acc >= target { return Double(v) / 255 } }
            return 1
        }
        let black = min(pct(0.01), 0.03)
        let med = max((pct(0.5) - black) / (1 - black), 0.01)
        let power = min(max(log(0.4) / log(med), 0.4), 1.0) // hedef orta ton 0.40; asla karartmaz, en fazla 0.4 güç
        return (Float(black), Float(power))
    }

    private func accumulate(_ frames: [CIImage], black: Float, power: Float) -> CIImage? {
        guard let dev = device, let queue = queue, let pa = pipe("nightAcc"), let po = pipe("nightOut") else { return nil }
        let aligned = align(frames) // frames[0] referans
        let rect = aligned[0].extent
        let w = Int(rect.width), h = Int(rect.height)
        let fd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: w, height: h, mipmapped: false)
        fd.usage = [.shaderRead, .shaderWrite]
        let od = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: w, height: h, mipmapped: false)
        od.usage = [.shaderRead, .shaderWrite]
        guard var accA = dev.makeTexture(descriptor: fd), var accB = dev.makeTexture(descriptor: fd),
              let refTex = upload([aligned[0]], rect: rect, dev)?.first else { return nil }
        let tg = MTLSize(width: 16, height: 16, depth: 1)
        let grid = MTLSize(width: (w + 15) / 16, height: (h + 15) / 16, depth: 1)
        for (i, img) in aligned.enumerated() { // kare kare: bellek N'den bağımsız kalır
            var ok = true
            autoreleasepool {
                var first: UInt32 = i == 0 ? 1 : 0
                guard let cur = i == 0 ? refTex : upload([img], rect: rect, dev)?.first,
                      let cb = queue.makeCommandBuffer(), let e = cb.makeComputeCommandEncoder() else { ok = false; return }
                e.setComputePipelineState(pa)
                e.setTexture(refTex, index: 0)
                e.setTexture(cur, index: 1)
                e.setTexture(accA, index: 2)
                e.setTexture(accB, index: 3)
                e.setBytes(&first, length: MemoryLayout<UInt32>.size, index: 0)
                e.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
                e.endEncoding()
                cb.commit()
                cb.waitUntilCompleted()
            }
            if !ok { return nil }
            swap(&accA, &accB) // accA hep son toplam
        }
        var tn = SIMD2<Float>(black, power)
        guard let outTex = dev.makeTexture(descriptor: od), let cb = queue.makeCommandBuffer(),
              let e = cb.makeComputeCommandEncoder() else { return nil }
        e.setComputePipelineState(po)
        e.setTexture(accA, index: 0)
        e.setTexture(outTex, index: 1)
        e.setBytes(&tn, length: MemoryLayout<SIMD2<Float>>.size, index: 0)
        e.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
        e.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        return download(outTex)
    }

    static let source = """
    #include <metal_stdlib>
    using namespace metal;
    float Y(texture2d<float, access::read> t, int2 p) {
        p = clamp(p, int2(0), int2(t.get_width() - 1, t.get_height() - 1));
        return dot(t.read(uint2(p)).rgb, float3(0.299, 0.587, 0.114));
    }
    float S(texture2d<float, access::read> t, int2 p) {
        float s = 0.0;
        for (int dy = -1; dy <= 1; dy += 2) {
            for (int dx = -1; dx <= 1; dx += 2) {
                int2 q = p + int2(dx, dy);
                s += fabs(4.0 * Y(t, q) - Y(t, q + int2(1, 0)) - Y(t, q - int2(1, 0))
                          - Y(t, q + int2(0, 1)) - Y(t, q - int2(0, 1)));
            }
        }
        return s;
    }
    kernel void fuse(texture2d<float, access::read> t0 [[texture(0)]],
                     texture2d<float, access::read> t1 [[texture(1)]],
                     texture2d<float, access::read> t2 [[texture(2)]],
                     texture2d<float, access::read> t3 [[texture(3)]],
                     texture2d<float, access::write> o [[texture(4)]],
                     uint2 g [[thread_position_in_grid]]) {
        if (g.x >= o.get_width() || g.y >= o.get_height()) return;
        int2 p = int2(g);
        float4 c0 = t0.read(g), c1 = t1.read(g), c2 = t2.read(g), c3 = t3.read(g);
        float y0 = Y(t0, p);
        float d1 = Y(t1, p) - y0, d2 = Y(t2, p) - y0, d3 = Y(t3, p) - y0;
        float w0 = pow(S(t0, p) + 0.002, 3.0);
        float w1 = pow(S(t1, p) + 0.002, 3.0) * exp(-d1 * d1 / 0.01);
        float w2 = pow(S(t2, p) + 0.002, 3.0) * exp(-d2 * d2 / 0.01);
        float w3 = pow(S(t3, p) + 0.002, 3.0) * exp(-d3 * d3 / 0.01);
        float4 r = (c0 * w0 + c1 * w1 + c2 * w2 + c3 * w3) / (w0 + w1 + w2 + w3);
        o.write(float4(r.rgb, 1.0), g);
    }
    float E(float4 c) {
        float y = dot(c.rgb, float3(0.299, 0.587, 0.114));
        float sat = max(c.r, max(c.g, c.b)) - min(c.r, min(c.g, c.b));
        return exp(-(y - 0.5) * (y - 0.5) / 0.08) * (0.2 + sat);
    }
    float Mn(texture2d<float, access::read> t, int2 p) { // 3x3 ortalama parlaklık (gürültüye dayanıklı kıyas)
        float s = 0.0;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) { s += Y(t, p + int2(dx, dy)); }
        }
        return s / 9.0;
    }
    kernel void nightAcc(texture2d<float, access::read> ref [[texture(0)]],
                         texture2d<float, access::read> cur [[texture(1)]],
                         texture2d<float, access::read> prev [[texture(2)]],
                         texture2d<float, access::write> next [[texture(3)]],
                         constant uint &first [[buffer(0)]],
                         uint2 g [[thread_position_in_grid]]) {
        if (g.x >= next.get_width() || g.y >= next.get_height()) return;
        int2 p = int2(g);
        float mr = Mn(ref, p), mc = Mn(cur, p);
        float sg = 0.03 + 0.12 * mr;
        float d = mc - mr;
        float w = exp(-d * d / (2.0 * sg * sg)); // referanstan çok sapan piksel (hareket) az ağırlık alır
        float4 a = first != 0 ? float4(0.0) : prev.read(g);
        next.write(float4(a.rgb + cur.read(g).rgb * w, a.a + w), g);
    }
    kernel void nightOut(texture2d<float, access::read> acc [[texture(0)]],
                         texture2d<float, access::write> o [[texture(1)]],
                         constant float2 &tn [[buffer(0)]],
                         uint2 g [[thread_position_in_grid]]) {
        if (g.x >= o.get_width() || g.y >= o.get_height()) return;
        float4 a = acc.read(g);
        float3 c = a.rgb / max(a.a, 0.0001);
        c = saturate((c - tn.x) / (1.0 - tn.x)); // kara seviyesi
        c = pow(c, float3(tn.y));                // gölgeleri aç, parlak ışıkları koru
        o.write(float4(c, 1.0), g);
    }
    kernel void hdrWeights(texture2d<float, access::read> t0 [[texture(0)]],
                           texture2d<float, access::read> t1 [[texture(1)]],
                           texture2d<float, access::read> t2 [[texture(2)]],
                           texture2d<float, access::write> wt [[texture(3)]],
                           uint2 g [[thread_position_in_grid]]) {
        if (g.x >= wt.get_width() || g.y >= wt.get_height()) return;
        uint W = t0.get_width(), H = t0.get_height();
        float3 a = float3(0.0);
        for (uint j = 0; j < 4; j++) {
            for (uint i = 0; i < 4; i++) {
                uint2 p = uint2(min(g.x * 8 + i * 2, W - 1), min(g.y * 8 + j * 2, H - 1));
                a += float3(E(t0.read(p)), E(t1.read(p)), E(t2.read(p)));
            }
        }
        wt.write(float4(a, 1.0), g);
    }
    kernel void hdrBlend(texture2d<float, access::read> t0 [[texture(0)]],
                         texture2d<float, access::read> t1 [[texture(1)]],
                         texture2d<float, access::read> t2 [[texture(2)]],
                         texture2d<float, access::sample> wt [[texture(3)]],
                         texture2d<float, access::write> o [[texture(4)]],
                         uint2 g [[thread_position_in_grid]]) {
        if (g.x >= o.get_width() || g.y >= o.get_height()) return;
        constexpr sampler s(coord::normalized, filter::linear, address::clamp_to_edge);
        float2 uv = (float2(g) + 0.5) / float2(o.get_width(), o.get_height());
        float3 w = wt.sample(s, uv).rgb + 0.0001;
        float3 r = (t0.read(g).rgb * w.x + t1.read(g).rgb * w.y + t2.read(g).rgb * w.z) / (w.x + w.y + w.z);
        o.write(float4(r, 1.0), g);
    }
    """
}

// MARK: - Kamera
final class CameraManager: NSObject, ObservableObject, AVCapturePhotoCaptureDelegate, AVCaptureFileOutputRecordingDelegate,
                           AVCaptureVideoDataOutputSampleBufferDelegate, CLLocationManagerDelegate {
    enum Mode { case photo, video }

    static let presets: [(name: String, w: Int32, h: Int32, fps: Int32)] = [
        ("4K 60", 3840, 2160, 60), ("4K 30", 3840, 2160, 30), ("4K 24", 3840, 2160, 24), ("1080p 120", 1920, 1080, 120)
    ]

    static let nightFrames = 8

    /// Arayüz değerlerinin ana iş parçacığında alınmış kopyası (arka planda güvenle okunur)
    struct Ctl {
        var autoExp: Bool; var iso: Float; var denom: Double
        var af: Bool; var lens: Float
        var awb: Bool; var kelvin: Float
        var zoom: CGFloat
    }

    let session = AVCaptureSession()
    @Published var mode: Mode = .photo
    @Published var preset = 0
    @Published var auto = true
    @Published var iso: Float = 100
    @Published var shutterPos: Double = 0.5
    @Published var af = true
    @Published var lens: Float = 0.5
    @Published var awb = true
    @Published var kelvin: Float = 5000
    @Published var zoom: CGFloat = 1
    @Published var lookLevel = 2
    @Published var status = ""
    @Published var info = ""
    @Published var busy = false
    @Published var recording = false
    @Published var hdr = false
    @Published var night = true // OTO: yalnızca karanlık sahnede devreye girer
    @Published var monLevel = 0 { didSet { monFlag = monLevel } } // 0 kapalı, 1 hist, 2 zebra, 3 odak, 4 hepsi
    @Published var mon = Monitor()

    struct Monitor {
        static let cols = 36, rows = 64
        var hist = [Float]()
        var zebra = [Bool]()
        var peak = [Bool]()
    }
    var previewAspect: CGFloat { mode == .photo ? 0.75 : 9.0 / 16.0 }

    var shutterDenom: Double { 24 * pow(2000.0 / 24.0, shutterPos) }
    private var ctl: Ctl {
        Ctl(autoExp: auto, iso: iso, denom: shutterDenom, af: af, lens: lens, awb: awb, kelvin: kelvin, zoom: zoom)
    }

    private let q = DispatchQueue(label: "procam.session")
    private var device: AVCaptureDevice?
    private let photoOut = AVCapturePhotoOutput()
    private let movieOut = AVCaptureMovieFileOutput()
    private let videoOut = AVCaptureVideoDataOutput()
    private let monQ = DispatchQueue(label: "procam.monitor")
    private var monFlag = 0
    private var frameN = 0
    private let loc = CLLocationManager()
    private var lastLoc: CLLocation?
    private var meta: [String: Any] = [:]
    private var hdrShot = false
    private var frames: [CIImage] = []
    private var pending = 0
    private var liveISO: Float = 100
    private var liveShutter: Double = 1.0 / 60
    private var ev: Double = 9
    private var bias: Float = 0
    private var shotEV: Double = 9
    private var shotLook = 2
    private var started = false
    private var nightShot = false
    private var nightActive = false
    private var shotCtl: Ctl?

    // MARK: Başlatma
    func start() {
        guard !started else { return }
        started = true
        UIDevice.current.beginGeneratingDeviceOrientationNotifications()
        loc.delegate = self
        loc.desiredAccuracy = kCLLocationAccuracyHundredMeters
        loc.requestWhenInUseAuthorization()
        loc.startUpdatingLocation()
        PHPhotoLibrary.requestAuthorization(for: .addOnly) { _ in }
        AVCaptureDevice.requestAccess(for: .video) { ok in
            guard ok else { self.say("Kamera izni yok (Ayarlar > ProCam)"); return }
            AVCaptureDevice.requestAccess(for: .audio) { _ in self.q.async { self.configure() } }
        }
        Timer.scheduledTimer(withTimeInterval: 0.4, repeats: true) { [weak self] _ in self?.tick() }
    }

    private func say(_ s: String) { DispatchQueue.main.async { self.status = s } }
    private func finish(_ s: String) { DispatchQueue.main.async { self.status = s; self.busy = false } }

    private func configure() {
        guard let d = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let vin = try? AVCaptureDeviceInput(device: d) else { say("Kamera açılamadı"); return }
        session.beginConfiguration()
        session.sessionPreset = .inputPriority
        if session.canAddInput(vin) { session.addInput(vin) }
        if let m = AVCaptureDevice.default(for: .audio), let ain = try? AVCaptureDeviceInput(device: m),
           session.canAddInput(ain) { session.addInput(ain) }
        if session.canAddOutput(photoOut) { session.addOutput(photoOut); photoOut.isHighResolutionCaptureEnabled = true }
        if session.canAddOutput(movieOut) { session.addOutput(movieOut) }
        videoOut.alwaysDiscardsLateVideoFrames = true
        videoOut.setSampleBufferDelegate(self, queue: monQ)
        if session.canAddOutput(videoOut) {
            session.addOutput(videoOut)
            videoOut.connection(with: .video)?.videoOrientation = .portrait // ekranla aynı yön
        }
        session.commitConfiguration()
        device = d
        DispatchQueue.main.async { let c = self.ctl; self.q.async { self.applyMode(.photo, 0, c); self.session.startRunning() } }
    }

    // Sahne parlaklığı (EV) canlı hesaplanır; Otomatik modda pozlama ve Look buna uyarlanır
    private func tick() {
        guard let d = device else { return }
        liveISO = d.iso
        liveShutter = max(CMTimeGetSeconds(d.exposureDuration), 0.0001)
        ev = log2(1.8 * 1.8 / liveShutter) - log2(Double(max(liveISO, 1)) / 100)
        info = String(format: "ISO %.0f · 1/%.0f · EV %.1f · x%.1f", liveISO, 1 / liveShutter, ev, Double(zoom))
        guard auto, !nightActive else { return }
        let target: Float = ev > 13 ? -0.5 : (ev < 4 ? 0.3 : 0) // parlakta ışığı koru, karanlıkta biraz aç
        guard target != bias else { return }
        bias = target
        q.async {
            guard let d = self.device, (try? d.lockForConfiguration()) != nil else { return }
            d.setExposureTargetBias(target, completionHandler: nil)
            d.unlockForConfiguration()
        }
    }

    // MARK: Kontroller
    func push() { let c = ctl; q.async { self.apply(c) } }

    private func apply(_ c: Ctl) {
        guard let d = device, (try? d.lockForConfiguration()) != nil else { return }
        defer { d.unlockForConfiguration() }
        if c.autoExp {
            if d.isExposureModeSupported(.continuousAutoExposure) { d.exposureMode = .continuousAutoExposure }
        } else {
            let f = d.activeFormat
            var t = CMTime(seconds: 1 / c.denom, preferredTimescale: 1_000_000)
            t = CMTimeMinimum(CMTimeMaximum(t, f.minExposureDuration), f.maxExposureDuration)
            d.setExposureModeCustom(duration: t, iso: min(max(c.iso, f.minISO), f.maxISO), completionHandler: nil)
        }
        if c.af || !d.isLockingFocusWithCustomLensPositionSupported {
            if d.isFocusModeSupported(.continuousAutoFocus) { d.focusMode = .continuousAutoFocus }
        } else {
            d.setFocusModeLocked(lensPosition: c.lens, completionHandler: nil)
        }
        if c.awb {
            if d.isWhiteBalanceModeSupported(.continuousAutoWhiteBalance) { d.whiteBalanceMode = .continuousAutoWhiteBalance }
        } else if d.isLockingWhiteBalanceWithCustomDeviceGainsSupported {
            let tt = AVCaptureDevice.WhiteBalanceTemperatureAndTintValues(temperature: c.kelvin, tint: 0)
            var g = d.deviceWhiteBalanceGains(for: tt)
            let m = d.maxWhiteBalanceGain
            g.redGain = min(max(g.redGain, 1), m)
            g.greenGain = min(max(g.greenGain, 1), m)
            g.blueGain = min(max(g.blueGain, 1), m)
            d.setWhiteBalanceModeLocked(with: g, completionHandler: nil)
        }
        d.videoZoomFactor = min(max(c.zoom, 1), min(d.activeFormat.videoMaxZoomFactor, 5))
    }

    func toggleAE() {
        if auto {
            iso = liveISO
            shutterPos = log(min(max(1 / liveShutter, 24), 2000) / 24) / log(2000.0 / 24.0)
        }
        auto.toggle()
        push()
    }
    func toggleAF() {
        if af, let d = device { lens = d.lensPosition }
        af.toggle()
        push()
    }
    func toggleWB() {
        if awb, let d = device {
            kelvin = min(max(d.temperatureAndTintValues(for: d.deviceWhiteBalanceGains).temperature, 2500), 8000)
        }
        awb.toggle()
        push()
    }
    func pinch(_ delta: CGFloat) { zoom = min(max(zoom * delta, 1), 5); push() }

    func focus(at p: CGPoint) {
        let c = ctl
        q.async {
            guard let d = self.device, (try? d.lockForConfiguration()) != nil else { return }
            if c.af, d.isFocusPointOfInterestSupported {
                d.focusPointOfInterest = p
                d.focusMode = .continuousAutoFocus
            }
            if c.autoExp, d.isExposurePointOfInterestSupported {
                d.exposurePointOfInterest = p
                d.exposureMode = .continuousAutoExposure
            }
            d.unlockForConfiguration()
        }
        say("Odak/pozlama noktası ayarlandı")
    }

    // MARK: Mod / çözünürlük
    private func applyMode(_ m: Mode, _ pi: Int, _ c: Ctl) {
        guard let d = device, (try? d.lockForConfiguration()) != nil else { return }
        let p = Self.presets[pi]
        let (w, h, fps): (Int32, Int32, Int32) = m == .video ? (p.w, p.h, p.fps) : (4032, 3024, 30)
        if let f = d.formats.first(where: { fm in
            let dim = CMVideoFormatDescriptionGetDimensions(fm.formatDescription)
            return dim.width == w && dim.height == h
                && fm.videoSupportedFrameRateRanges.contains { $0.minFrameRate <= Double(fps) && Double(fps) <= $0.maxFrameRate }
        }) {
            session.beginConfiguration()
            d.activeFormat = f
            d.activeVideoMinFrameDuration = CMTime(value: 1, timescale: fps)
            d.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: fps)
            session.commitConfiguration()
            videoOut.videoSettings = [
                kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_420YpCbCr8BiPlanarFullRange),
                kCVPixelBufferWidthKey as String: 960,
                kCVPixelBufferHeightKey as String: m == .video ? 540 : 720
            ] // monitör analizi küçük kareyle yapılır, 4K kayıt etkilenmez
            say(m == .video ? "\(p.name) fps" : "12 MP · 4 kare AI birleştirme")
        } else {
            say("Bu mod bu cihazda desteklenmiyor")
        }
        d.unlockForConfiguration()
        apply(c) // format değişince pozlama/odak/zoom sıfırlanabilir; yeniden uygula
    }

    func setMode(_ m: Mode) {
        guard !recording, !busy else { return }
        mode = m
        let p = preset, c = ctl
        q.async { self.applyMode(m, p, c) }
    }
    func nextPreset() {
        guard !recording, !busy else { return }
        preset = (preset + 1) % Self.presets.count
        setMode(.video)
    }

    private func orientation() -> AVCaptureVideoOrientation { // ana iş parçacığında çağır
        switch UIDevice.current.orientation {
        case .landscapeLeft: return .landscapeRight
        case .landscapeRight: return .landscapeLeft
        case .portraitUpsideDown: return .portraitUpsideDown
        default: return .portrait
        }
    }

    // MARK: Fotoğraf: 4 kare -> birleştir
    func shoot() {
        guard !busy, mode == .photo else { return }
        busy = true
        let c = ctl, li = liveISO, ls = liveShutter
        let useNight = night && auto && ev < 6.5 // karanlık sahne
        status = useNight ? "Gece modu: telefonu sabit tut…"
            : (hdr && !auto) ? "HDR için AE OTO gerekir, normal çekim yapılıyor…" : "Çekiliyor…"
        shotEV = ev
        shotLook = lookLevel
        let o = orientation()
        let useHDR = hdr && auto && !useNight && photoOut.maxBracketedCapturePhotoCount >= 3
        q.async {
            self.photoOut.connection(with: .video)?.videoOrientation = o
            self.frames = []
            self.meta = [:]
            self.nightShot = false
            if useNight, self.startNight(li, ls, c) { return }
            if useHDR { // -2 / 0 / +2 EV pozlama serisi
                let ex = [-2, 0, 2].map { AVCaptureAutoExposureBracketedStillImageSettings.autoExposureSettings(exposureTargetBias: Float($0)) }
                let s = AVCapturePhotoBracketSettings(rawPixelFormatType: 0,
                                                      processedFormat: [AVVideoCodecKey: AVVideoCodecType.jpeg],
                                                      bracketedSettings: ex)
                s.isHighResolutionPhotoEnabled = true
                self.hdrShot = true
                self.pending = 3
                self.photoOut.capturePhoto(with: s, delegate: self)
                return
            }
            self.hdrShot = false
            self.pending = 4
            for _ in 0..<4 {
                let s = AVCapturePhotoSettings()
                s.isHighResolutionPhotoEnabled = true
                s.photoQualityPrioritization = .speed
                s.flashMode = .off
                self.photoOut.capturePhoto(with: s, delegate: self)
            }
        }
    }

    /// Karanlıkta: daha uzun pozlu, düşük ISO'lu kareler çeker. Başarısızsa false döner, normal çekim sürer.
    private func startNight(_ li: Float, _ ls: Double, _ c: Ctl) -> Bool {
        guard let d = device, (try? d.lockForConfiguration()) != nil else { return false }
        let f = d.activeFormat
        let lo = f.videoSupportedFrameRateRanges.map { $0.minFrameRate }.min() ?? 1
        let hi = f.videoSupportedFrameRateRanges.map { $0.maxFrameRate }.max() ?? 30
        let light = Double(li) * ls // ISO x saniye: sahnenin ışık bütçesi
        var t = min(max(light / 200, 1.0 / 30), 1.0 / 8) // ISO ~200 hedefi, elde çekim için en çok 1/8 sn
        t = min(t, CMTimeGetSeconds(f.maxExposureDuration))
        t = min(max(t, 1 / hi), 1 / lo)
        let dur = CMTime(seconds: t, preferredTimescale: 1_000_000)
        let exp = CMTimeMinimum(CMTimeMaximum(dur, f.minExposureDuration), f.maxExposureDuration)
        let iso = min(max(Float(light / CMTimeGetSeconds(exp)), f.minISO), f.maxISO)
        d.activeVideoMaxFrameDuration = dur // önce uzun uç: geçersiz aralık oluşmaz
        d.activeVideoMinFrameDuration = dur
        nightActive = true
        shotCtl = c
        d.setExposureModeCustom(duration: exp, iso: iso) { [weak self] _ in
            self?.q.async { self?.fireNight() }
        }
        d.unlockForConfiguration()
        return true
    }

    private func fireNight() {
        nightShot = true
        hdrShot = false
        pending = Self.nightFrames
        for _ in 0..<Self.nightFrames {
            let s = AVCapturePhotoSettings()
            s.isHighResolutionPhotoEnabled = true
            s.photoQualityPrioritization = .speed
            s.flashMode = .off
            photoOut.capturePhoto(with: s, delegate: self)
        }
    }

    private func endNight() {
        nightActive = false
        if let d = device, (try? d.lockForConfiguration()) != nil {
            d.activeVideoMinFrameDuration = CMTime(value: 1, timescale: 30) // önce kısa uç
            d.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: 30)
            d.unlockForConfiguration()
        }
        if let c = shotCtl { apply(c) } // otomatik pozlama geri gelir
    }

    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        let img = photo.fileDataRepresentation()
            .flatMap { CIImage(data: $0, options: [.applyOrientationProperty: true]) }
            .map { $0.transformed(by: CGAffineTransform(translationX: -$0.extent.origin.x, y: -$0.extent.origin.y)) }
        let m = photo.metadata
        q.async {
            if self.meta.isEmpty { self.meta = m }
            if let img = img, error == nil { self.frames.append(img) }
            self.pending -= 1
            if self.nightShot, self.pending > 0 { self.say("Sabit tut… \(Self.nightFrames - self.pending)/\(Self.nightFrames)") }
            if self.pending == 0 { self.finishBurst() }
        }
    }

    private func finishBurst() {
        let fr = frames
        frames = []
        if nightActive { endNight() }
        say(nightShot ? "Gece birleştiriyor…" : "AI birleştiriyor…")
        DispatchQueue.global(qos: .userInitiated).async {
            guard let first = fr.first else { self.finish("Çekim başarısız"); return }
            let out: CIImage
            if self.nightShot { out = fr.count >= 2 ? Fusion.shared.night(fr) : first }
            else if self.hdrShot { out = fr.count == 3 ? Fusion.shared.fuseHDR(fr) : fr[fr.count / 2] }
            else { out = fr.count == 4 ? Fusion.shared.fuse(fr) : first }
            self.savePhoto(out)
        }
    }

    private func savePhoto(_ img: CIImage) {
        let out = Look.apply(img, level: shotLook, ev: shotEV).cropped(to: img.extent)
        guard let data = jpeg(out, meta: meta) else { finish("Kaydedilemedi"); return }
        let loc = lastLoc
        PHPhotoLibrary.shared().performChanges({
            let r = PHAssetCreationRequest.forAsset()
            r.location = loc
            r.addResource(with: .photo, data: data, options: nil)
        }) { ok, _ in self.finish(ok ? "Kaydedildi ✓" : "Galeriye yazılamadı") }
    }

    /// JPEG yazarken orijinal EXIF/TIFF bilgisi korunur (piksel yönü zaten dik, yön etiketi yazılmaz)
    private func jpeg(_ img: CIImage, meta: [String: Any]) -> Data? {
        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        guard let cg = Look.context.createCGImage(img, from: img.extent, format: .RGBA8, colorSpace: cs) else { return nil }
        let data = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(data, "public.jpeg" as CFString, 1, nil) else { return nil }
        var props: [String: Any] = [kCGImageDestinationLossyCompressionQuality as String: 0.95]
        for k in [kCGImagePropertyExifDictionary, kCGImagePropertyTIFFDictionary] {
            if var d = meta[k as String] as? [String: Any] {
                d.removeValue(forKey: kCGImagePropertyExifPixelXDimension as String)
                d.removeValue(forKey: kCGImagePropertyExifPixelYDimension as String)
                d.removeValue(forKey: kCGImagePropertyTIFFOrientation as String)
                props[k as String] = d
            }
        }
        CGImageDestinationAddImage(dest, cg, props as CFDictionary)
        return CGImageDestinationFinalize(dest) ? (data as Data) : nil
    }

    // MARK: Monitör (histogram / zebra / odak vurgusu), küçük kare üzerinde
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard monFlag != 0 else { return }
        frameN &+= 1
        guard frameN % 4 == 0, let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddressOfPlane(pb, 0) else { return }
        let w = CVPixelBufferGetWidthOfPlane(pb, 0), h = CVPixelBufferGetHeightOfPlane(pb, 0)
        let bpr = CVPixelBufferGetBytesPerRowOfPlane(pb, 0)
        guard w > 8, h > 8 else { return }
        let px = base.assumingMemoryBound(to: UInt8.self)
        let cols = Monitor.cols, rows = Monitor.rows
        var hist = [Float](repeating: 0, count: 64)
        var n = [Int](repeating: 0, count: cols * rows)
        var hot = [Int](repeating: 0, count: cols * rows)
        var edge = [Int](repeating: 0, count: cols * rows)
        for y in stride(from: 1, to: h - 1, by: 2) {
            for x in stride(from: 1, to: w - 1, by: 2) {
                let v = Int(px[y * bpr + x])
                hist[v >> 2] += 1
                let c = (y * rows / h) * cols + (x * cols / w)
                n[c] += 1
                if v >= 242 { hot[c] += 1 }
                if abs(v - Int(px[y * bpr + x + 1])) + abs(v - Int(px[(y + 1) * bpr + x])) > 24 { edge[c] += 1 }
            }
        }
        var m = Monitor()
        m.hist = hist
        m.zebra = (0..<n.count).map { n[$0] > 0 && hot[$0] * 3 > n[$0] }
        m.peak = (0..<n.count).map { n[$0] > 0 && edge[$0] * 5 > n[$0] }
        DispatchQueue.main.async { self.mon = m }
    }

    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        lastLoc = locations.last
    }

    // MARK: Video
    func toggleRecord() {
        guard mode == .video, !busy else { return }
        shotEV = ev
        shotLook = lookLevel
        let o = orientation()
        q.async {
            if self.movieOut.isRecording { self.movieOut.stopRecording(); return }
            let url = FileManager.default.temporaryDirectory
                .appendingPathComponent("v\(Int(Date().timeIntervalSince1970)).mov")
            if let c = self.movieOut.connection(with: .video) {
                c.videoOrientation = o
                if c.isVideoStabilizationSupported { c.preferredVideoStabilizationMode = .auto }
                if self.movieOut.availableVideoCodecTypes.contains(.hevc) {
                    self.movieOut.setOutputSettings([AVVideoCodecKey: AVVideoCodecType.hevc], for: c)
                }
            }
            self.movieOut.startRecording(to: url, recordingDelegate: self)
            DispatchQueue.main.async { self.recording = true; self.status = "● REC" }
        }
    }

    func fileOutput(_ output: AVCaptureFileOutput, didFinishRecordingTo url: URL,
                    from connections: [AVCaptureConnection], error: Error?) {
        DispatchQueue.main.async { self.recording = false }
        if let e = error as NSError?, (e.userInfo[AVErrorRecordingSuccessfullyFinishedKey] as? Bool) != true {
            finish("Kayıt hatası"); return
        }
        if shotLook > 0 { exportLook(url) } else { saveVideo(url, cleanup: [url]) }
    }

    // Look 4K60'ta canlı uygulanamaz (A11 yetmez); kayıttan sonra işlenir
    private func exportLook(_ url: URL) {
        DispatchQueue.main.async { self.busy = true; self.status = "Look uygulanıyor…" }
        let asset = AVAsset(url: url)
        let (lvl, ev0) = (shotLook, shotEV)
        let comp = AVMutableVideoComposition(asset: asset) { req in
            let e = req.sourceImage.extent
            req.finish(with: Look.apply(req.sourceImage.clampedToExtent(), level: lvl, ev: ev0, denoise: false).cropped(to: e),
                       context: nil)
        }
        let out = FileManager.default.temporaryDirectory.appendingPathComponent("look-\(url.lastPathComponent)")
        guard let ex = AVAssetExportSession(asset: asset, presetName: AVAssetExportPresetHEVCHighestQuality) else {
            saveVideo(url, cleanup: [url]); return
        }
        ex.videoComposition = comp
        ex.outputURL = out
        ex.outputFileType = .mov
        ex.exportAsynchronously {
            let ok = ex.status == .completed
            self.saveVideo(ok ? out : url, cleanup: ok ? [url, out] : [out])
        }
        DispatchQueue.main.async {
            Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self, weak ex] t in
                guard let ex = ex, ex.status != .completed, ex.status != .failed, ex.status != .cancelled else {
                    t.invalidate(); return
                }
                self?.status = "Look uygulanıyor %\(Int(ex.progress * 100))"
            }
        }
    }

    private func saveVideo(_ url: URL, cleanup: [URL]) {
        PHPhotoLibrary.shared().performChanges({
            let r = PHAssetChangeRequest.creationRequestForAssetFromVideo(atFileURL: url)
            r?.location = self.lastLoc
        }) { ok, _ in
            if ok { cleanup.forEach { try? FileManager.default.removeItem(at: $0) } } // başarısızsa dosya silinmez
            self.finish(ok ? "Video kaydedildi ✓" : "Galeriye yazılamadı (dosya korundu)")
        }
    }
}
