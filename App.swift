import SwiftUI
import AVFoundation

@main
struct ProCamApp: App {
    var body: some Scene { WindowGroup { ContentView() } }
}

final class PreviewView: UIView {
    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
    var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    var onTap: ((CGPoint) -> Void)?
    var onPinch: ((CGFloat) -> Void)?

    override init(frame: CGRect) {
        super.init(frame: frame)
        addGestureRecognizer(UITapGestureRecognizer(target: self, action: #selector(tap(_:))))
        addGestureRecognizer(UIPinchGestureRecognizer(target: self, action: #selector(pinch(_:))))
    }
    required init?(coder: NSCoder) { fatalError("init(coder:) yok") }

    @objc func tap(_ g: UITapGestureRecognizer) {
        onTap?(previewLayer.captureDevicePointConverted(fromLayerPoint: g.location(in: self)))
    }
    @objc func pinch(_ g: UIPinchGestureRecognizer) {
        if g.state == .changed { onPinch?(g.scale); g.scale = 1 }
    }
}

struct Preview: UIViewRepresentable {
    let session: AVCaptureSession
    let onTap: (CGPoint) -> Void
    let onPinch: (CGFloat) -> Void

    func makeUIView(context: Context) -> PreviewView {
        let v = PreviewView()
        v.previewLayer.session = session
        v.previewLayer.videoGravity = .resizeAspectFill
        v.onTap = onTap
        v.onPinch = onPinch
        return v
    }
    func updateUIView(_ v: PreviewView, context: Context) {
        v.onTap = onTap
        v.onPinch = onPinch
    }
}

struct ContentView: View {
    @StateObject private var cam = CameraManager()
    static let looks = ["Kapalı", "Doğal", "Sinema"]
    static let mons = ["Kapalı", "Histogram", "Zebra", "Odak", "Hepsi"]

    private func chip(_ t: String, _ action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(t).font(.system(size: 12, weight: .semibold))
                .padding(.horizontal, 10).padding(.vertical, 6)
                .background(Color.black.opacity(0.5)).clipShape(Capsule())
        }
    }

    private func row<V: BinaryFloatingPoint>(_ label: String, _ value: Binding<V>, _ range: ClosedRange<V>) -> some View
    where V.Stride: BinaryFloatingPoint {
        HStack {
            Text(label).font(.footnote.monospaced()).frame(width: 84, alignment: .leading)
            Slider(value: value, in: range)
        }
    }

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            Preview(session: cam.session, onTap: { cam.focus(at: $0) }, onPinch: { cam.pinch($0) })
                .ignoresSafeArea()
            MonitorOverlay(mon: cam.mon, level: cam.monLevel, aspect: cam.previewAspect)
                .ignoresSafeArea()
                .allowsHitTesting(false)
            VStack(spacing: 10) {
                HStack {
                    Text(cam.info).font(.system(.footnote, design: .monospaced))
                    Spacer()
                    chip("Look: " + Self.looks[cam.lookLevel]) { cam.lookLevel = (cam.lookLevel + 1) % 3 }
                }
                if cam.monLevel == 1 || cam.monLevel == 4 {
                    HistogramView(hist: cam.mon.hist).frame(height: 44).background(Color.black.opacity(0.35))
                }
                Spacer()
                Text(cam.status).font(.footnote)
                VStack(spacing: 4) {
                    if !cam.auto {
                        row("ISO \(Int(cam.iso))", $cam.iso, 50...1600)
                        row("1/\(Int(cam.shutterDenom))", $cam.shutterPos, 0...1)
                    }
                    if !cam.af { row("Odak", $cam.lens, 0...1) }
                    if !cam.awb { row("\(Int(cam.kelvin))K", $cam.kelvin, 2500...8000) }
                }
                .onChange(of: cam.iso) { _ in cam.push() }
                .onChange(of: cam.shutterPos) { _ in cam.push() }
                .onChange(of: cam.lens) { _ in cam.push() }
                .onChange(of: cam.kelvin) { _ in cam.push() }
                HStack(spacing: 8) {
                    chip(cam.auto ? "AE OTO" : "AE MANUEL") { cam.toggleAE() }
                    chip(cam.af ? "AF OTO" : "MF") { cam.toggleAF() }
                    chip(cam.awb ? "WB OTO" : "WB MANUEL") { cam.toggleWB() }
                }
                HStack(spacing: 8) {
                    if cam.mode == .photo { chip(cam.hdr ? "HDR AÇIK" : "HDR KAPALI") { cam.hdr.toggle() } }
                    if cam.mode == .photo { chip(cam.night ? "🌙 OTO" : "🌙 KAPALI") { cam.night.toggle() } }
                    chip("Monitör: " + Self.mons[cam.monLevel]) { cam.monLevel = (cam.monLevel + 1) % 5 }
                }
                HStack(spacing: 24) {
                    Group {
                        if cam.mode == .video {
                            chip(CameraManager.presets[cam.preset].name) { cam.nextPreset() }
                        } else {
                            Color.clear
                        }
                    }
                    .frame(width: 90)
                    Button {
                        if cam.mode == .photo { cam.shoot() } else { cam.toggleRecord() }
                    } label: {
                        ZStack {
                            Circle().stroke(Color.white, lineWidth: 4).frame(width: 74, height: 74)
                            Circle()
                                .fill(cam.mode == .video ? Color.red : Color.white)
                                .frame(width: cam.recording ? 30 : 60, height: cam.recording ? 30 : 60)
                        }
                    }
                    .opacity(cam.busy ? 0.4 : 1)
                    chip(cam.mode == .photo ? "FOTO" : "VİDEO") {
                        cam.setMode(cam.mode == .photo ? .video : .photo)
                    }
                    .frame(width: 90)
                }
            }
            .padding()
        }
        .foregroundColor(.white)
        .preferredColorScheme(.dark)
        .statusBarHidden()
        .onAppear { cam.start() }
    }
}

struct HistogramView: View {
    let hist: [Float]
    var body: some View {
        Canvas { ctx, size in
            guard let mx = hist.max(), mx > 0 else { return }
            let bw = size.width / CGFloat(hist.count)
            for (i, v) in hist.enumerated() {
                let h = CGFloat((v / mx).squareRoot()) * size.height
                ctx.fill(Path(CGRect(x: CGFloat(i) * bw, y: size.height - h, width: bw - 0.5, height: h)),
                         with: .color(Color.white.opacity(0.85)))
            }
        }
    }
}

/// Zebra (kırmızı: patlayan ışık) ve odak vurgusu (yeşil: net bölge); önizlemeyle aynı kırpmayla hizalanır
struct MonitorOverlay: View {
    let mon: CameraManager.Monitor
    let level: Int
    let aspect: CGFloat
    var body: some View {
        Canvas { ctx, size in
            let cols = CameraManager.Monitor.cols, rows = CameraManager.Monitor.rows
            guard level >= 2, mon.zebra.count == cols * rows, mon.peak.count == cols * rows else { return }
            let wide = size.width / size.height > aspect
            let iw = wide ? size.width : size.height * aspect
            let ih = wide ? size.width / aspect : size.height
            let ox = (size.width - iw) / 2, oy = (size.height - ih) / 2
            let cw = iw / CGFloat(cols), ch = ih / CGFloat(rows)
            for r in 0..<rows {
                for c in 0..<cols {
                    let i = r * cols + c
                    let rect = CGRect(x: ox + CGFloat(c) * cw, y: oy + CGFloat(r) * ch, width: cw, height: ch)
                    if (level == 2 || level == 4) && mon.zebra[i] {
                        ctx.fill(Path(rect), with: .color(Color.red.opacity(0.45)))
                    } else if (level == 3 || level == 4) && mon.peak[i] {
                        ctx.fill(Path(rect), with: .color(Color.green.opacity(0.4)))
                    }
                }
            }
        }
    }
}
