// YUYV (Y0 Cb Y1 Cr) → RGB, limited range. Runs once per camera frame into a
// W×H Rgba8Unorm texture that egui then draws as an image.

struct Params {
    y_offset: f32, y_scale: f32, c_offset: f32, _pad0: f32,
    r_v: f32, g_u: f32, g_v: f32, b_u: f32,
    width: u32, _pad1: u32, _pad2: u32, _pad3: u32,
};
@group(0) @binding(0) var yuyv: texture_2d<f32>;   // Rgba8Unorm, (width/2) x height, texel = Y0 U Y1 V
@group(0) @binding(1) var<uniform> p: Params;

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    // Full-screen triangle: (-1,-1), (3,-1), (-1,3).
    let xy = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(xy * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let x = u32(pos.x);
    let y = u32(pos.y);
    let tx = x >> 1u;
    let t = textureLoad(yuyv, vec2<u32>(tx, y), 0);
    var luma = t.r;
    var cb = t.g;
    var cr = t.a;
    if ((x & 1u) == 1u) {
        luma = t.b;
        // Odd pixel sits between two chroma samples: average with the next pair.
        let n = textureLoad(yuyv, vec2<u32>(min(tx + 1u, (p.width >> 1u) - 1u), y), 0);
        cb = 0.5 * (cb + n.g);
        cr = 0.5 * (cr + n.a);
    }
    let yy = (luma - p.y_offset) * p.y_scale;
    let u = cb - p.c_offset;
    let v = cr - p.c_offset;
    let rgb = vec3<f32>(yy + p.r_v * v, yy - p.g_u * u - p.g_v * v, yy + p.b_u * u);
    // egui textures hold gamma-encoded values in Rgba8Unorm: write R'G'B' as-is.
    return vec4<f32>(clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
