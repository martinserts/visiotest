//! GPU video path: raw YUYV bytes → `Rgba8Unorm` texture via one fragment-shader pass.
//!
//! All GPU resources are created once in `VideoTexture::new`. Per camera frame the CPU
//! does exactly one `queue.write_texture` of the raw bytes plus one tiny render pass.

use std::time::{Duration, Instant};

use eframe::egui;
use eframe::egui_wgpu::RenderState;
use eframe::wgpu;
use eframe::wgpu::util::DeviceExt;

use crate::capture::Frame;
use crate::color::ColorMatrix;

const SHADER: &str = include_str!("yuyv.wgsl");
/// Format of both textures. egui requires `Rgba8Unorm` for native textures and treats
/// the stored values as gamma-encoded.
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

pub struct VideoTexture {
    width: u32,
    height: u32,
    yuyv: wgpu::Texture,
    rgba_view: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,
    texture_id: egui::TextureId,
}

impl VideoTexture {
    pub fn new(rs: &RenderState, width: u32, height: u32, matrix: ColorMatrix) -> Self {
        let device = &rs.device;
        let texture = |label, w, usage| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: w,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: FORMAT,
                usage,
                view_formats: &[],
            })
        };
        // One texel = one pixel pair, Y0 U Y1 V.
        let yuyv = texture(
            "yuyv",
            width / 2,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        );
        let rgba = texture(
            "video_rgba",
            width,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let yuyv_view = yuyv.create_view(&wgpu::TextureViewDescriptor::default());
        let rgba_view = rgba.create_view(&wgpu::TextureViewDescriptor::default());

        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("yuv_params"),
            contents: bytemuck::bytes_of(&matrix.params(width)),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("yuyv_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        // No sampler: filtering would blend Y0 with U.
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("yuyv_bind_group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&yuyv_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params.as_entire_binding(),
                },
            ],
        });

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("yuyv.wgsl"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("yuyv_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("yuyv_to_rgba"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(), // TriangleList, no culling
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let texture_id = rs.renderer.write().register_native_texture(
            device,
            &rgba_view,
            wgpu::FilterMode::Linear,
        );

        Self {
            width,
            height,
            yuyv,
            rgba_view,
            bind_group,
            pipeline,
            texture_id,
        }
    }

    /// Returns None (and uploads nothing) if the frame's width/height differ from the texture.
    /// Some(d): d = CPU time of write_texture + conversion-pass submit.
    pub fn upload(&mut self, rs: &RenderState, frame: &Frame) -> Option<Duration> {
        let l = frame.layout;
        if l.width != self.width || l.height != self.height {
            return None;
        }
        let start = Instant::now();
        let len = l.stride as usize * l.height as usize;
        rs.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.yuyv,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &frame.buffer.buffer()[..len],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(l.stride),
                rows_per_image: Some(l.height),
            },
            // Copy only the image width; row padding beyond it is skipped.
            wgpu::Extent3d {
                width: l.width / 2,
                height: l.height,
                depth_or_array_layers: 1,
            },
        );

        let mut encoder = rs
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("yuyv_to_rgba"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("yuyv_to_rgba"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.rgba_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        // Same queue egui submits to later this frame, so ordering is guaranteed.
        rs.queue.submit([encoder.finish()]);
        Some(start.elapsed())
    }

    pub fn texture_id(&self) -> egui::TextureId {
        self.texture_id
    }

    /// Width × height in pixels (for aspect fit).
    pub fn size(&self) -> egui::Vec2 {
        egui::vec2(self.width as f32, self.height as f32)
    }
}

#[cfg(test)]
mod tests {
    use eframe::wgpu::naga;

    #[test]
    fn shader_is_valid() {
        let module = naga::front::wgsl::parse_str(super::SHADER).expect("WGSL parses");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("WGSL validates");
    }
}
