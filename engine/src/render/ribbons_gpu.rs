use crate::ribbons::{RibbonProfile, RibbonSnapshot, MAX_RIBBONS, MAX_RIBBON_POINTS};
use wgpu::util::DeviceExt;
const MAX_VERTICES: usize = MAX_RIBBONS * (MAX_RIBBON_POINTS - 1) * 12;
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    along: f32,
    color: [f32; 4],
    uv: [f32; 2],
    profile: u32,
    half_width: f32,
    tangent: [f32; 3],
    _pad: u32,
}
pub struct RibbonGpu {
    buffer: wgpu::Buffer,
    pipeline: wgpu::RenderPipeline,
    vertex_count: u32,
    scratch: Vec<Vertex>,
}
impl RibbonGpu {
    pub fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        scene: &wgpu::BindGroupLayout,
    ) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ribbon-fixed-vertices"),
            contents: &vec![0u8; MAX_VERTICES * std::mem::size_of::<Vertex>()],
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ribbon-shader"),
            source: wgpu::ShaderSource::Wgsl(WGSL.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ribbon-pipeline-layout"),
            bind_group_layouts: &[scene],
            push_constant_ranges: &[],
        });
        let attrs = wgpu::vertex_attr_array![0=>Float32x3,1=>Float32,2=>Float32x4,3=>Float32x2,4=>Uint32,5=>Float32,6=>Float32x3];
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("additive-world-ribbons"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &attrs,
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent::REPLACE,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: super::DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: super::DEPTH_COMPARE,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });
        Self {
            buffer,
            pipeline,
            vertex_count: 0,
            scratch: Vec::with_capacity(MAX_VERTICES),
        }
    }
    pub fn sync(&mut self, queue: &wgpu::Queue, ribbons: Vec<RibbonSnapshot>) {
        self.scratch.clear();
        for r in ribbons {
            self.append(r);
        }
        assert!(
            self.scratch.len() <= MAX_VERTICES,
            "ribbon vertex capacity exceeded"
        );
        self.vertex_count = self.scratch.len() as u32;
        if !self.scratch.is_empty() {
            queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&self.scratch));
        }
    }
    fn append(&mut self, r: RibbonSnapshot) {
        if r.points.len() < 2 {
            return;
        }
        let style = r.style;
        let profile = match style.profile() {
            RibbonProfile::Smooth => 0,
            RibbonProfile::Turbulent => 1,
            RibbonProfile::Jagged => 2,
            RibbonProfile::Organic => 3,
            RibbonProfile::Orbit => 4,
        };
        for (index, pair) in r.points.windows(2).enumerate() {
            let a = pair[0];
            let b = pair[1];
            let life_a = (1.0 - a.age_s / style.lifetime_s()).clamp(0.0, 1.0);
            let life_b = (1.0 - b.age_s / style.lifetime_s()).clamp(0.0, 1.0);
            let along_a = index as f32 / (r.points.len() - 1) as f32;
            let along_b = (index + 1) as f32 / (r.points.len() - 1) as f32;
            self.quad(
                a.position,
                b.position,
                along_a,
                along_b,
                life_a,
                life_b,
                style.width_m(),
                style,
                profile,
                0.0,
            );
            if style.cross_ribbon() {
                self.quad(
                    a.position,
                    b.position,
                    along_a,
                    along_b,
                    life_a,
                    life_b,
                    style.width_m(),
                    style,
                    profile,
                    1.0,
                );
            }
        }
    }
    #[allow(clippy::too_many_arguments)]
    fn quad(
        &mut self,
        a: glam::Vec3,
        b: glam::Vec3,
        ta: f32,
        tb: f32,
        la: f32,
        lb: f32,
        w: f32,
        s: crate::RibbonStyle,
        profile: u32,
        cross: f32,
    ) {
        let c0 = s.primary();
        let c1 = s.secondary();
        let color = |t: f32, l: f32| {
            let c = c0.lerp(c1, 1.0 - t);
            [
                c.r * s.emissive_intensity(),
                c.g * s.emissive_intensity(),
                c.b * s.emissive_intensity(),
                c.a * l,
            ]
        };
        let tangent = (b - a).normalize();
        let v = |p: glam::Vec3, t: f32, l: f32, side: f32| Vertex {
            position: p.into(),
            along: t,
            color: color(t, l),
            uv: [side, cross],
            profile,
            half_width: w,
            tangent: tangent.into(),
            _pad: 0,
        };
        self.scratch.extend_from_slice(&[
            v(a, ta, la, -w),
            v(b, tb, lb, -w),
            v(a, ta, la, w),
            v(a, ta, la, w),
            v(b, tb, lb, -w),
            v(b, tb, lb, w),
        ]);
    }
    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>, scene: &wgpu::BindGroup) {
        if self.vertex_count == 0 {
            return;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, scene, &[]);
        pass.set_vertex_buffer(0, self.buffer.slice(..));
        pass.draw(0..self.vertex_count, 0..1);
    }
}
const WGSL: &str = r#"
@group(0) @binding(0) var<uniform> scene:array<vec4<f32>,14>;
struct In{@location(0) position:vec3<f32>,@location(1) along:f32,@location(2) color:vec4<f32>,@location(3) uv:vec2<f32>,@location(4) profile:u32,@location(5) half_width:f32,@location(6) tangent:vec3<f32>};
struct Out{@builtin(position) clip:vec4<f32>,@location(0) color:vec4<f32>,@location(1) edge:f32,@location(2) along:f32,@location(3) profile:f32};
fn hash(p:vec2<f32>)->f32{return fract(sin(dot(p,vec2<f32>(127.1,311.7)))*43758.5453);}
@vertex fn vs_main(i:In)->Out{var o:Out;let camera_forward=normalize(cross(scene[10].xyz,scene[11].xyz));var side=cross(i.tangent,camera_forward);if(length(side)<0.0001){side=cross(i.tangent,scene[11].xyz);}side=normalize(side);if(i.uv.y>0.5){side=normalize(cross(i.tangent,side));}let taper=sin(clamp(i.along,0.0,1.0)*3.14159)*0.75+0.25;o.clip=mat4x4<f32>(scene[0],scene[1],scene[2],scene[3])*vec4<f32>(i.position+side*i.uv.x*taper,1.0);o.color=i.color;o.edge=abs(i.uv.x)/i.half_width;o.along=i.along;o.profile=f32(i.profile);return o;}
@fragment fn fs_main(i:Out)->@location(0) vec4<f32>{let n=hash(vec2<f32>(floor(i.along*83.0),i.profile*7.0));var erosion=0.045;if(i.profile>0.5){erosion=0.07+n*0.05;}if(i.profile>1.5&&i.profile<2.5){erosion=0.10+n*0.07;}let edge=1.0-smoothstep(1.0-erosion,1.0,i.edge);let pulse=0.78+0.22*sin(i.along*45.0+n*6.28);let alpha=i.color.a*edge*pulse;if(alpha<0.015){discard;}return vec4<f32>(i.color.rgb*(0.65+edge*0.9),alpha);}"#;
