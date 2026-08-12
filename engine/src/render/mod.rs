mod clipmap;
mod gpu_mesh;
mod pipeline;
mod skinned;
mod terrain_pipeline;
mod water_pipeline;

use crate::mesh::InstanceRaw;
use crate::texture::{MaterialId, TextureId, WaterMaterialId};
use crate::world::{EntityId, SurfaceMaterialRef, World};
use clipmap::ClipmapRenderer;
use gpu_mesh::GpuMesh;
use pipeline::{create_pipelines, Pipelines, Uniforms};
use skinned::{create_skinned_pipelines, GpuSkinnedEntity, SkinnedPipelines};
use std::collections::HashMap;
use terrain_pipeline::{
    build_terrain_material, create_terrain_pipelines, upload_texture, GpuTerrainMaterial,
    GpuTexture, TerrainPipelines,
};
use water_pipeline::{
    build_water_material, create_water_pipelines, GpuWaterMaterial, WaterPipelines,
};
use winit::dpi::PhysicalSize;

pub struct Renderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipelines: Pipelines,
    terrain: TerrainPipelines,
    water: WaterPipelines,
    skinned: SkinnedPipelines,
    clipmap: Option<ClipmapRenderer>,
    depth_view: wgpu::TextureView,
    depth_texture: wgpu::Texture,
    gpu_meshes: HashMap<EntityId, GpuMesh>,
    gpu_skinned: HashMap<EntityId, GpuSkinnedEntity>,
    gpu_textures: HashMap<TextureId, GpuTexture>,
    gpu_materials: HashMap<MaterialId, GpuTerrainMaterial>,
    gpu_water_materials: HashMap<WaterMaterialId, GpuWaterMaterial>,
    /// Origin the terrain material phases were last written for.
    terrain_origin: crate::space::RenderOrigin,
    size: PhysicalSize<u32>,
}

impl Renderer {
    pub async fn new(window: std::sync::Arc<winit::window::Window>) -> Self {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });
        let surface = instance
            .create_surface(window.clone())
            .expect("failed to create surface");

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("no suitable GPU adapter");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("engine-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .expect("failed to create device");

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);

        // Uncapped present for stress testing / real FPS (Immediate → Mailbox → …).
        let present_mode = [
            wgpu::PresentMode::Immediate,
            wgpu::PresentMode::Mailbox,
            wgpu::PresentMode::FifoRelaxed,
            wgpu::PresentMode::Fifo,
        ]
        .into_iter()
        .find(|m| caps.present_modes.contains(m))
        .unwrap_or(wgpu::PresentMode::Fifo);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let pipelines = create_pipelines(&device, format);
        let terrain = create_terrain_pipelines(&device, format, &pipelines.bind_layout);
        let water = create_water_pipelines(&device, format, &pipelines.bind_layout);
        let skinned = create_skinned_pipelines(&device, format, &pipelines.bind_layout);
        let (depth_texture, depth_view) = create_depth(&device, config.width, config.height);

        Self {
            surface,
            device,
            queue,
            config,
            pipelines,
            terrain,
            water,
            skinned,
            clipmap: None,
            depth_view,
            depth_texture,
            gpu_meshes: HashMap::new(),
            gpu_skinned: HashMap::new(),
            gpu_textures: HashMap::new(),
            gpu_materials: HashMap::new(),
            gpu_water_materials: HashMap::new(),
            terrain_origin: crate::space::RenderOrigin::default(),
            size,
        }
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }
        self.size = new_size;
        self.config.width = new_size.width;
        self.config.height = new_size.height;
        self.surface.configure(&self.device, &self.config);
        let (depth_texture, depth_view) =
            create_depth(&self.device, self.config.width, self.config.height);
        self.depth_texture = depth_texture;
        self.depth_view = depth_view;
    }

    pub fn sync_world(&mut self, world: &World) {
        self.sync_textures_and_materials(world);

        if let Some(proc) = world.proc_terrain() {
            let format = self.config.format;
            match self.clipmap.as_mut() {
                Some(clip) => clip.ensure_config(&self.device, format, &proc.config),
                None => {
                    self.clipmap = Some(ClipmapRenderer::new(
                        &self.device,
                        format,
                        proc.config.clone(),
                    ));
                }
            }
            if let Some(clip) = self.clipmap.as_mut() {
                let aspect = self.config.width as f32 / self.config.height.max(1) as f32;
                let vp = world.camera.view_projection(aspect);
                let light_dir = world.light.direction.normalize_or_zero();
                clip.prepare(
                    &self.queue,
                    vp,
                    light_dir,
                    world.light.ambient,
                    world.light.color,
                    world.camera.eye,
                    proc,
                );
            }
        } else {
            self.clipmap = None;
        }

        let live: std::collections::HashSet<EntityId> =
            world.entities().map(|(id, _)| id).collect();
        self.gpu_meshes.retain(|id, _| live.contains(id));

        for (id, entity) in world.entities() {
            let instances: Vec<InstanceRaw> = if entity.instanced {
                entity
                    .instances
                    .iter()
                    .map(|m| InstanceRaw::from_matrix(entity.transform * *m))
                    .collect()
            } else {
                vec![InstanceRaw::from_matrix(entity.transform)]
            };
            // An empty instance buffer cannot be created, so an instanced entity
            // with nothing placed either drops to a zero draw or stays unuploaded
            // until it has work.
            if instances.is_empty() {
                if let Some(gpu) = self.gpu_meshes.get_mut(&id) {
                    gpu.clear_instances();
                }
                continue;
            }

            match self.gpu_meshes.get_mut(&id) {
                Some(gpu) => {
                    // Rebuild if topology changed (index count / vertex count).
                    if gpu.vertex_count != entity.mesh().vertex_count()
                        || gpu.index_count != entity.mesh().index_count()
                    {
                        *gpu = GpuMesh::upload(&self.device, entity.mesh(), &instances);
                    } else {
                        gpu.update_instances(&self.device, &self.queue, &instances);
                    }
                }
                None => {
                    self.gpu_meshes
                        .insert(id, GpuMesh::upload(&self.device, entity.mesh(), &instances));
                }
            }
        }

        let live_anim: std::collections::HashSet<EntityId> =
            world.animated_entities().map(|(id, _)| *id).collect();
        self.gpu_skinned.retain(|id, _| live_anim.contains(id));

        for (id, anim) in world.animated_entities() {
            let joints = anim.animator.joint_matrices();
            match self.gpu_skinned.get_mut(id) {
                Some(gpu) => gpu.update(&self.queue, anim.transform, &joints),
                None => {
                    self.gpu_skinned.insert(
                        *id,
                        GpuSkinnedEntity::upload(
                            &self.device,
                            &self.skinned.joint_bind_layout,
                            &anim.animator.model.meshes,
                            anim.transform,
                            &joints,
                        ),
                    );
                }
            }
        }
    }

    /// 3D-only frame (no overlay pass).
    #[allow(dead_code)]
    pub fn render(&mut self, world: &World) -> Result<(), wgpu::SurfaceError> {
        self.render_with(world, |_, _, _, _| {})
    }

    /// Render the 3D world, then invoke `after` for overlay passes (egui).
    pub fn render_with(
        &mut self,
        world: &World,
        after: impl FnOnce(&wgpu::Device, &wgpu::Queue, &mut wgpu::CommandEncoder, &wgpu::TextureView),
    ) -> Result<(), wgpu::SurfaceError> {
        self.write_uniforms(world);

        let frame = self.surface.get_current_texture()?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame-encoder"),
            });
        self.encode_pass(&mut encoder, &view, world);
        after(&self.device, &self.queue, &mut encoder, &view);
        self.queue.submit(std::iter::once(encoder.finish()));
        frame.present();
        Ok(())
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    #[allow(dead_code)]
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.config.format
    }

    /// Render the current world to a PNG (for demo QA / automation).
    pub fn capture_png(&mut self, world: &World, path: impl AsRef<std::path::Path>) {
        self.write_uniforms(world);

        let width = self.config.width.max(1);
        let height = self.config.height.max(1);
        let format = self.config.format;

        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("capture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let bpp = 4u32;
        let unpadded = width * bpp;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded = unpadded.div_ceil(align) * align;
        let buffer_size = (padded * height) as u64;
        let output = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("capture-buf"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("capture-encoder"),
            });
        self.encode_pass(&mut encoder, &view, world);
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &output,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = output.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        self.device
            .poll(wgpu::PollType::Wait)
            .expect("device poll failed");
        rx.recv().expect("map channel closed").expect("map failed");

        let data = slice.get_mapped_range();
        let mut rgba = vec![0u8; (width * height * 4) as usize];
        let is_bgra = matches!(
            format,
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
        );
        for y in 0..height as usize {
            let src = &data[y * padded as usize..y * padded as usize + unpadded as usize];
            let dst = &mut rgba[y * unpadded as usize..(y + 1) * unpadded as usize];
            if is_bgra {
                for (i, px) in src.chunks_exact(4).enumerate() {
                    dst[i * 4] = px[2];
                    dst[i * 4 + 1] = px[1];
                    dst[i * 4 + 2] = px[0];
                    dst[i * 4 + 3] = px[3];
                }
            } else {
                dst.copy_from_slice(src);
            }
        }
        drop(data);
        output.unmap();

        image::save_buffer(path.as_ref(), &rgba, width, height, image::ColorType::Rgba8)
            .unwrap_or_else(|e| panic!("failed to save screenshot: {e}"));
    }

    fn write_uniforms(&self, world: &World) {
        let aspect = self.config.width as f32 / self.config.height.max(1) as f32;
        let vp = world.camera.view_projection(aspect);
        let light_dir = world.light.direction.normalize_or_zero();
        let eye = world.camera.eye;
        let uniforms = Uniforms {
            view_proj: vp.to_cols_array_2d(),
            light_dir: [light_dir.x, light_dir.y, light_dir.z],
            ambient: world.light.ambient,
            light_color: world.light.color.into(),
            _pad: 0.0,
            eye: [eye.x, eye.y, eye.z],
            time: world.time(),
        };
        self.queue.write_buffer(
            &self.pipelines.uniform_buf,
            0,
            bytemuck::bytes_of(&uniforms),
        );
    }

    fn encode_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        world: &World,
    ) {
        let clear = world.clear_color.to_vec3();
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("main-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: clear.x as f64,
                        g: clear.y as f64,
                        b: clear.z as f64,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        // GPU procgen terrain (depth-writing land), then entity meshes.
        if let Some(clip) = self.clipmap.as_ref() {
            clip.draw_land(&mut pass);
        }

        pass.set_bind_group(0, &self.pipelines.bind_group, &[]);

        // Opaque untextured, then terrain-textured, then skinned / transparent.
        pass.set_pipeline(&self.pipelines.opaque);
        for (id, entity) in world.entities() {
            if entity.material.is_some() {
                continue;
            }
            let Some(gpu) = self.gpu_meshes.get(&id) else {
                continue;
            };
            if gpu.opaque_index_count == 0 {
                continue;
            }
            pass.set_vertex_buffer(0, gpu.vertex_buf.slice(..));
            pass.set_vertex_buffer(1, gpu.instance_buf.slice(..));
            pass.set_index_buffer(gpu.index_buf.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(
                0..gpu.opaque_index_count as u32,
                0,
                0..gpu.instance_count as u32,
            );
        }

        pass.set_pipeline(&self.terrain.opaque);
        for (id, entity) in world.entities() {
            let Some(SurfaceMaterialRef::Terrain(mid)) = entity.material else {
                continue;
            };
            let Some(gpu) = self.gpu_meshes.get(&id) else {
                continue;
            };
            let Some(mat) = self.gpu_materials.get(&mid) else {
                continue;
            };
            if gpu.opaque_index_count == 0 {
                continue;
            }
            pass.set_bind_group(1, &mat.bind_group, &[]);
            pass.set_vertex_buffer(0, gpu.vertex_buf.slice(..));
            pass.set_vertex_buffer(1, gpu.instance_buf.slice(..));
            pass.set_index_buffer(gpu.index_buf.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(
                0..gpu.opaque_index_count as u32,
                0,
                0..gpu.instance_count as u32,
            );
        }

        pass.set_pipeline(&self.skinned.opaque);
        for gpu in self.gpu_skinned.values() {
            pass.set_bind_group(1, &gpu.joint_bind, &[]);
            pass.set_vertex_buffer(1, gpu.instance_buf.slice(..));
            for mesh in &gpu.meshes {
                pass.set_vertex_buffer(0, mesh.vertex_buf.slice(..));
                pass.set_index_buffer(mesh.index_buf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..mesh.index_count, 0, 0..1);
            }
        }
        // Restore scene bind group for subsequent passes.
        pass.set_bind_group(0, &self.pipelines.bind_group, &[]);

        pass.set_pipeline(&self.pipelines.transparent);
        for (id, entity) in world.entities() {
            if matches!(entity.material, Some(SurfaceMaterialRef::Water(_))) {
                continue;
            }
            let Some(gpu) = self.gpu_meshes.get(&id) else {
                continue;
            };
            if gpu.opaque_index_count >= gpu.index_count {
                continue;
            }
            pass.set_vertex_buffer(0, gpu.vertex_buf.slice(..));
            pass.set_vertex_buffer(1, gpu.instance_buf.slice(..));
            pass.set_index_buffer(gpu.index_buf.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(
                gpu.opaque_index_count as u32..gpu.index_count as u32,
                0,
                0..gpu.instance_count as u32,
            );
        }

        // Water sheets last, so everything standing in them is already in the
        // colour buffer to blend against.
        pass.set_pipeline(&self.water.blend);
        for (id, entity) in world.entities() {
            let Some(SurfaceMaterialRef::Water(mid)) = entity.material else {
                continue;
            };
            let Some(gpu) = self.gpu_meshes.get(&id) else {
                continue;
            };
            let Some(mat) = self.gpu_water_materials.get(&mid) else {
                continue;
            };
            pass.set_bind_group(1, &mat.bind_group, &[]);
            pass.set_vertex_buffer(0, gpu.vertex_buf.slice(..));
            pass.set_vertex_buffer(1, gpu.instance_buf.slice(..));
            pass.set_index_buffer(gpu.index_buf.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(
                gpu.opaque_index_count as u32..gpu.index_count as u32,
                0,
                0..gpu.instance_count as u32,
            );
        }

        // Translucent clipmap water after meshes so the walker can occlude shorelines.
        if let Some(clip) = self.clipmap.as_ref() {
            clip.draw_water(&mut pass);
        }
    }

    pub fn size(&self) -> PhysicalSize<u32> {
        self.size
    }

    fn sync_textures_and_materials(&mut self, world: &World) {
        for (id, cpu) in world.textures() {
            if self.gpu_textures.contains_key(id) {
                continue;
            }
            let gpu = upload_texture(&self.device, &self.queue, cpu.width, cpu.height, &cpu.rgba);
            self.gpu_textures.insert(*id, gpu);
        }
        self.gpu_textures
            .retain(|id, _| world.textures().contains_key(id));

        for (id, mat) in world.materials() {
            if self.gpu_materials.contains_key(id) {
                continue;
            }
            let grass = self
                .gpu_textures
                .get(&mat.desc.grass)
                .expect("terrain grass texture missing on GPU");
            let sand = self
                .gpu_textures
                .get(&mat.desc.sand)
                .expect("terrain sand texture missing on GPU");
            let rock = self
                .gpu_textures
                .get(&mat.desc.rock)
                .expect("terrain rock texture missing on GPU");
            let gpu = build_terrain_material(
                &self.device,
                &self.terrain.mat_bind_layout,
                &self.terrain.sampler,
                [grass, sand, rock],
                &mat.desc,
                world.render_origin(),
            );
            self.gpu_materials.insert(*id, gpu);
        }
        self.gpu_materials
            .retain(|id, _| world.materials().contains_key(id));

        for (id, mat) in world.water_materials() {
            if self.gpu_water_materials.contains_key(id) {
                continue;
            }
            let gpu = build_water_material(
                &self.device,
                &self.water.mat_bind_layout,
                &mat.desc,
                world.render_origin(),
            );
            self.gpu_water_materials.insert(*id, gpu);
        }
        self.gpu_water_materials
            .retain(|id, _| world.water_materials().contains_key(id));

        // Rebase moved render space under the terrain; re-phase the tiling so
        // the ground texture and the waves stay locked to world coordinates.
        if world.render_origin() != self.terrain_origin {
            self.terrain_origin = world.render_origin();
            for mat in self.gpu_materials.values() {
                mat.write_origin(&self.queue, self.terrain_origin);
            }
            for mat in self.gpu_water_materials.values() {
                mat.write_origin(&self.queue, self.terrain_origin);
            }
        }
    }
}

fn create_depth(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}
