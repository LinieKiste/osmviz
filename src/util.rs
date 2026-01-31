use crate::vector_tile::BuildingVertex;
use crate::App;
use crate::TerrainVertex;
use crate::shaders::*;
use crate::terrain::TileInstance;
use std::sync::Arc;
use anyhow::{Result, Context};

use glam::{UVec3, Vec2, Vec3, Vec3Swizzles};
use vulkano::Validated;
use vulkano::VulkanError;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::command_buffer::CommandBufferUsage;
use vulkano::command_buffer::RenderPassBeginInfo;
use vulkano::descriptor_set::DescriptorSet;
use vulkano::swapchain::SwapchainPresentInfo;
use vulkano::swapchain::acquire_next_image;
use vulkano::{
    image::{Image, ImageUsage},
    sync::{self, GpuFuture},
    pipeline::{
        Pipeline, PipelineBindPoint,
        DynamicState, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo,
        graphics::{
            depth_stencil::{DepthState, DepthStencilState},
            GraphicsPipelineCreateInfo, color_blend::{ColorBlendAttachmentState, ColorBlendState}, input_assembly::{InputAssemblyState, PrimitiveTopology}, multisample::MultisampleState, rasterization::{PolygonMode, RasterizationState, CullMode}, tessellation::TessellationState, vertex_input::{Vertex, VertexDefinition}, viewport::ViewportState
        }, layout::PipelineDescriptorSetLayoutCreateInfo
    }, render_pass::{RenderPass, Subpass}, swapchain::{Surface, Swapchain, SwapchainCreateInfo}
};
use winit::window::Window;

pub const WORLD_ORIGIN: Vec2 = Vec2::new(21_720., 14_330.); // in km, relative to mapbox tile 0,0

impl App {
    pub fn create_swapchain(&self, window: Arc<Window>) -> Result<(Arc<Swapchain>, Vec<Arc<Image>>)> {
        let surface = Surface::from_window(self.instance.clone(), window.clone())?;
        let surface_capabilities = self
            .device
            .physical_device()
            .surface_capabilities(&surface, Default::default())?;

        let (image_format, _) = self
            .device
            .physical_device()
            .surface_formats(&surface, Default::default())?[0];

        Ok(Swapchain::new(
                self.device.clone(),
                surface,
                SwapchainCreateInfo {
                    min_image_count: surface_capabilities.min_image_count.max(2),
                    image_format,
                    image_extent: window.inner_size().into(),
                    image_usage: ImageUsage::COLOR_ATTACHMENT,
                    composite_alpha: surface_capabilities
                        .supported_composite_alpha
                        .into_iter()
                        .next().context("No fitting surface capability found")?,
                        ..Default::default()
                },
        )?)
    }

    pub fn create_tessellation_pipeline(&self, render_pass: &Arc<RenderPass>) -> Result<Arc<GraphicsPipeline>> {
        let vs = vs::load(self.device.clone())?
            .entry_point("main")
            .unwrap();
        let tcs = tcs::load(self.device.clone())?
            .entry_point("main")
            .unwrap();
        let tes = tes::load(self.device.clone())?
            .entry_point("main")
            .unwrap();
        let fs = fs::load(self.device.clone())?
            .entry_point("main")
            .unwrap();
        let vertex_input_state =
            [TerrainVertex::per_vertex(), TileInstance::per_instance()]
            .definition(&vs)?;
        let stages = [
            PipelineShaderStageCreateInfo::new(vs),
            PipelineShaderStageCreateInfo::new(tcs),
            PipelineShaderStageCreateInfo::new(tes),
            PipelineShaderStageCreateInfo::new(fs),
        ];

        let layout = PipelineLayout::new(
            self.device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
            .into_pipeline_layout_create_info(self.device.clone())?
        )?;
        let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

        Ok(GraphicsPipeline::new(
                self.device.clone(),
                None,
                GraphicsPipelineCreateInfo {
                    stages: stages.into_iter().collect(),
                    vertex_input_state: Some(vertex_input_state),
                    input_assembly_state: Some(InputAssemblyState {
                        topology: PrimitiveTopology::PatchList,
                        ..Default::default()
                    }),
                    tessellation_state: Some(TessellationState {
                        // Use a patch_control_points of 3, because we want to convert one
                        // *triangle* into lots of little ones. A value of 4 would convert a
                        // *rectangle* into lots of little triangles.
                        patch_control_points: 4,
                        ..Default::default()
                    }),
                    viewport_state: Some(ViewportState::default()),
                    rasterization_state: Some(RasterizationState {
                        polygon_mode: PolygonMode::Fill,
                        cull_mode: CullMode::None,
                        ..Default::default()
                    }),
                    depth_stencil_state: Some(DepthStencilState {
                        depth: Some(DepthState::simple()),
                        ..Default::default()
                    }),
                    multisample_state: Some(MultisampleState::default()),
                    color_blend_state: Some(ColorBlendState::with_attachment_states(
                            subpass.num_color_attachments(),
                            ColorBlendAttachmentState::default(),
                    )),
                    dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                    subpass: Some(subpass.into()),
                    ..GraphicsPipelineCreateInfo::layout(layout)
                },
                )?)
    }

    pub fn create_building_pipeline(&self, render_pass: &Arc<RenderPass>) -> Result<Arc<GraphicsPipeline>> {
        let vs = building_vs::load(self.device.clone())?.entry_point("main").unwrap();
        let fs = building_fs::load(self.device.clone())?.entry_point("main").unwrap();

        let vertex_input_state = BuildingVertex::per_vertex()
            .definition(&vs)?;

        let stages = [
            PipelineShaderStageCreateInfo::new(vs),
            PipelineShaderStageCreateInfo::new(fs),
        ];

        let layout = PipelineLayout::new(
            self.device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(self.device.clone())?
        )?;

        let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

        Ok(GraphicsPipeline::new(
            self.device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(vertex_input_state),
                input_assembly_state: Some(InputAssemblyState {
                    topology: PrimitiveTopology::TriangleList, // Standard triangles
                    ..Default::default()
                }),
                viewport_state: Some(ViewportState::default()),
                rasterization_state: Some(RasterizationState {
                    polygon_mode: PolygonMode::Fill,
                    cull_mode: CullMode::None, // Cull back-faces
                    front_face: vulkano::pipeline::graphics::rasterization::FrontFace::CounterClockwise,
                    ..Default::default()
                }),
                depth_stencil_state: Some(DepthStencilState {
                    depth: Some(DepthState::simple()), // Read/Write depth buffer
                    ..Default::default()
                }),
                multisample_state: Some(MultisampleState::default()),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    subpass.num_color_attachments(),
                    ColorBlendAttachmentState::default(),
                )),
                dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                subpass: Some(subpass.into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )?)
    }

    pub fn render(&mut self, descriptor_sets: &[Arc<DescriptorSet>]) -> Result<()> {
        let rcx = self.rcx.as_mut().unwrap();

        let (image_index, suboptimal, acquire_future) = match acquire_next_image(
            rcx.swapchain.clone(),
            None,
        )
            .map_err(Validated::unwrap)
            {
                Ok(r) => r,
                Err(VulkanError::OutOfDate) => {
                    rcx.recreate_swapchain = true;
                    return Ok(());
                }
                Err(e) => panic!("failed to acquire next image: {e}"),
            };

        if suboptimal {
            rcx.recreate_swapchain = true;
        }

        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;

        let instance_buffer = self.terrain.get_instance_buffer();
        let building_vertex_buffer = self.terrain.get_building_vertices();

        // draw terrain
        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![
                        Some([0.0, 0.0, 0.0, 1.0].into()),
                        Some(1f32.into()),
                    ],
                    ..RenderPassBeginInfo::framebuffer(
                        rcx.framebuffers[image_index as usize].clone(),
                    )
                },
                Default::default(),
            )?
            .set_viewport(0, [rcx.viewport.clone()].into_iter().collect())?
            .bind_pipeline_graphics(rcx.terrain_pipeline.clone())?
            .bind_descriptor_sets(PipelineBindPoint::Graphics, rcx.terrain_pipeline.layout().clone(), 0, descriptor_sets[0].clone())?
            .bind_vertex_buffers(0, (self.vertex_buffer.clone(), instance_buffer.clone()))?;
        unsafe { builder.draw(self.vertex_buffer.len() as u32, instance_buffer.len() as u32, 0, 0) }?;

        // draw buildings
        builder
            .bind_pipeline_graphics(rcx.building_pipeline.clone())?
            .bind_descriptor_sets(PipelineBindPoint::Graphics, rcx.building_pipeline.layout().clone(), 0, descriptor_sets[1].clone())? // Re-use camera set (set 0)
            .bind_vertex_buffers(0, building_vertex_buffer.clone())?;
        unsafe { builder.draw(building_vertex_buffer.len() as u32, 1, 0, 0) }?;

        builder.end_render_pass(Default::default())?;

        let command_buffer = builder.build()?;
        let future = rcx
            .previous_frame_end
            .take().context("previous frame did not end")?
            .join(acquire_future)
            .then_execute(self.queue.clone(), command_buffer)?
            .then_swapchain_present(
                self.queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(
                    rcx.swapchain.clone(),
                    image_index,
                ),
            )
            .then_signal_fence_and_flush();

        match future.map_err(Validated::unwrap) {
            Ok(future) => {
                rcx.previous_frame_end = Some(future.boxed());
            }
            Err(VulkanError::OutOfDate) => {
                rcx.recreate_swapchain = true;
                rcx.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
            }
            Err(e) => {
                println!("failed to flush future: {e}");
                rcx.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
            }
        };
        Ok(())
    }
}

/// Returns tile index in the format `(x, y, z)`
pub fn world_to_tile_idx(pos: Vec2, zoom: u32) -> UVec3 {
    (pos / zoom_to_dist(zoom)).as_uvec2().extend(zoom)
}

/// Returns world coords in the format `(x, y, z)`
pub fn tile_to_world_coords(pos: UVec3) -> Vec2 {
    pos.as_vec3().xy() * zoom_to_dist(pos.z) 
}

/// Calculates the side length of a patch given a zoom level
pub fn zoom_to_dist(z: u32) -> f32 {
    debug_assert!(z < 23 && z >= 2, "Zoom level {} is out of bounds", z);
    const EARTH_CIRCUMFERENCE: f32 = 40_960.0; // in km, approximately. Increased so tile length is 10 at zoom 12
    EARTH_CIRCUMFERENCE / (2_f32.powf(z as f32))
}

pub fn zoom_from_height(height: f32) -> u32 {
    (5. + 3000./(height+400.)).clamp(4., 18.) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_zoom_to_dist() {
        assert_eq!(zoom_to_dist(12), 10.)
    }
}

