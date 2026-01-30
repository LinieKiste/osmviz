mod camera;
mod shaders;
mod terrain;
mod util;
mod vector_tile;

use std::sync::Arc;
use anyhow::{Result, Context};
use vulkano::{
    VulkanLibrary, buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer, allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo}}, command_buffer::allocator::StandardCommandBufferAllocator, descriptor_set::{
        DescriptorSet, WriteDescriptorSet, allocator::StandardDescriptorSetAllocator
    }, device::{
        Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue, QueueCreateInfo, QueueFlags, physical::PhysicalDeviceType
    }, format::Format, image::{Image, ImageCreateInfo, ImageType, ImageUsage, sampler::{Filter, Sampler, SamplerCreateInfo}, view::ImageView}, instance::{Instance, InstanceCreateFlags, InstanceCreateInfo, InstanceExtensions}, memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator}, pipeline::{
        GraphicsPipeline, Pipeline, graphics::{
            vertex_input::Vertex, viewport::Viewport
        }
    }, render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass}, swapchain::{
        Surface, Swapchain, SwapchainCreateInfo
    }, sync::{self, GpuFuture}
};
use winit::{
    application::ApplicationHandler, event::{KeyEvent, WindowEvent}, event_loop::{ActiveEventLoop, EventLoop}, keyboard::{KeyCode, PhysicalKey}, window::{Window, WindowId}
};

use crate::{camera::Camera, terrain::Terrain};

#[derive(BufferContents, Vertex)]
#[repr(C)]
pub struct TerrainVertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}

fn main() -> Result<()> {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new(&event_loop)?;

    event_loop.run_app(&mut app)
        .context("Failure in event loop")
}

struct App {
    instance: Arc<Instance>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    uniform_buffer_allocator: SubbufferAllocator,
    memory_allocator: Arc<StandardMemoryAllocator>,
    vertex_buffer: Subbuffer<[TerrainVertex]>,
    rcx: Option<RenderContext>,
    terrain: Terrain,
}

struct RenderContext {
    window: Arc<Window>,
    swapchain: Arc<Swapchain>,
    render_pass: Arc<RenderPass>,
    framebuffers: Vec<Arc<Framebuffer>>,
    pipeline: Arc<GraphicsPipeline>,
    viewport: Viewport,
    recreate_swapchain: bool,
    previous_frame_end: Option<Box<dyn GpuFuture>>,
    camera: Camera,
}

impl App {
    fn new(event_loop: &EventLoop<()>) -> Result<Self> {
        let library = VulkanLibrary::new().unwrap();
        let required_extensions = Surface::required_extensions(event_loop)
            .unwrap()
            .union(&InstanceExtensions{ext_debug_utils: true, ..InstanceExtensions::empty()});
        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: required_extensions,
                ..Default::default()
            },
        )
        .unwrap();

        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };
        let device_features = DeviceFeatures {
            tessellation_shader: true,
            fill_mode_non_solid: true,
            ..DeviceFeatures::empty()
        };
        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .unwrap()
            .filter(|p| p.supported_extensions().contains(&device_extensions))
            .filter(|p| p.supported_features().contains(&device_features))
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        q.queue_flags.intersects(QueueFlags::GRAPHICS)
                            && p.presentation_support(i as u32, event_loop).unwrap()
                    })
                    .map(|i| (p, i as u32))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            })
            .unwrap();

        println!(
            "Using device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type,
        );

        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                enabled_extensions: device_extensions,
                enabled_features: device_features,
                ..Default::default()
            },
        )?;

        let queue = queues.next().context("No available queue found")?;

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));
        let uniform_buffer_allocator = SubbufferAllocator::new(
            memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );

        let vertices = terrain::generate_patch_grid(4);
        let vertex_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vertices,
        )?;

        // load texture
        let terrain = Terrain::new(memory_allocator.clone(), queue.clone(), command_buffer_allocator.clone())?;

        Ok(App {
            instance,
            device,
            queue,
            descriptor_set_allocator,
            command_buffer_allocator,
            uniform_buffer_allocator,
            memory_allocator,
            vertex_buffer,
            rcx: None,
            terrain,
        })
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title("osmviz")).unwrap(),
        );
        let window_size = window.inner_size();

        let (swapchain, images) = self.create_swapchain(window.clone())
            .expect("Failed to create swapchain");

        let render_pass = vulkano::single_pass_renderpass!(
            self.device.clone(),
            attachments: {
                color: {
                    format: swapchain.image_format(),
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                },
                depth_stencil: {
                    format: Format::D16_UNORM,
                    samples: 1,
                    load_op: Clear,
                    store_op: DontCare,
                },
            },
            pass: {
                color: [color],
                depth_stencil: {depth_stencil},
            },
        ).unwrap();

        let framebuffers = window_size_dependent_setup(&images, &render_pass, &self.memory_allocator);

        let pipeline = self.create_tessellation_pipeline(&render_pass)
            .expect("failed to create tessellation pipeline");

        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window_size.into(),
            depth_range: 0.0..=1.0,
        };

        let previous_frame_end = Some(sync::now(self.device.clone()).boxed());

        let camera = Camera::new(1920., 1080.);

        self.rcx = Some(RenderContext {
            window,
            swapchain,
            render_pass,
            framebuffers,
            pipeline,
            viewport,
            recreate_swapchain: false,
            previous_frame_end,
            camera,
        });
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let rcx = self.rcx.as_mut().unwrap();

        rcx.camera.handle_input(&event);
        match event {
            WindowEvent::CloseRequested |
                WindowEvent::KeyboardInput {
                    event: KeyEvent {
                        physical_key: PhysicalKey::Code(KeyCode::Escape),
                    .. },
                .. } => {
                event_loop.exit();
            }
            WindowEvent::Resized(_) => {
                rcx.recreate_swapchain = true;
            }
            WindowEvent::RedrawRequested => {
                let window_size = rcx.window.inner_size();

                if window_size.width == 0 || window_size.height == 0 {
                    return;
                }

                rcx.previous_frame_end.as_mut().unwrap().cleanup_finished();

                if rcx.recreate_swapchain {
                    let (new_swapchain, new_images) = rcx
                        .swapchain
                        .recreate(SwapchainCreateInfo {
                            image_extent: window_size.into(),
                            ..rcx.swapchain.create_info()
                        })
                        .expect("failed to recreate swapchain");

                    rcx.swapchain = new_swapchain;
                    rcx.camera.resize(window_size.width as f32, window_size.height as f32);
                    rcx.framebuffers = window_size_dependent_setup(&new_images, &rcx.render_pass, &self.memory_allocator);
                    rcx.viewport.extent = window_size.into();
                    rcx.recreate_swapchain = false;
                }

                rcx.camera.update(0.016);

                self.terrain.set_origin(rcx.camera.get_position());
                self.terrain.update()
                    .expect("Terrain update failed");

                let camera_ubo = {
                    let uniform_data = rcx.camera.get_uniform_data();

                    let buffer = self.uniform_buffer_allocator.allocate_sized().unwrap();
                    *buffer.write().unwrap() = uniform_data;

                    buffer
                };
                let sampler = Sampler::new(
                    self.device.clone(),
                    SamplerCreateInfo {
                        min_filter: Filter::Linear,
                        mag_filter: Filter::Linear,
                        ..SamplerCreateInfo::default() // clamps to border
                },
                ).unwrap();

                let layout = &rcx.pipeline.layout().set_layouts()[0];
                let descriptor_set_camera = DescriptorSet::new(
                    self.descriptor_set_allocator.clone(),
                    layout.clone(),
                    [
                    WriteDescriptorSet::buffer(0, camera_ubo),
                    WriteDescriptorSet::image_view_sampler(1, self.terrain.get_heightmaps(), sampler.clone()),
                    WriteDescriptorSet::image_view_sampler(2, self.terrain.get_colormaps(), sampler),
                    ],
                    [],
                )
                .unwrap();

                self.render(descriptor_set_camera).expect("Rendering failed!");
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let rcx = self.rcx.as_mut().unwrap();
        rcx.window.request_redraw();
    }
}

#[derive(BufferContents, Vertex)]
#[repr(C)]
struct MyVertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}

/// This function is called once during initialization, then again whenever the window is resized.
fn window_size_dependent_setup(
    images: &[Arc<Image>],
    render_pass: &Arc<RenderPass>,
    memory_allocator: &Arc<StandardMemoryAllocator>,
) -> Vec<Arc<Framebuffer>> {
    let depth_buffer = ImageView::new_default(
        Image::new(
            memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::D16_UNORM,
                extent: images[0].extent(),
                usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT | ImageUsage::TRANSIENT_ATTACHMENT,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .unwrap(),
    )
    .unwrap();

    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();

            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view, depth_buffer.clone()],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}
