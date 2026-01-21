use anyhow::{Result, Context};

use image::{ImageBuffer, Luma};
use vulkano::sync::GpuFuture;
use std::sync::Arc;
use reqwest::blocking::Client;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::device::Queue;
use vulkano::format::Format;
use vulkano::image::{Image, ImageUsage, ImageCreateInfo};
use vulkano::image::view::ImageView;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};

type Gray16Image = ImageBuffer<Luma<u16>, Vec<u16>>;

pub struct Terrain {
    origin: (u32, u32),
    zoom_level: u32,
    heightmap_cache: [Gray16Image; 16],
}

impl Terrain {
    pub fn new() -> Self {

        Terrain { 
            origin: (0, 0),
            zoom_level: 2,
            heightmap_cache: Default::default(),
        }
    }

    fn load_heightmaps_cpu(&mut self) -> Result<()> {
        let max_extent: u32 = 2_u32.pow(self.zoom_level)-1;
        let client = Client::new();

        let (o_x, o_y) = self.origin;
        for x in 0..4 {
            for y in 0..4 {
                let url = format!("http://localhost:3000/terrain/{}/{}/{}", self.zoom_level, o_x+x, o_y+y);
                eprintln!("Fetching tile: {}", url);

                let resp = client.get(url).send()
                    .context("Failed to connect to Tileserver")?;
                let bytes = resp.bytes().context("Failed to read bytes")?;

                let img: Gray16Image = image::load_from_memory(&bytes)
                    .context("Failed to decode image")?.to_luma16();
                self.heightmap_cache[(4*x + y) as usize] = img;
            }
        }
        Ok(())
    }
}

pub fn load_tile(
    at: (u32, u32, u32),
    allocator: Arc<StandardMemoryAllocator>,
    queue: Arc<Queue>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
) -> Arc<ImageView> {
    
    let url = format!("http://localhost:3000/terrain/{}/{}/{}", at.0, at.1, at.2);
    println!("Fetching tile: {}", url);
    
    let client = Client::new();
    let resp = client.get(url).send().expect("Failed to connect to Martin");
    let bytes = resp.bytes().expect("Failed to read bytes");

    // 2. Decode Image
    let img: Gray16Image = image::load_from_memory(&bytes).expect("Failed to decode image").to_luma16();
    let (width, height) = img.dimensions();

    // 3. Create Vulkan Image (Immutable = Optimized for Shader Reading)
    let upload_buffer = vulkano::buffer::Buffer::from_iter(
        allocator.clone(),
        vulkano::buffer::BufferCreateInfo {
            usage: vulkano::buffer::BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        img.into_raw(),
    ).unwrap();

    let image = Image::new(
        allocator.clone(),
        ImageCreateInfo {
            format: Format::R16_UNORM,
            usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
            extent: [width, height, 1],
            ..Default::default()
        },
        AllocationCreateInfo::default(),
    ).unwrap();

    let mut builder = AutoCommandBufferBuilder::primary(
        command_buffer_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    ).unwrap();

    builder.copy_buffer_to_image(vulkano::command_buffer::CopyBufferToImageInfo::buffer_image(
        upload_buffer,
        image.clone(),
    )).unwrap();

    let command_buffer = builder.build().unwrap();
    
    // Execute upload immediately and wait
    let future = vulkano::sync::now(queue.device().clone())
        .then_execute(queue.clone(), command_buffer).unwrap()
        .then_signal_fence_and_flush().unwrap();
        
    future.wait(None).unwrap();

    ImageView::new_default(image).unwrap()
}
