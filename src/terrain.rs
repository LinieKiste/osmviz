use crate::TerrainVertex;
use anyhow::{Result, Context};

use std::sync::Arc;
use reqwest::blocking::Client;
use image::{EncodableLayout, ImageBuffer, Luma};
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer}, command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, allocator::StandardCommandBufferAllocator}, device::Queue, format::Format, image::{Image, ImageCreateInfo, ImageType, ImageUsage, view::{ImageView, ImageViewCreateInfo, ImageViewType}}, memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator}, pipeline::graphics::vertex_input::Vertex, sync::GpuFuture
};

type Gray16Image = ImageBuffer<Luma<u16>, Vec<u16>>;

#[derive(BufferContents, Vertex)]
#[repr(C)]
pub struct TileInstance {
    #[format(R32G32_SFLOAT)]
    pub world_offset: [f32; 2], // Where to draw this tile (Meters)
    
    #[format(R32_UINT)]
    pub texture_layer: u32,     // Which texture slot to use (0..15)
}

pub struct Terrain {
    origin: (u32, u32),
    zoom_level: u32,
    heightmap_cache: [Gray16Image; TERRAIN_RADIUS.pow(2) as usize],
    tile_size_meters: f32,
}

const TERRAIN_RADIUS: u32 = 15;
const HEIGHTMAP_SIZE: (usize, usize) = (256, 256);
impl Terrain {
    pub fn new() -> Self {
        Terrain { 
            origin: (135, 89),
            zoom_level: 8,
            heightmap_cache: core::array::from_fn(|_i| Default::default()),
            tile_size_meters: 1.0,
        }
    }

    /// Generates the Instance Buffer for the current 4x4 grid
    pub fn create_instance_buffer(
        &self,
        allocator: Arc<StandardMemoryAllocator>,
    ) -> Subbuffer<[TileInstance]> {
        let mut instances = Vec::with_capacity(TERRAIN_RADIUS.pow(2) as usize);

        // Simple 4x4 Grid loop relative to origin
        for x in 0..TERRAIN_RADIUS {
            for y in 0..TERRAIN_RADIUS {
                // Calculate position in world meters
                // We draw relative to (0,0), so the first tile is at 0, second at 2000, etc.
                let offset_x = x as f32 * self.tile_size_meters;
                let offset_y = y as f32 * self.tile_size_meters;

                // Simple 1:1 mapping of Grid Index -> Texture Layer
                let layer_index = (x * TERRAIN_RADIUS + y) as u32;

                instances.push(TileInstance {
                    world_offset: [offset_x, offset_y],
                    texture_layer: layer_index,
                });
            }
        }

        Buffer::from_iter(
            allocator,
            BufferCreateInfo { 
                usage: BufferUsage::VERTEX_BUFFER, 
                ..Default::default() 
            },
            AllocationCreateInfo { 
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE, 
                ..Default::default() 
            },
            instances,
        ).unwrap().into()
    }

    fn load_heightmaps_cpu(&mut self) -> Result<()> {
        let max_extent: u32 = 2_u32.pow(self.zoom_level)-1;
        let client = Client::new();

        let (o_x, o_y) = self.origin;
        for x in 0..TERRAIN_RADIUS {
            for y in 0..TERRAIN_RADIUS {
                let url = format!("http://localhost:3000/terrain/{}/{}/{}", self.zoom_level, o_x+x, o_y+y);
                eprintln!("Fetching tile: {}", url);

                let resp = client.get(url).send()
                    .context("Failed to connect to Tileserver")?;
                let bytes = resp.bytes().context("Failed to read bytes")?;

                let img: Gray16Image = image::load_from_memory(&bytes)
                    .context("Failed to decode image")?.to_luma16();
                self.heightmap_cache[(TERRAIN_RADIUS*x + y) as usize] = img;
            }
        }
        Ok(())
    }

    pub fn upload_heightmaps(
        &mut self,
        allocator: Arc<StandardMemoryAllocator>,
        queue: Arc<Queue>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        ) -> Result<Arc<ImageView>> {

        self.load_heightmaps_cpu()?;
        let (width, height) = self.heightmap_cache[0].dimensions();

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
            self.heightmap_cache.iter().map(|x| x.as_raw().clone()) // TODO: remove clone
            .flatten()
            .collect::<Vec<u16>>()
            ,
        )?;

        let image = Image::new(
            allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::R16_UNORM,
                usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                extent: [width, height, 1],
                array_layers: TERRAIN_RADIUS.pow(2),
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )?;

        let mut builder = AutoCommandBufferBuilder::primary(
            command_buffer_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;

        builder.copy_buffer_to_image(vulkano::command_buffer::CopyBufferToImageInfo::buffer_image(
                upload_buffer,
                image.clone(),
        ))?;

        let command_buffer = builder.build()?;

        // Execute upload immediately and wait
        let future = vulkano::sync::now(queue.device().clone())
            .then_execute(queue.clone(), command_buffer)?
            .then_signal_fence_and_flush()?;

        future.wait(None)?;

        let view = ImageView::new(
            image.clone(),
            ImageViewCreateInfo {
                view_type: ImageViewType::Dim2dArray,
                ..ImageViewCreateInfo::from_image(&image)
            },
        )?;

        Ok(view)
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
            image_type: ImageType::Dim2d,
            format: Format::R16_UNORM,
            usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
            extent: [width, height, 1],
            array_layers: 32,
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

pub fn generate_patch_grid(subdivisions: u32) -> Vec<TerrainVertex> {
    let mut vertices = Vec::new();
    let step = 1.0 / subdivisions as f32;

    for z in 0..subdivisions {
        for x in 0..subdivisions {
            let x0 = x as f32 * step;
            let z0 = z as f32 * step;
            let x1 = x0 + step;
            let z1 = z0 + step;

            // Add 4 vertices for this patch (CCW winding)
            // Note: Our shader expects flat X/Z coords packed into 'position'
            vertices.push(TerrainVertex { position: [x0, z0] }); // TL
            vertices.push(TerrainVertex { position: [x1, z0] }); // TR
            vertices.push(TerrainVertex { position: [x1, z1] }); // BR
            vertices.push(TerrainVertex { position: [x0, z1] }); // BL
        }
    }
    vertices
}
