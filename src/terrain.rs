use crate::TerrainVertex;
use anyhow::{Result, Context};
use lru::LruCache;

use std::{num::NonZeroUsize, sync::Arc};
use reqwest::blocking::Client;
use image::{ImageBuffer, Luma};
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer}, command_buffer::{AutoCommandBufferBuilder, BufferImageCopy, CommandBufferUsage, CopyBufferToImageInfo, allocator::StandardCommandBufferAllocator}, device::Queue, format::Format, image::{Image, ImageCreateInfo, ImageSubresourceLayers, ImageType, ImageUsage, view::{ImageView, ImageViewCreateInfo, ImageViewType}}, memory::allocator::{AllocationCreateInfo, DeviceLayout, MemoryTypeFilter, StandardMemoryAllocator}, pipeline::graphics::vertex_input::Vertex, sync::GpuFuture
};

type Gray16Image = ImageBuffer<Luma<u16>, Vec<u16>>;

#[derive(BufferContents, Vertex, Default)]
#[repr(C)]
pub struct TileInstance {
    #[format(R32G32_SFLOAT)]
    pub world_offset: [f32; 2], // Where to draw this tile (Meters)
    
    #[format(R32_UINT)]
    pub texture_layer: u32,     // Which texture slot to use (0..15)
}

pub struct Terrain {
    needs_rebuild: bool,
    // cpu resources
    origin: (u32, u32),
    zoom_level: u32,
    heightmap_cache: LruCache<(u32, u32), Gray16Image>,
    // gpu resources
    gpu_resources: TerrainGpuResources,
}

struct TerrainGpuResources {
    memory_allocator: Arc<StandardMemoryAllocator>,
    queue: Arc<Queue>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,

    instance_buffer: Subbuffer<[TileInstance]>,

    heightmap_array: Arc<Image>,
    heightmap_array_view: Arc<ImageView>,
}

/// While the terrain is a square grid, this controls the grid radius
const TERRAIN_RADIUS: usize = 10;
/// Resolution of a single heightmap image, retrieved from the server
const HEIGHTMAP_SIZE: (usize, usize) = (256, 256);
/// How many images the image array on the GPU can hold
const IMAGE_BUFFER_SIZE: usize = 101;
/// Diameter of a single tile
const TILE_SIZE_METERS: f32 = 1.0;

impl Terrain {
    pub fn new(
        memory_allocator: Arc<StandardMemoryAllocator>,
        queue: Arc<Queue>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        ) -> Result<Self> {
        let gpu_resources = TerrainGpuResources::new(memory_allocator, queue, command_buffer_allocator)?;

        let mut terrain = Terrain { 
            needs_rebuild: true,
            // CPU
            origin: (135, 89),
            zoom_level: 8,
            heightmap_cache: LruCache::new(NonZeroUsize::new(128).unwrap()),
            // GPU
            gpu_resources,
        };

        terrain.update()?;

        Ok(terrain)
    }

    pub fn get_heightmaps(&self) -> Arc<ImageView> {
        self.gpu_resources.heightmap_array_view.clone()
    }
    pub fn get_instance_buffer(&self) -> Subbuffer<[TileInstance]> {
        self.gpu_resources.instance_buffer.clone()
    }

    /// Update heightmap array and instance buffer
    pub fn update(&mut self) -> Result<()> {
        self.load_heightmaps_cpu()
            .context("Failed to cache heightmaps")?;

        let mut instances = Vec::with_capacity(TERRAIN_RADIUS.pow(2));
        let mut heightmap_data: Vec<u16> = Vec::new();

        let (o_x, o_y) = self.origin;
        for (i_x, x) in (o_x..o_x+TERRAIN_RADIUS as u32).enumerate() {
            for (i_y, y) in (o_y..o_y+TERRAIN_RADIUS as u32).enumerate() {
                let offset_x = i_x as f32 * TILE_SIZE_METERS;
                let offset_y = i_y as f32 * TILE_SIZE_METERS;

                // Simple 1:1 mapping of Grid Index -> Texture Layer
                let layer_index = (i_x * TERRAIN_RADIUS + i_y) as u32;

                instances.push(TileInstance {
                    world_offset: [offset_x, offset_y],
                    texture_layer: layer_index,
                });

                heightmap_data.append(
                    &mut self.heightmap_cache
                    .get_mut(&(x, y))
                    .context("Tried to use uncached image")?
                    .as_raw().clone()
                    );
            }
        }

        self.gpu_resources.upload_heightmaps(heightmap_data)
            .context("Failed to upload heightmaps to GPU")?;

        self.gpu_resources.instance_buffer = Buffer::from_iter(
            self.gpu_resources.memory_allocator.clone(),
            BufferCreateInfo { 
                usage: BufferUsage::VERTEX_BUFFER, 
                ..Default::default() 
            },
            AllocationCreateInfo { 
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE, 
                ..Default::default() 
            },
            instances,
        ).unwrap().into();

        self.needs_rebuild = false;

        Ok(())
    }

    fn load_heightmaps_cpu(&mut self) -> Result<()> {
        let max_extent: u32 = 2_u32.pow(self.zoom_level)-1;
        let client = Client::new();

        let (o_x, o_y) = self.origin;
        for x in o_x..o_x+TERRAIN_RADIUS as u32 {
            for y in o_y..o_y+TERRAIN_RADIUS as u32 {
                if self.heightmap_cache.get(&(x, y)) != None { continue; } // skips if already loaded, but marks as recently used

                let url = format!("http://localhost:3000/terrain/{}/{}/{}", self.zoom_level, x, y);
                eprintln!("Fetching tile: {}", url);

                let resp = client.get(url).send()
                    .context("Failed to connect to Tileserver")?;
                let bytes = resp.bytes().context("Failed to read bytes")?;

                let img: Gray16Image = image::load_from_memory(&bytes)
                    .context("Failed to decode image")?.to_luma16();
                self.heightmap_cache.put((x, y), img);
            }
        }
        Ok(())
    }
}

impl TerrainGpuResources {

    pub fn new(
        memory_allocator: Arc<StandardMemoryAllocator>,
        queue: Arc<Queue>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        ) -> Result<Self> {

        // image
        let (width, height) = HEIGHTMAP_SIZE;
        let heightmap_array = Image::new(
            memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::R16_UNORM,
                usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                extent: [width as u32, height as u32, 1],
                array_layers: IMAGE_BUFFER_SIZE as u32,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )?;

        let heightmap_array_view = ImageView::new(
            heightmap_array.clone(),
            ImageViewCreateInfo {
                view_type: ImageViewType::Dim2dArray,
                ..ImageViewCreateInfo::from_image(&heightmap_array)
            },
        )?;

        // Garbage init
        let instance_buffer = Buffer::new_slice(
            memory_allocator.clone(),
            BufferCreateInfo { 
                usage: BufferUsage::VERTEX_BUFFER, 
                ..Default::default() 
            },
            AllocationCreateInfo { 
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE, 
                ..Default::default() 
            },
            (TERRAIN_RADIUS*TERRAIN_RADIUS) as u64,
        ).unwrap().into();


        Ok(TerrainGpuResources {
            memory_allocator,
            queue,
            command_buffer_allocator,

            instance_buffer,

            heightmap_array,
            heightmap_array_view,
        })
    }

    pub fn upload_heightmaps( &mut self, data: Vec<u16>) -> Result<()> {
        let heightmap_upload_buffer = Buffer::from_iter(
            self.memory_allocator.clone(),
            vulkano::buffer::BufferCreateInfo {
                usage: vulkano::buffer::BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            data.into_iter()
        )?;

        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;

        let copy_buffer_to_image_info = CopyBufferToImageInfo {
            regions: [BufferImageCopy {
                        image_subresource: ImageSubresourceLayers{
                            array_layers: 0..(TERRAIN_RADIUS*TERRAIN_RADIUS) as u32, // IMPORTANT: only copy layers we can actuall fill
                            ..self.heightmap_array.subresource_layers()
                        },
                        image_extent: self.heightmap_array.extent(),
                        ..Default::default()
                    }]
                    .into(),
            ..CopyBufferToImageInfo::buffer_image(
                heightmap_upload_buffer,
                self.heightmap_array.clone(),
            )
        };

        builder.copy_buffer_to_image(copy_buffer_to_image_info)?;

        let command_buffer = builder.build()?;

        // Execute upload immediately and wait
        let future = vulkano::sync::now(self.queue.device().clone())
            .then_execute(self.queue.clone(), command_buffer)?
            .then_signal_fence_and_flush()?;

        future.wait(None)?;

        Ok(())
    }
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
