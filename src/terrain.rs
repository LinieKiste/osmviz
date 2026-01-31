use crate::{TerrainVertex, util, vector_tile::{BuildingVertex, VectorTile}};
use anyhow::{Result, Context};
use glam::{UVec2, UVec3, Vec2, Vec3, Vec3Swizzles};
use lru::LruCache;

use std::{num::NonZeroUsize, sync::Arc};
use reqwest::blocking::Client;
use image::{DynamicImage, ImageBuffer, Luma, RgbaImage};
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer}, command_buffer::{AutoCommandBufferBuilder, BufferImageCopy, CommandBufferUsage, CopyBufferToImageInfo, allocator::StandardCommandBufferAllocator}, device::Queue, format::Format, image::{Image, ImageCreateInfo, ImageSubresourceLayers, ImageType, ImageUsage, view::{ImageView, ImageViewCreateInfo, ImageViewType}}, memory::allocator::{AllocationCreateInfo, DeviceLayout, MemoryTypeFilter, StandardMemoryAllocator}, pipeline::graphics::vertex_input::Vertex, sync::GpuFuture
};

pub type Gray16Image = ImageBuffer<Luma<u16>, Vec<u16>>;

#[derive(BufferContents, Vertex, Default)]
#[repr(C)]
pub struct TileInstance {
    #[format(R32G32_SFLOAT)]
    pub world_offset: [f32; 2], // Where to draw this tile (Meters)
    
    #[format(R32_UINT)]
    pub texture_layer: u32,     // Which texture slot to use (0..15)
    #[format(R32_SFLOAT)]
    pub scale: f32,
}

type CacheEntry = (Gray16Image, RgbaImage, Option<VectorTile>);
pub struct Terrain {
    needs_rebuild: bool,

    // cpu resources
    origin: Vec2,
    zoom: u32,
    heightmap_cache: LruCache<UVec3, CacheEntry>,
    clipmap: Clipmap,
    prev_top_left: UVec3,

    // gpu resources
    gpu_resources: TerrainGpuResources,
}

struct TerrainGpuResources {
    memory_allocator: Arc<StandardMemoryAllocator>,
    queue: Arc<Queue>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,

    instance_buffer: Subbuffer<[TileInstance]>,
    building_vertices: Subbuffer<[BuildingVertex]>,

    heightmap_array: Arc<Image>,
    heightmap_array_view: Arc<ImageView>,
    color_array: Arc<Image>,
    color_array_view: Arc<ImageView>,
}

/// Resolution of a single heightmap image, retrieved from the server
const HEIGHTMAP_SIZE: (usize, usize) = (256, 256);
/// How many images the image array on the GPU can hold
const IMAGE_BUFFER_SIZE: usize = 512;
const USE_API: bool = true;

impl Terrain {
    pub fn new(
        memory_allocator: Arc<StandardMemoryAllocator>,
        queue: Arc<Queue>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        ) -> Result<Self> {
        let gpu_resources = TerrainGpuResources::new(memory_allocator, queue, command_buffer_allocator)?;
        let origin = Vec2::new(21_720., 14_330.);
        let zoom = 12;

        let mut terrain = Terrain { 
            needs_rebuild: true,

            // CPU
            origin,
            zoom,
            heightmap_cache: LruCache::new(NonZeroUsize::new(1000).unwrap()),
            clipmap: Clipmap::new(origin, 12, 1),
            prev_top_left: UVec3::ZERO,

            // GPU
            gpu_resources,
        };

        terrain.update()?;

        Ok(terrain)
    }

    pub fn get_building_vertices(&mut self) -> Subbuffer<[BuildingVertex]> {
        // TODO: remove Create a simple debug quad hovering at Y=100
        // let building_vertices = vec![
        //     BuildingVertex { position: [00.0, 20.0, 0.0] },
        //     BuildingVertex { position: [10.0, 20.0, 0.0] },
        //     BuildingVertex { position: [00.0, 40.0, 0.0] }, // Just a triangle for now
        // ];
        self.gpu_resources.building_vertices.clone()
    }

    pub fn get_heightmaps(&self) -> Arc<ImageView> {
        self.gpu_resources.heightmap_array_view.clone()
    }
    pub fn get_colormaps(&self) -> Arc<ImageView> {
        self.gpu_resources.color_array_view.clone()
    }
    pub fn get_instance_buffer(&self) -> Subbuffer<[TileInstance]> {
        self.gpu_resources.instance_buffer.clone()
    }
    pub fn set_origin(&mut self, coords: Vec3) {
        let rings = self.clipmap.rings.len();
        let zoom = util::zoom_from_height(coords.z);
        let terrain_zoom = util::zoom_from_height(coords.z).min(MAX_ZOOM_TERRAIN);
        let top_left_tile = Clipmap::find_starting_tile(coords.xy(), terrain_zoom, rings as u8);

        if self.prev_top_left != top_left_tile {
            self.prev_top_left = top_left_tile;
            self.origin = util::tile_to_world_coords(top_left_tile);
            self.zoom = zoom;
            self.needs_rebuild = true;
        }
    }

    /// Update heightmap array and instance buffer
    pub fn update(&mut self) -> Result<()> {
        if !self.needs_rebuild { return Ok(()) }

        self.clipmap.rebuild(self.origin, self.zoom.min(MAX_ZOOM_TERRAIN));
        self.load_heightmaps_cpu()
            .context("Failed to cache heightmaps")?;

        let mut instances = Vec::new();
        let mut heightmap_data: Vec<u16> = Vec::new();
        let mut color_data: Vec<u8> = Vec::new();
        let mut building_vertices: Vec<BuildingVertex> = Vec::new();

        const CLIPMAP_CENTER_RADIUS: u32 = 4;
        for (x, row) in self.clipmap.center.iter().enumerate() {
            for (y, coords) in row.iter().enumerate() {
                let scale = util::zoom_to_dist(self.zoom);
                let offset = self.origin + Vec2::new(y as f32, x as f32) * scale - util::WORLD_ORIGIN;

                let layer_index = (x as u32 * CLIPMAP_CENTER_RADIUS + y as u32) as u32;

                instances.push(TileInstance {
                    world_offset: offset.into(),
                    texture_layer: layer_index,
                    scale,
                });

                let entry = self.heightmap_cache
                    .get_mut(coords)
                    .context(format!("Tried to use uncached entry at ({}, {}, {})", x, y, self.zoom))?;

                heightmap_data.append(&mut entry.0.as_raw().clone());
                color_data.append(&mut entry.1.as_raw().clone());
                if let Some(v) = &entry.2 {
                    println!("We have vertices.");
                    building_vertices.append(&mut v.clone().mesh);
                }
            }
        }
        let mut top_left_inner = self.origin;
        for (r_idx, ring) in self.clipmap.rings.iter().enumerate() {
                let scale = util::zoom_to_dist(self.zoom-(r_idx + 1) as u32);
                for (tile_idx, coords) in ring.iter().enumerate() {
                    let offset = Clipmap::map_ring_offset(tile_idx, top_left_inner, scale) - util::WORLD_ORIGIN;

                    let layer_index = (CLIPMAP_CENTER_RADIUS.pow(2) as usize + r_idx*12 + tile_idx) as u32;

                    instances.push(TileInstance {
                        world_offset: offset.into(),
                        texture_layer: layer_index,
                        scale,
                    });

                    let entry = self.heightmap_cache
                        .get_mut(coords)
                        .context(format!("Tried to use uncached entry at {}", coords))?;

                    heightmap_data.append(&mut entry.0.as_raw().clone());
                    color_data.append(&mut entry.1.as_raw().clone());
                    if let Some(v) = &entry.2 {
                        building_vertices.append(&mut v.clone().mesh);
                    }
                }
                top_left_inner -= Vec2::ONE*scale;
        }

        let layers = 16 + self.clipmap.rings.len() * 12;
        self.gpu_resources.upload_image_array(heightmap_data, layers as u32, self.gpu_resources.heightmap_array.clone())
            .context("Failed to upload heightmaps to GPU")?;
        self.gpu_resources.upload_image_array(color_data, layers as u32, self.gpu_resources.color_array.clone())
            .context("Failed to upload color maps to GPU")?;

        dbg!(building_vertices.len());
        if building_vertices.len() > 0 {
            self.gpu_resources.building_vertices = Buffer::from_iter(
                self.gpu_resources.memory_allocator.clone(),
                BufferCreateInfo { 
                    usage: BufferUsage::VERTEX_BUFFER, 
                    ..Default::default() 
                },
                AllocationCreateInfo { 
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE, 
                    ..Default::default() 
                },
                building_vertices,
            )?.into();
        }

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
        )?.into();

        self.needs_rebuild = false;

        Ok(())
    }

    fn load_heightmaps_cpu(&mut self) -> Result<()> {
        let client = &Client::new();

        eprintln!("Fetching tiles");
        for UVec3 { x, y, z } in self.clipmap.center.iter().flatten()
            .chain(self.clipmap.rings.iter().flatten()) {
            if self.heightmap_cache.get(&UVec3::new(*x, *y, *z)).is_some() { continue; } // skips if already loaded, but marks as recently used

            // load heightmap
            let url = format!("http://localhost:3000/terrain/{}/{}/{}", z, x, y);
            eprintln!("Fetching tile @ {} {} {}", x, y, z);
            let resp = client.get(url).send()
                .context("Failed to connect to Tileserver")?;
            let bytes = resp.bytes().context("Failed to read bytes")?;
            let heightmap: Gray16Image = image::load_from_memory(&bytes)
                .unwrap_or( DynamicImage::new_luma16(HEIGHTMAP_SIZE.0 as u32, HEIGHTMAP_SIZE.1 as u32))
                .to_luma16();

            // load color
            // the api expects the ordering z, y, x for some reason
            let color: RgbaImage = match USE_API {
                true => {
                    let url = format!("https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless_3857/default/GoogleMapsCompatible/{}/{}/{}.jpg", z, y, x);
                    let resp = client.get(url).send()
                        .context("Failed to connect to EOX server")?;
                    let bytes = resp.bytes().context("Failed to read bytes")?;
                    image::load_from_memory(&bytes)
                        .context("Failed to decode image")?.to_rgba8()
                },
                false => {
                    DynamicImage::ImageLuma16(heightmap.clone()).into_rgba8()
                }
            };

            // load vector tile
            let tile = VectorTile::new(client, UVec3::new(*x,*y,*z), &heightmap)
                .ok();

            self.heightmap_cache.put(UVec3::new(*x, *y, *z), (heightmap, color, tile));
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

        // heightmap image
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

        // color image
        let (width, height) = HEIGHTMAP_SIZE;
        let color_array = Image::new(
            memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::R8G8B8A8_UNORM,
                usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                extent: [width as u32, height as u32, 1],
                array_layers: IMAGE_BUFFER_SIZE as u32,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )?;

        let color_array_view = ImageView::new(
            color_array.clone(),
            ImageViewCreateInfo {
                view_type: ImageViewType::Dim2dArray,
                ..ImageViewCreateInfo::from_image(&color_array)
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
            128
        ).unwrap().into();
        // Garbage init
        let building_vertex_buffer = Buffer::new_slice(
            memory_allocator.clone(),
            BufferCreateInfo { 
                usage: BufferUsage::VERTEX_BUFFER, 
                ..Default::default() 
            },
            AllocationCreateInfo { 
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE, 
                ..Default::default() 
            },
            1
        ).unwrap().into();


        Ok(TerrainGpuResources {
            memory_allocator,
            queue,
            command_buffer_allocator,

            instance_buffer,

            heightmap_array,
            heightmap_array_view,
            color_array,
            color_array_view,
            building_vertices: building_vertex_buffer,
        })
    }

    pub fn upload_image_array<T: BufferContents>( &mut self, data: Vec<T>, layers: u32, target: Arc<Image>) -> Result<()> {
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
                            array_layers: 0..layers, // IMPORTANT: only copy layers we can actuall fill
                            ..target.subresource_layers()
                        },
                        image_extent: target.extent(),
                        ..Default::default()
                    }]
                    .into(),
            ..CopyBufferToImageInfo::buffer_image(
                heightmap_upload_buffer,
                target.clone(),
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

// Clipmap structure
// Maps terrain patches to mapbox tile coordinates in clockwise order
#[derive(Debug)]
struct Clipmap {
    center: [[UVec3; 4]; 4],
    rings: Vec<[UVec3; 12]>,
}

const MAX_ZOOM_TERRAIN: u32 = 12;
impl Clipmap {
    pub fn new(origin: Vec2, zoom: u32, rings: u8) -> Self {
        let mut top_left_tile = Self::find_starting_tile(origin, zoom, rings);

        let center = Self::make_center(top_left_tile);

        let mut rings_vec = Vec::with_capacity(rings.into());
        for _r in 0..rings {
            rings_vec.push(Self::make_ring(top_left_tile));
            top_left_tile = (top_left_tile / 2).with_z(top_left_tile.z) - 1;
        }

        Clipmap {
            center,
            rings: rings_vec,
        }
    }

    pub fn rebuild(&mut self, origin: Vec2, zoom: u32) {
        let rings = self.rings.len();
        let new = Clipmap::new(origin, zoom, rings as u8);

        *self = new;
    }

    pub fn map_ring_offset(idx: usize, top_left_inner: Vec2, stride: f32) -> Vec2 {
        let top_left_ring = top_left_inner - Vec2::X*stride - Vec2::Y*stride;
        match idx {
            0  => top_left_ring + Vec2::X*stride*0. + Vec2::Y*stride*0.,
            1  => top_left_ring + Vec2::X*stride*1. + Vec2::Y*stride*0.,
            2  => top_left_ring + Vec2::X*stride*2. + Vec2::Y*stride*0.,
            3  => top_left_ring + Vec2::X*stride*3. + Vec2::Y*stride*0.,

            4  => top_left_ring + Vec2::X*stride*3. + Vec2::Y*stride*1.,
            5  => top_left_ring + Vec2::X*stride*3. + Vec2::Y*stride*2.,

            6  => top_left_ring + Vec2::X*stride*3. + Vec2::Y*stride*3.,
            7  => top_left_ring + Vec2::X*stride*2. + Vec2::Y*stride*3.,
            8  => top_left_ring + Vec2::X*stride*1. + Vec2::Y*stride*3.,
            9  => top_left_ring + Vec2::X*stride*0. + Vec2::Y*stride*3.,

            10 => top_left_ring + Vec2::X*stride*0. + Vec2::Y*stride*2.,
            11 => top_left_ring + Vec2::X*stride*0. + Vec2::Y*stride*1.,
            _  => panic!()
        }
    }

    // The top left tile of the center (and each ring) must be bottom right of a tile with zoom level - 1
    // truncating division ensures the tile is a top-left subtile
    fn find_starting_tile(pos: Vec2, zoom: u32, rings: u8) -> UVec3 {
        let mut start = util::world_to_tile_idx(pos, zoom);

        let zoom = start.z;
        for _ in 0..rings {
            start /= 2;
        }
        for _ in 0..rings-1 {
            start *= 2;
            start += UVec3::ONE;
        }
        start *= 2;
        start.with_z(zoom)
    }

    /// Creates a counter-clockwise ring surrounding the inner patch
    /// `top_left_inner`: the top-left tile of the inner patch we want to encircle
    fn make_ring(top_left_inner: UVec3) -> [UVec3; 12] {
        debug_assert!(1 < top_left_inner.min_element(), "top left inner is {}", top_left_inner);
        let top_left = (top_left_inner / 2)
            .with_z(top_left_inner.z) - 1;
        [
            top_left + UVec3::X*0,
            top_left + UVec3::X*1,
            top_left + UVec3::X*2,
            top_left + UVec3::X*3,

            top_left + UVec3::X*3 + UVec3::Y*1,
            top_left + UVec3::X*3 + UVec3::Y*2,

            top_left + UVec3::X*3 + UVec3::Y*3,
            top_left + UVec3::X*2 + UVec3::Y*3,
            top_left + UVec3::X*1 + UVec3::Y*3,
            top_left + UVec3::X*0 + UVec3::Y*3,

            top_left + UVec3::Y*2,
            top_left + UVec3::Y*1,
        ]
    }
    /// Creates the center of our clipmap
    fn make_center(top_left: UVec3) -> [[UVec3; 4]; 4] {
        [
            [top_left, top_left + UVec3::X, top_left + UVec3::X*2, top_left + UVec3::X*3],
            [top_left + UVec3::Y, top_left + UVec3::Y + UVec3::X, top_left + UVec3::Y + UVec3::X*2, top_left + UVec3::Y + UVec3::X*3],
            [top_left + UVec3::Y*2, top_left + UVec3::Y*2 + UVec3::X, top_left + UVec3::Y*2 + UVec3::X*2, top_left + UVec3::Y*2 + UVec3::X*3],
            [top_left + UVec3::Y*3, top_left + UVec3::Y*3 + UVec3::X, top_left + UVec3::Y*3 + UVec3::X*2, top_left + UVec3::Y*3 + UVec3::X*3],
        ]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clipmap_center_logic() {
        let top_left = UVec3::new(2174, 1430, 12);
        let center = Clipmap::make_center(top_left);

        let expected = [
            [UVec3::new(2174, 1430, 12), UVec3::new(2175, 1430, 12), UVec3::new(2176, 1430, 12), UVec3::new(2177, 1430, 12)],
            [UVec3::new(2174, 1431, 12), UVec3::new(2175, 1431, 12), UVec3::new(2176, 1431, 12), UVec3::new(2177, 1431, 12)],
            [UVec3::new(2174, 1432, 12), UVec3::new(2175, 1432, 12), UVec3::new(2176, 1432, 12), UVec3::new(2177, 1432, 12)],
            [UVec3::new(2174, 1433, 12), UVec3::new(2175, 1433, 12), UVec3::new(2176, 1433, 12), UVec3::new(2177, 1433, 12)],
        ];

        assert_eq!(center, expected)
    }

    #[test]
    fn clipmap_ring_logic() {
        let top_left_inner = UVec3::new(2174, 1430, 12);
        let ring = Clipmap::make_ring(top_left_inner);

        let expected = [
            UVec3::new(1086, 714, 11),
            UVec3::new(1087, 714, 11),
            UVec3::new(1088, 714, 11),
            UVec3::new(1089, 714, 11),

            UVec3::new(1089, 715, 11),
            UVec3::new(1089, 716, 11),

            UVec3::new(1089, 717, 11),
            UVec3::new(1088, 717, 11),
            UVec3::new(1087, 717, 11),
            UVec3::new(1086, 717, 11),

            UVec3::new(1086, 716, 11),
            UVec3::new(1086, 715, 11),
        ];

        assert_eq!(ring, expected)
    }

    #[test]
    fn find_starting_tile_test() {
        let actual = Clipmap::find_starting_tile(Vec2::new(21715., 14314.), 12, 2);
        let expected = UVec3::new(2170, 1430, 12);

        assert_eq!(actual, expected);
    }

    #[test]
    fn clipmap_test() {
        let cm = Clipmap::new(Vec2::new(21715., 14314.), 12, 2);

        let center = [
            [UVec3::new(2170, 1430, 12), UVec3::new(2171, 1430, 12), UVec3::new(2172, 1430, 12), UVec3::new(2173, 1430, 12)],
            [UVec3::new(2170, 1431, 12), UVec3::new(2171, 1431, 12), UVec3::new(2172, 1431, 12), UVec3::new(2173, 1431, 12)],
            [UVec3::new(2170, 1432, 12), UVec3::new(2171, 1432, 12), UVec3::new(2172, 1432, 12), UVec3::new(2173, 1432, 12)],
            [UVec3::new(2170, 1433, 12), UVec3::new(2171, 1433, 12), UVec3::new(2172, 1433, 12), UVec3::new(2173, 1433, 12)],
        ];

        let ring1 = [
            UVec3::new(1084, 714, 11),
            UVec3::new(1085, 714, 11),
            UVec3::new(1086, 714, 11),
            UVec3::new(1087, 714, 11),

            UVec3::new(1087, 715, 11),
            UVec3::new(1087, 716, 11),

            UVec3::new(1087, 717, 11),
            UVec3::new(1086, 717, 11),
            UVec3::new(1085, 717, 11),
            UVec3::new(1084, 717, 11),

            UVec3::new(1084, 716, 11),
            UVec3::new(1084, 715, 11),
        ];
        let ring2 = [
            UVec3::new(541, 356, 10),
            UVec3::new(542, 356, 10),
            UVec3::new(543, 356, 10),
            UVec3::new(544, 356, 10),

            UVec3::new(544, 357, 10),
            UVec3::new(544, 358, 10),

            UVec3::new(544, 359, 10),
            UVec3::new(543, 359, 10),
            UVec3::new(542, 359, 10),
            UVec3::new(541, 359, 10),

            UVec3::new(541, 358, 10),
            UVec3::new(541, 357, 10),
        ];

        assert_eq!(cm.center, center, "Clipmap center is wrong");
        assert_eq!(cm.rings[0], ring1, "Clipmap ring 1 is wrong");
        assert_eq!(cm.rings[1], ring2, "Clipmap ring 2 is wrong");
    }
}

