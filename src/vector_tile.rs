use glam::{Vec2};
use crate::{util, terrain::Gray16Image};
use vulkano::{
    buffer::BufferContents,
    pipeline::graphics::vertex_input::Vertex,
};
use anyhow::{Result, Context};
use image::{self, DynamicImage};
use reqwest::blocking::Client;
use glam::UVec3;
use geozero::{ToGeo, mvt::{self, Message}};
use geo::{
    CoordsIter, Geometry, MultiPolygon, Polygon, algorithm::TriangulateEarcut, coord, map_coords::MapCoords
};

trait AsGeoCoord {
    fn as_coord(&self) -> geo::Coord;
}
impl AsGeoCoord for Vec2 {
    fn as_coord(&self) -> geo::Coord {
        coord! {
            x: self.x as f64,
            y: self.y as f64,
        }
    }
}

#[derive(BufferContents, Vertex, Debug, Clone)]
#[repr(C)]
pub struct BuildingVertex {
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],
}

#[derive(Clone)]
pub struct VectorTile {
    pub mesh: Vec<BuildingVertex>,
    tile: mvt::Tile,
}

impl VectorTile {
    pub fn new(client: &Client, coords: UVec3, heightmap: &Gray16Image) -> Result<Self> {
        let UVec3 { x, y, z} = coords;
        let url = format!("http://localhost:3000/germany/{}/{}/{}", z, x, y);

        eprintln!("Fetching tile @ {} {} {}", x, y, z);

        let resp = client.get(url).send()
            .context("Failed to connect to Tileserver")?;
        let bytes = resp.bytes().context("Failed to read bytes")?;

        let tile = mvt::Tile::decode(bytes)?;
        let buildings_layer = tile.layers.iter()
            .find(|x| x.name == "buildings")
            .context("No buildings layer present")?;

        // generate mesh
        let mut mesh = Vec::new();
        // 1. Prepare Transformation Data
        let tile_origin = util::tile_to_world_coords(coords);
        let tile_width = util::zoom_to_dist(z) as f64;
        let extent = buildings_layer.extent.unwrap_or(4096) as f64;
        let scale = tile_width / extent;

        // 2. Define the Transformer (MVT Int -> World Float)
        let transform = |coords: geo::Coord| {
            tile_origin.as_coord() + coords*scale - util::WORLD_ORIGIN.as_coord()
        };

        for feature in &buildings_layer.features {
            let Ok(geo) = feature.to_geo() else { continue };

            let height = get_prop(feature, buildings_layer, "height")
                .or_else(|| get_prop(feature, buildings_layer, "levels").map(|l| l * 3.5))
                .unwrap_or(6.0)
                * 0.001; // convert from m to km
            
            let min_height = get_prop(feature, buildings_layer, "min_height").unwrap_or(0.0) * 0.001;

            let ctx = BuildingContext {
                roof_y: height,
                base_y: min_height,
                tile_origin,
                tile_width: tile_width as f32,
                heightmap,
            };

            // 3. Process Geometry
            match geo {
                Geometry::Polygon(poly) => {
                    // Transform to World Space FIRST, then process
                    let world_poly = poly.map_coords(transform);
                    process_polygon(&world_poly, &ctx, &mut mesh);
                }
                Geometry::MultiPolygon(mpoly) => {
                    for poly in mpoly {
                        let world_poly = poly.map_coords(transform);
                        process_polygon(&world_poly, &ctx, &mut mesh);
                    }
                }
                _ => {}
            }
        }

        Ok(Self {
            tile,
            mesh,
        })
    }
}
// Helper struct to bundle data for the polygon processor
struct BuildingContext<'a> {
    roof_y: f32,
    base_y: f32,
    tile_origin: Vec2,
    tile_width: f32,
    heightmap: &'a Gray16Image,
}

impl<'a> BuildingContext<'a> {
    // Helper to sample height at a specific WORLD X/Z coordinate
    fn sample_ground(&self, world_x: f32, world_z: f32) -> f32 {
        // 1. Calculate UV relative to the tile
        // Note: world_z corresponds to Y in the tile texture space
        let u = (world_x - (self.tile_origin.x - util::WORLD_ORIGIN.x)) / self.tile_width;
        let v = (world_z - (self.tile_origin.y - util::WORLD_ORIGIN.y)) / self.tile_width;

        // 2. Clamp to safe image bounds [0.0, 1.0]
        let u = u.clamp(0.0, 1.0);
        let v = v.clamp(0.0, 1.0);

        // 3. Map to pixel coordinates
        // Assuming 256x256 image. -1 to ensure we don't go out of bounds at 1.0
        let (w, h) = self.heightmap.dimensions();
        let px = (u * (w - 1) as f32).round() as u32;
        let py = (v * (h - 1) as f32).round() as u32;

        // 4. Sample and Scale
        let raw_val = self.heightmap.get_pixel(px, py)[0];
        raw_val as f32 / 1000.0
    }
}

fn process_polygon(
    poly: &geo::Polygon<f64>, 
    ctx: &BuildingContext,
    out: &mut Vec<BuildingVertex>
) {
    // --- 1. Roof Triangulation ---
    let triangulation = poly.earcut_triangles_raw();

    for idx in triangulation.triangle_indices {
        let vx = triangulation.vertices[idx * 2] as f32;
        let vz = triangulation.vertices[idx * 2 + 1] as f32;

        // Sample ground height specifically for this vertex
        let ground = ctx.sample_ground(vx, vz);

        out.push(BuildingVertex {
            // Apply ground height to the roof
            position: [vx, ctx.roof_y + ground, vz],
        });
    }

    // --- 2. Wall Extrusion ---
    let rings = std::iter::once(poly.exterior()).chain(poly.interiors());

    for ring in rings {
        let points: Vec<_> = ring.coords_iter().collect();
        if points.len() < 2 { continue; }

        for i in 0..points.len() - 1 {
            let c1 = points[i];
            let c2 = points[i+1];

            let p1 = [c1.x as f32, c1.y as f32];
            let p2 = [c2.x as f32, c2.y as f32];

            // Sample height for START and END of the wall segment separately
            let g1 = ctx.sample_ground(p1[0], p1[1]);
            let g2 = ctx.sample_ground(p2[0], p2[1]);

            // Define corners with their respective ground displacements
            let b1 = ctx.base_y + g1; // Bottom 1
            let b2 = ctx.base_y + g2; // Bottom 2
            let t1 = ctx.roof_y + g1; // Top 1
            let t2 = ctx.roof_y + g2; // Top 2

            // Wall Quad (2 Triangles)
            // Triangle 1: BL -> BR -> TR
            out.push(BuildingVertex { position: [p1[0], b1, p1[1]] });
            out.push(BuildingVertex { position: [p2[0], b2, p2[1]] });
            out.push(BuildingVertex { position: [p2[0], t2, p2[1]] });

            // Triangle 2: BL -> TR -> TL
            out.push(BuildingVertex { position: [p1[0], b1, p1[1]] });
            out.push(BuildingVertex { position: [p2[0], t2, p2[1]] });
            out.push(BuildingVertex { position: [p1[0], t1, p1[1]] });
        }
    }
}

fn get_prop(feature: &mvt::tile::Feature, layer: &mvt::tile::Layer, key: &str) -> Option<f32> {
    let key_idx = layer.keys.iter().position(|k| k == key)?;
    
    for chunk in feature.tags.chunks(2) {
        if chunk[0] as usize == key_idx {
            let val_idx = chunk[1] as usize;
            return match &layer.values[val_idx] {
                mvt::tile::Value { float_value: Some(v), .. } => Some(*v),
                mvt::tile::Value { double_value: Some(v), .. } => Some(*v as f32),
                mvt::tile::Value { int_value: Some(v), .. } => Some(*v as f32),
                mvt::tile::Value { uint_value: Some(v), .. } => Some(*v as f32),
                mvt::tile::Value { sint_value: Some(v), .. } => Some(*v as f32),
                mvt::tile::Value { string_value: Some(s), .. } => s.parse().ok(),
                _ => None,
            };
        }
    }
    None
}

