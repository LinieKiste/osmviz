use glam::{DVec2, UVec2, Vec2};
use crate::{util};
use vulkano::{
    buffer::BufferContents,
    pipeline::graphics::vertex_input::Vertex,
};
use anyhow::{Result, Context};
use image::{self, RgbaImage};
use reqwest::blocking::Client;
use glam::UVec3;
use geozero::{ToGeo, mvt::{self, Message}};
use geo::{
    CoordsIter, Geometry, algorithm::TriangulateEarcut, coord, map_coords::MapCoords
};

trait AsGeoCoord {
    fn as_coord(&self) -> geo::Coord;
}
impl AsGeoCoord for DVec2 {
    fn as_coord(&self) -> geo::Coord {
        coord! {
            x: self.x,
            y: self.y,
        }
    }
}

#[derive(BufferContents, Vertex, Debug, Clone)]
#[repr(C)]
pub struct BuildingVertex {
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],

    #[format(R32G32B32_SFLOAT)]
    pub normal: [f32; 3], 

    #[format(R32G32B32_SFLOAT)]
    pub color: [f32; 3],
}

#[derive(Clone)]
pub struct VectorTile {
    pub mesh: Vec<BuildingVertex>,
}

impl VectorTile {
    pub fn new(client: &Client, coords: UVec3, heightmap: &RgbaImage) -> Result<Self> {
        let UVec3 { x, y, z} = coords;
        let url = format!("http://localhost:3000/germany/{}/{}/{}", z, x, y);

        // log::info!("Fetching vector tile @ {} {} {}", x, y, z);

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

            let height = get_prop(feature, buildings_layer, "height").map(|x| x.float_value())
                .or_else(|| get_prop(feature, buildings_layer, "levels").map(|l| l.float_value() * 3.5))
                .unwrap_or(6.0)
                * 0.001; // convert from m to km
            
            let min_height = get_prop(feature, buildings_layer, "min_height")
                .map_or(0.0, |x| x.float_value()) * 0.001;

            // colors
            let wall_hex = get_prop(feature, buildings_layer, "building_color")
                .map(|x| x.string_value().to_owned());
            let roof_hex = get_prop(feature, buildings_layer, "roof_color")
                .map(|x| x.string_value().to_owned());

            let ctx = BuildingContext {
                roof_y: height.into(),
                base_y: min_height.into(),
                wall_color: parse_osm_color(wall_hex.as_deref().unwrap_or("default")),
                roof_color: parse_osm_color(roof_hex.as_deref().unwrap_or("roof_default")),
                tile_origin,
                tile_width,
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
            mesh,
        })
    }
}
// Helper struct to bundle data for the polygon processor
struct BuildingContext<'a> {
    roof_y: f64,
    base_y: f64,
    wall_color: [f32; 3],
    roof_color: [f32; 3],
    tile_origin: DVec2,
    tile_width: f64,
    heightmap: &'a RgbaImage,
}

impl<'a> BuildingContext<'a> {
    // Helper to sample height at a specific WORLD X/Z coordinate
    fn sample_ground(&self, world_x: f64, world_z: f64) -> f64 {
        // 1. Calculate UV relative to the tile
        // Note: world_z corresponds to Y in the tile texture space
        let uv = (DVec2::new(world_x, world_z) - (self.tile_origin - util::WORLD_ORIGIN)) / self.tile_width;
        let uv = uv.clamp(DVec2::ZERO, DVec2::ONE);

        // 3. Map to pixel coordinates
        // Assuming 256x256 image. -1 to ensure we don't go out of bounds at 1.0
        let wh: UVec2 = self.heightmap.dimensions().into();
        let p = (uv * (wh-UVec2::ONE).as_dvec2()).round().as_uvec2();

        // 4. Sample and Scale
        let pixel = self.heightmap.get_pixel(p.x, p.y);
        let r = pixel[0] as f64;
        let g = pixel[1] as f64;
        let b = pixel[2] as f64;
        let height_meters = -10000.0 + ((r * 256.0 * 256.0 + g * 256.0 + b) * 0.1);
        height_meters / 1000.0
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
        let ground = ctx.sample_ground(vx.into(), vz.into());

        out.push(BuildingVertex {
            // Apply ground height to the roof
            position: [vx, (ctx.roof_y + ground) as f32, vz],
            normal: [0.0, 1.0, 0.0],       // Roofs point UP
            color:  ctx.roof_color,       // Brick Red
        });
    }

    // --- 2. Wall Extrusion ---
    let rings = std::iter::once(poly.exterior()).chain(poly.interiors());

    for ring in rings {
        let points: Vec<_> = ring.coords_iter().collect();
        if points.len() < 2 { continue; }

        for i in 0..points.len() - 1 {
            let (p1, p2) = ([points[i].x as f32, points[i].y as f32], [points[i+1].x as f32, points[i+1].y as f32]);
            
            // Sample ground for both ends of the wall segment
            let g1 = ctx.sample_ground(p1[0].into(), p1[1].into());
            let g2 = ctx.sample_ground(p2[0].into(), p2[1].into());

            // Use base_y (min_height) instead of assuming 0.0
            let b1 = ctx.base_y + g1; 
            let b2 = ctx.base_y + g2; 
            let t1 = ctx.roof_y + g1;
            let t2 = ctx.roof_y + g2;

            // Calculate Flat Face Normal
            let wall_vec = [p2[0] - p1[0], 0.0, p2[1] - p1[1]];
            let len = (wall_vec[0].powi(2) + wall_vec[2].powi(2)).sqrt();
            let normal = [-wall_vec[2] / len, 0.0, wall_vec[0] / len];

            // Push vertices (Triangle 1 & 2) using ctx.wall_color
            // (Standard quad generation code here, just ensure you use 'b1/b2' for bottom Y)
            out.push(BuildingVertex { position: [p1[0], b1 as f32, p1[1]], normal, color: ctx.wall_color });
            out.push(BuildingVertex { position: [p2[0], b2 as f32, p2[1]], normal, color: ctx.wall_color });
            out.push(BuildingVertex { position: [p2[0], t2 as f32, p2[1]], normal, color: ctx.wall_color });

            out.push(BuildingVertex { position: [p1[0], b1 as f32, p1[1]], normal, color: ctx.wall_color });
            out.push(BuildingVertex { position: [p2[0], t2 as f32, p2[1]], normal, color: ctx.wall_color });
            out.push(BuildingVertex { position: [p1[0], t1 as f32, p1[1]], normal, color: ctx.wall_color });
        }
    }
}

/// Helper to parse standard OSM colors and Hex codes
fn parse_osm_color(color: &str) -> [f32; 3] {
    match color {
        // Common OSM colors
        "white" => [0.9, 0.9, 0.9],
        "black" => [0.1, 0.1, 0.1],
        "gray" | "grey" => [0.5, 0.5, 0.5],
        "light_gray" => [0.7, 0.7, 0.7],
        "red" => [0.8, 0.1, 0.1],
        "green" => [0.1, 0.6, 0.1],
        "blue" => [0.2, 0.2, 0.8],
        "yellow" => [0.9, 0.9, 0.1],
        "brown" => [0.4, 0.2, 0.1],
        "orange" => [0.9, 0.5, 0.1],
        "beige" => [0.96, 0.96, 0.86],
        "brick_red" => [0.8, 0.25, 0.15],
        "roof_default" => [0.55, 0.2, 0.2], // Terracotta/Reddish standard roof
        "default" => [0.85, 0.85, 0.85],    // Standard white/gray walls
        
        // Hex Parsing (e.g., #FFFFFF)
        s if s.starts_with('#') && s.len() == 7 => {
            let r = u8::from_str_radix(&s[1..3], 16).unwrap_or(255) as f32 / 255.0;
            let g = u8::from_str_radix(&s[3..5], 16).unwrap_or(255) as f32 / 255.0;
            let b = u8::from_str_radix(&s[5..7], 16).unwrap_or(255) as f32 / 255.0;
            [r, g, b]
        }
        _ => [0.8, 0.8, 0.8],
    }
}

fn get_prop(feature: &mvt::tile::Feature, layer: &mvt::tile::Layer, key: &str) -> Option<mvt::tile::Value> {
    let key_idx = layer.keys.iter().position(|k| k == key)?;
    
    for chunk in feature.tags.chunks(2) {
        if chunk[0] as usize == key_idx {
            let val_idx = chunk[1] as usize;
            return Some(layer.values[val_idx].clone());
        }
    }
    None
}

