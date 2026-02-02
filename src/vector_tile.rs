use rand::Rng;
use glam::{DVec2, UVec2, Vec2, Vec3};
use rayon::iter::IntoParallelRefIterator;
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
    Area, Centroid, CoordsIter, Geometry, algorithm::TriangulateEarcut, coord, map_coords::MapCoords, point
};
use geo_buffer::buffer_polygon;

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
trait AsDVec2 {
    fn as_dvec2(&self) -> DVec2;
}
impl AsDVec2 for geo::Coord {
    fn as_dvec2(&self) -> DVec2 {
        DVec2 { x: self.x, y: self.y }
    }
}

#[derive(BufferContents, Vertex, Debug, Clone)]
#[repr(C)]
pub struct BuildingVertex {
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    pub color: [f32; 3],
}

#[derive(BufferContents, Vertex, Debug, Clone, Copy)]
#[repr(C)]
pub struct TreeInstance {
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    pub color: [f32; 3],
    #[format(R32_SFLOAT)]
    pub scale: f32,
}

#[derive(Clone)]
pub struct VectorTile {
    pub mesh: Vec<BuildingVertex>,
    pub trees: Vec<TreeInstance>,
}

// Helper struct to bundle data for the polygon processor
struct BuildingContext {
    top_y: f64,      // Peak height
    eaves_y: f64,    // Gutter height
    base_y: f64,     // Ground floor height
    roof_height: f64,
    roof_direction: f32,       // Angle in degrees
    roof_orientation: Option<String>,
    wall_color: [f32; 3],
    roof_color: [f32; 3],
    roof_shape: String,
    ground_elevation: f64,
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

            let centroid_geo = geo.centroid().unwrap_or(point!{x: 0.0, y: 0.0});
            let centroid_world = transform(centroid_geo.0); // Convert to world space
            
            let ground_elevation = sample_terrain_height(
                heightmap, 
                tile_origin, 
                tile_width, 
                centroid_world.x, 
                centroid_world.y
            );

            // --- Parsing OSM Tags ---
            let total_height: f64 = get_prop(feature, buildings_layer, "height").map(|x| x.float_value())
                .or_else(|| get_prop(feature, buildings_layer, "levels").map(|l| l.float_value() * 3.5))
                .unwrap_or(6.0) as f64
                * 0.001; // meters to km
            
            let min_height = get_prop(feature, buildings_layer, "min_height")
                .map_or(0.0, |x| x.float_value()) as f64 * 0.001;

            // colors
            let wall_hex = get_prop(feature, buildings_layer, "building_color")
                .map(|x| x.string_value().to_owned());
            let roof_hex = get_prop(feature, buildings_layer, "roof_color")
                .map(|x| x.string_value().to_owned());

            let roof_shape_tag = get_prop(feature, buildings_layer, "roof_shape")
                .map(|x| x.string_value().to_owned())
                .unwrap_or_else(|| "flat".to_string());

            // 1. Explicit Roof Height (Priority: Tag > Levels > Heuristic)
            let explicit_roof_h = get_prop(feature, buildings_layer, "roof_height")
                .map(|x| x.float_value() as f64 * 0.001); // meters to km

            let levels_roof_h = get_prop(feature, buildings_layer, "roof_levels")
                .map(|x| x.float_value() as f64 * 0.0035); // ~3.5m per level

            let is_flat = roof_shape_tag == "flat";
            let roof_h = explicit_roof_h
                .or(levels_roof_h)
                .unwrap_or(if is_flat { 0.0 } else { 0.003 }); // Default 3m if unknown

            // 2. Orientation & Direction
            let roof_orientation = get_prop(feature, buildings_layer, "roof_orientation")
                .map(|x| x.string_value().to_owned());

            let roof_direction = get_prop(feature, buildings_layer, "roof_direction")
                .map(|x| parse_direction_tag(&x))
                .unwrap_or(0.0);

            let eaves_y = (total_height - roof_h).max(min_height);

            let ctx = BuildingContext {
                top_y: total_height,
                eaves_y,
                base_y: min_height,
                roof_height: roof_h,
                roof_direction,
                roof_orientation,
                wall_color: parse_osm_color(wall_hex.as_deref().unwrap_or("default")),
                roof_color: parse_osm_color(roof_hex.as_deref().unwrap_or("roof_default")),
                roof_shape: roof_shape_tag, 
                ground_elevation,
            };

            // 3. Process Geometry
            match geo {
                Geometry::Polygon(poly) => {
                    // Transform to World Space FIRST, then process
                    let world_poly = poly.map_coords(transform);
                    mesh.extend(process_polygon(&world_poly, &ctx));
                }
                Geometry::MultiPolygon(mpoly) => {
                    for poly in mpoly {
                        let world_poly = poly.map_coords(transform);
                        mesh.extend(process_polygon(&world_poly, &ctx));
                    }
                }
                _ => {}
            }
        }

        // Parse trees
        let mut trees = Vec::new();
        let mut rng = rand::rng();

        if let Some(layer) = tile.layers.iter().find(|x| x.name == "natural") {
            // println!("{:?}", &layer);
            let extent = layer.extent.unwrap_or(4096) as f64;
            let layer_scale = tile_width / extent;
            for feature in &layer.features {
                // Check for tree tag
                let is_tree = feature.tags.chunks(2).any(|chunk| {
                    let k = &layer.keys[chunk[0] as usize];
                    let v = &layer.values[chunk[1] as usize];
                    k == "natural" && v.string_value() == "tree" 
                });

                if is_tree {
                    let geometry = feature.to_geo();

                    // We define a helper closure to process a single point coordinate
                    // This avoids duplicating code for Point vs MultiPoint
                    let mut process_point = |coord: geo::Coord| {
                        let world_pos_2d = tile_origin.as_coord() + coord * layer_scale - util::WORLD_ORIGIN.as_coord();

                        // Sample Height (Reuse your existing logic)
                        let uv = (DVec2::new(world_pos_2d.x, world_pos_2d.y) - (tile_origin - util::WORLD_ORIGIN)) / tile_width;
                        let uv = uv.clamp(DVec2::ZERO, DVec2::ONE);
                        let wh: UVec2 = heightmap.dimensions().into();
                        let p = (uv * (wh-UVec2::ONE).as_dvec2()).round().as_uvec2();
                        let pixel = heightmap.get_pixel(p.x, p.y);
                        let h_km = (-10000.0 + ((pixel[0] as f64 * 65536.0 + pixel[1] as f64 * 256.0 + pixel[2] as f64) * 0.1)) / 1000.0;

                        let scale_var: f32 = rng.random_range(0.8..1.5);

                        trees.push(TreeInstance {
                            // Add +0.001 to height to ensure it sits slightly above ground to prevent z-fighting
                            position: [world_pos_2d.x as f32, (h_km + 0.001) as f32, world_pos_2d.y as f32],
                            color: [0.1, rng.random_range(0.3..0.6), 0.1],
                            scale: scale_var,
                        });
                    };

                    // Handle both Point and MultiPoint
                    match geometry {
                        Ok(Geometry::Point(pt)) => {
                            process_point(pt.0);
                        },
                        Ok(Geometry::MultiPoint(mp)) => {
                            // Iterate over all points in the MultiPoint feature
                            for pt in mp.0 {
                                process_point(pt.0);
                            }
                        },
                        _ => {} // Ignore lines/polygons
                    }
                }
            }
        }

        Ok(Self {
            mesh,
            trees,
        })
    }
}

impl BuildingContext {
    // Helper to sample height at a specific WORLD X/Z coordinate
    fn sample_ground(&self, world_x: f64, world_z: f64) -> f64 {
        self.ground_elevation
    }
}

fn process_polygon(poly: &geo::Polygon<f64>, ctx: &BuildingContext) -> Vec<BuildingVertex> {
    let mut vertices = process_roof(poly, ctx);
    vertices.extend(process_walls(poly, ctx));
    vertices
}

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////// START ROOF HELPERS //////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

fn process_roof(poly: &geo::Polygon<f64>, ctx: &BuildingContext) -> Vec<BuildingVertex> {
    match ctx.roof_shape.as_str() {
        "hipped" | "gabled" => {
            // For hipped, we use the roof height as the inset distance (assuming ~45 deg slope).
            // If the polygon is narrow, this naturally produces a ridge or peak.
            // We clamp min inset to avoid degenerate offsets on tiny roofs.
            let inset = ctx.roof_height.max(0.001); 
            create_hipped_roof(poly, ctx, inset, ctx.top_y)
        },
        "mansard" | "gambrel" => {
            // Mansard has a steep lower slope. We use a small fixed inset (e.g., 1.5m).
            // This creates the characteristic steep "wall" of the roof.
            create_hipped_roof(poly, ctx, 0.0015, ctx.top_y)
        },
        "pyramidal" | "tented" => create_pyramidal_roof(poly, ctx),
        "dome" | "onion" => create_dome_roof(poly, ctx),
        "skillion" => create_skillion_roof(poly, ctx),
        _ => create_flat_roof(poly, ctx),
    }
}

fn create_flat_roof(poly: &geo::Polygon<f64>, ctx: &BuildingContext) -> Vec<BuildingVertex> {
    let mut vertices = Vec::new();
    let triangulation = poly.earcut_triangles_raw();
    
    for idx in triangulation.triangle_indices {
        let p = DVec2::new(
            triangulation.vertices[idx * 2], 
            triangulation.vertices[idx * 2 + 1]
        );
        let ground = ctx.sample_ground(p.x, p.y);
        let pos = Vec3::new(p.x as f32, (ctx.top_y + ground) as f32, p.y as f32);

        vertices.push(BuildingVertex {
            position: pos.to_array(),
            color: ctx.roof_color,
        });
    }
    vertices
}

fn create_hipped_roof(
    poly: &geo::Polygon<f64>, 
    ctx: &BuildingContext, 
    inset: f64,
    ridge_y: f64
) -> Vec<BuildingVertex> {
    let mut vertices = Vec::new();
    
    // 1. Generate the inner "ridge" polygons
    let buffered_multi = buffer_polygon(poly, -inset);
    
    // 2. Collect ALL inner rings to act as holes for the slope generation
    //    If the building is L-shaped, buffer_polygon returns 2+ polygons.
    //    We must treat them all as holes in ONE loft operation to avoid overlapping meshes.
    let mut inner_rings = Vec::new();
    for inner_poly in &buffered_multi {
        inner_rings.push(inner_poly.clone());
    }

    if inner_rings.is_empty() {
         return create_pyramidal_roof(poly, ctx);
    }

    // 3. Loft the Slope (One pass for the whole building)
    //    This creates the angled roof surface between the Eaves (outer) and ALL Ridges (inners)
    vertices.extend(loft_polygons(poly, &inner_rings, ctx.eaves_y, ridge_y, ctx));

    // 4. Cap the Ridges (Flat top parts)
    for inner_poly in buffered_multi {
        let triangulation = inner_poly.earcut_triangles_raw();
        for idx in triangulation.triangle_indices {
            let p = DVec2::new(triangulation.vertices[idx * 2], triangulation.vertices[idx * 2 + 1]);
            let ground = ctx.sample_ground(p.x, p.y);
            let pos = Vec3::new(p.x as f32, (ridge_y + ground) as f32, p.y as f32);
            vertices.push(BuildingVertex { position: pos.to_array(), color: ctx.roof_color });
        }
    }

    vertices
}

fn create_pyramidal_roof(poly: &geo::Polygon<f64>, ctx: &BuildingContext) -> Vec<BuildingVertex> {
    let mut vertices = Vec::new();
    let centroid = poly.centroid().unwrap_or(point!{x: 0.0, y:0.0});
    
    // Calculate Peak Position
    let mut peak_x = centroid.x();
    let mut peak_z = centroid.y();

    // If a direction is provided (and it's not 0 which is default N), shift the peak
    // This handles "tented" or offset pyramidal roofs.
    if ctx.roof_direction != 0.0 {
        // Shift by ~1/4 of the bounding box size or a fixed amount? 
        // A robust way is hard without OBB, but a small offset works for visual flair.
        let offset_dist = 0.002; // 2 meters
        let angle_rad = ctx.roof_direction.to_radians();
        // OSM Direction is usually "down slope", so peak is opposite? 
        // Or "direction of the peak"? Standard is "direction of slope". 
        // Let's assume direction points *to* the peak offset for visual distinctness.
        peak_x += angle_rad.sin() as f64 * offset_dist;
        peak_z -= angle_rad.cos() as f64 * offset_dist;
    }

    let g_center = ctx.sample_ground(peak_x, peak_z);
    let peak = Vec3::new(peak_x as f32, (ctx.top_y + g_center) as f32, peak_z as f32);

    let ring: Vec<_> = poly.exterior().coords_iter().collect();

    for i in 0..ring.len() - 1 {
        let p1 = DVec2::new(ring[i].x, ring[i].y);
        let p2 = DVec2::new(ring[i+1].x, ring[i+1].y);

        let g1 = ctx.sample_ground(p1.x, p1.y);
        let g2 = ctx.sample_ground(p2.x, p2.y);

        let v1 = Vec3::new(p1.x as f32, (ctx.eaves_y + g1) as f32, p1.y as f32);
        let v2 = Vec3::new(p2.x as f32, (ctx.eaves_y + g2) as f32, p2.y as f32);
        
        vertices.push(BuildingVertex { position: v1.to_array(), color: ctx.roof_color });
        vertices.push(BuildingVertex { position: v2.to_array(), color: ctx.roof_color });
        vertices.push(BuildingVertex { position: peak.to_array(), color: ctx.roof_color });
    }
    vertices
}

fn create_dome_roof(poly: &geo::Polygon<f64>, ctx: &BuildingContext) -> Vec<BuildingVertex> {
    let mut vertices = Vec::new();
    let steps = 6; // More steps for smoother dome
    let centroid = poly.centroid().unwrap_or(point!{x:0.0, y:0.0});
    let center = DVec2::new(centroid.x(), centroid.y());
    
    let mut prev_ring: Vec<DVec2> = poly.exterior().coords_iter().map(|c| DVec2::new(c.x, c.y)).collect();
    let mut prev_y = ctx.eaves_y;

    let is_onion = ctx.roof_shape == "onion";

    for s in 1..=steps {
        let t = s as f64 / steps as f64; 
        
        // Profile Calculation
        let (scale, h_factor) = if is_onion {
            // Onion profile: bulges out (scale > 1.0) then comes back in
            // Simple sigmoid-like curve approximation
            let angle = t * std::f64::consts::PI; 
            let scale_val = if t < 0.5 { 
                1.0 + (angle.sin() * 0.3) // Bulge out by 30%
            } else {
                1.0 - ((t - 0.5) * 2.0).powf(2.0) // Collapse to 0
            };
            // Height accumulates
            let h = t; 
            (scale_val.max(0.0), h)
        } else {
            // Standard Dome (Hemisphere)
            let angle = t * std::f64::consts::PI / 2.0;
            (angle.cos(), angle.sin())
        };

        let h_offset = h_factor * ctx.roof_height;
        let current_y = ctx.eaves_y + h_offset;
        
        // Scale ring towards centroid
        let current_ring: Vec<DVec2> = poly.exterior().coords_iter().map(|p| {
             let pt = DVec2::new(p.x, p.y);
             center + (pt - center) * scale
        }).collect();

        // Lofting (Ring i to Ring i+1)
        for i in 0..prev_ring.len() - 1 {
            let p1 = prev_ring[i];
            let p2 = prev_ring[i+1];
            let p3 = current_ring[i+1];
            let p4 = current_ring[i];
            
            let g1 = ctx.sample_ground(p1.x, p1.y);
            let g3 = ctx.sample_ground(p3.x, p3.y);

            let v1 = Vec3::new(p1.x as f32, (prev_y+g1) as f32, p1.y as f32);
            let v2 = Vec3::new(p2.x as f32, (prev_y+ctx.sample_ground(p2.x, p2.y)) as f32, p2.y as f32);
            let v3 = Vec3::new(p3.x as f32, (current_y+g3) as f32, p3.y as f32);
            let v4 = Vec3::new(p4.x as f32, (current_y+ctx.sample_ground(p4.x, p4.y)) as f32, p4.y as f32);

            vertices.push(BuildingVertex { position: v1.to_array(), color: ctx.roof_color });
            vertices.push(BuildingVertex { position: v2.to_array(), color: ctx.roof_color });
            vertices.push(BuildingVertex { position: v3.to_array(), color: ctx.roof_color });

            vertices.push(BuildingVertex { position: v1.to_array(), color: ctx.roof_color });
            vertices.push(BuildingVertex { position: v3.to_array(), color: ctx.roof_color });
            vertices.push(BuildingVertex { position: v4.to_array(), color: ctx.roof_color });
        }
        prev_ring = current_ring;
        prev_y = current_y;
    }
    vertices
}

fn create_skillion_roof(poly: &geo::Polygon<f64>, ctx: &BuildingContext) -> Vec<BuildingVertex> {
    let mut vertices = Vec::new();
    let triangulation = poly.earcut_triangles_raw();
    
    // OSM Direction is usually the "down slope" direction in degrees (0=North, 90=East)
    // We need the UPHILL vector to calculate height. Uphill = Direction + 180.
    let angle_rad = ctx.roof_direction.to_radians();
    let uphill_angle = angle_rad + std::f32::consts::PI;
    
    // Direction vector (x=sin, y=-cos for North=Up in 2D map space)
    let gradient_dir = DVec2::new(uphill_angle.sin() as f64, -(uphill_angle.cos() as f64)); 

    // Project vertices onto gradient vector to find min/max span
    let ring: Vec<DVec2> = poly.exterior().coords_iter().map(|c| DVec2::new(c.x, c.y)).collect();
    let (min_u, max_u) = ring.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |acc, p| {
        let u = p.dot(gradient_dir);
        (acc.0.min(u), acc.1.max(u))
    });
    let span = (max_u - min_u).max(0.00001);

    for idx in triangulation.triangle_indices {
        let p = DVec2::new(triangulation.vertices[idx * 2], triangulation.vertices[idx * 2 + 1]);
        let ground = ctx.sample_ground(p.x, p.y);
        
        // Interpolate height based on position along gradient
        let u = p.dot(gradient_dir);
        let t = (u - min_u) / span; // 0.0 = bottom of roof, 1.0 = top
        let local_h = ctx.eaves_y + (t * ctx.roof_height);

        let pos = Vec3::new(p.x as f32, (local_h + ground) as f32, p.y as f32);

        vertices.push(BuildingVertex { position: pos.to_array(), color: ctx.roof_color });
    }
    vertices
}

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////// END ROOF HELPERS ////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

fn process_walls(poly: &geo::Polygon<f64>, ctx: &BuildingContext) -> Vec<BuildingVertex> {
    let mut vertices = Vec::new();
    let rings = std::iter::once(poly.exterior()).chain(poly.interiors());

    for ring in rings {
        let points: Vec<_> = ring.coords_iter().collect();
        if points.len() < 2 { continue; }

        for i in 0..points.len() - 1 {
            let p1 = DVec2::new(points[i].x, points[i].y);
            let p2 = DVec2::new(points[i+1].x, points[i+1].y);

            let g1 = ctx.sample_ground(p1.x, p1.y);
            let g2 = ctx.sample_ground(p2.x, p2.y);

            let b1 = ctx.base_y + g1;
            let b2 = ctx.base_y + g2;
            let t1 = ctx.eaves_y + g1;
            let t2 = ctx.eaves_y + g2;

            let v_bl = Vec3::new(p1.x as f32, b1 as f32, p1.y as f32); // Bottom Left
            let v_br = Vec3::new(p2.x as f32, b2 as f32, p2.y as f32); // Bottom Right
            let v_tr = Vec3::new(p2.x as f32, t2 as f32, p2.y as f32); // Top Right
            let v_tl = Vec3::new(p1.x as f32, t1 as f32, p1.y as f32); // Top Left

            // Triangle 1
            vertices.push(BuildingVertex { position: v_bl.to_array(), color: ctx.wall_color });
            vertices.push(BuildingVertex { position: v_br.to_array(), color: ctx.wall_color });
            vertices.push(BuildingVertex { position: v_tr.to_array(), color: ctx.wall_color });

            // Triangle 2
            vertices.push(BuildingVertex { position: v_bl.to_array(), color: ctx.wall_color });
            vertices.push(BuildingVertex { position: v_tr.to_array(), color: ctx.wall_color });
            vertices.push(BuildingVertex { position: v_tl.to_array(), color: ctx.wall_color });
        }
    }
    vertices
}

fn loft_polygons(
    outer: &geo::Polygon<f64>, 
    inners: &[geo::Polygon<f64>], // Now accepts multiple inner polygons
    y_outer: f64, 
    y_inner: f64, 
    ctx: &BuildingContext
) -> Vec<BuildingVertex> {
    // 1. Prepare the Holes List
    // The 'holes' for our slope polygon include:
    //  a. The original building's holes (e.g. courtyards) -> These should be at EAVES height
    //  b. The buffered inner polygons (ridges) -> These should be at RIDGE height
    
    let mut holes = Vec::new();
    
    // Add original holes (Courtyards)
    for hole in outer.interiors() {
        holes.push(hole.clone());
    }
    let courtyard_hole_count = holes.len();

    // Add ridge holes
    for inner in inners {
        holes.push(inner.exterior().clone());
    }

    // 2. Construct the "Slope" Polygon
    let poly = geo::Polygon::new(
        outer.exterior().clone(),
        holes,
    );

    // 3. Triangulate
    let triangulation = poly.earcut_triangles_raw();
    
    let mut vertices = Vec::new();

    // 4. Determine Vertex Heights
    // Earcut flattens all rings into a single buffer: [Exterior, Hole1, Hole2, ...]
    // We need to map the index back to which ring it belongs to.
    
    let mut offset = 0;
    
    // Range for Exterior Ring (Eaves)
    let ext_len = outer.exterior().coords_count();
    let range_exterior = offset..(offset + ext_len);
    offset += ext_len;

    // Range for Courtyard Holes (Eaves)
    let mut range_courtyards = Vec::new();
    for hole in outer.interiors() {
        let len = hole.coords_count();
        range_courtyards.push(offset..(offset + len));
        offset += len;
    }

    // Anything after this belongs to the Ridge Holes (Inner Height)
    let split_index = offset; 

    for idx in triangulation.triangle_indices {
        let idx_usize = idx as usize;
        let x = triangulation.vertices[idx_usize * 2];
        let z = triangulation.vertices[idx_usize * 2 + 1];

        // Decide Height
        let y = if idx_usize < split_index {
            y_outer // It's part of the exterior or a courtyard wall
        } else {
            y_inner // It's part of the ridge
        };

        let ground = ctx.sample_ground(x, z);

        vertices.push(BuildingVertex {
            position: [x as f32, (y + ground) as f32, z as f32],
            color: ctx.roof_color,
        });
    }

    vertices
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

fn parse_direction_tag(val: &mvt::tile::Value) -> f32 {
    let s = match val {
        mvt::tile::Value { string_value: Some(s), .. } => s,
        mvt::tile::Value { float_value: Some(f), .. } => return *f as f32,
        mvt::tile::Value { int_value: Some(i), .. } => return *i as f32,
        _ => return 0.0,
    };

    match s.trim().to_uppercase().as_str() {
        "N" => 0.0,
        "NNE" => 22.5,
        "NE" => 45.0,
        "ENE" => 67.5,
        "E" => 90.0,
        "ESE" => 112.5,
        "SE" => 135.0,
        "SSE" => 157.5,
        "S" => 180.0,
        "SSW" => 202.5,
        "SW" => 225.0,
        "WSW" => 247.5,
        "W" => 270.0,
        "WNW" => 292.5,
        "NW" => 315.0,
        "NNW" => 337.5,
        val => val.parse::<f32>().unwrap_or(0.0),
    }
}

fn sample_terrain_height(
    heightmap: &RgbaImage,
    tile_origin: DVec2,
    tile_width: f64,
    world_x: f64,
    world_z: f64
) -> f64 {
    let uv = (DVec2::new(world_x, world_z) - (tile_origin - util::WORLD_ORIGIN)) / tile_width;
    let uv = uv.clamp(DVec2::ZERO, DVec2::ONE);

    let wh: UVec2 = heightmap.dimensions().into();
    let p = (uv * (wh-UVec2::ONE).as_dvec2()).round().as_uvec2();

    let pixel = heightmap.get_pixel(p.x, p.y);
    let height_meters = -10000.0 + ((pixel[0] as f64 * 65536.0 + pixel[1] as f64 * 256.0 + pixel[2] as f64) * 0.1);
    height_meters / 1000.0
}

