// src/shaders.rs
pub mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
            #version 450
            layout(location = 0) in vec2 position;      // 0..1 Grid Mesh
            layout(location = 1) in vec2 world_offset;  // Meter offset (Instance)
            layout(location = 2) in uint texture_layer; // Texture ID (Instance)
            layout(location = 3) in float scale;

            layout(location = 0) out vec2 out_uv;       // Pass 0..1 to FS for texture sampling
            layout(location = 1) out flat uint out_layer; 

            void main() {
                out_uv = position; 
                out_layer = texture_layer;

                // TRANSFORM TO WORLD SPACE HERE
                vec2 world_pos = (position * scale) + world_offset;

                // gl_Position is now in pure World Meters (X, 0, Z)
                gl_Position = vec4(world_pos.x, 0.0, world_pos.y, 1.0);
            }
        ",
    }
}

pub mod tcs {
    vulkano_shaders::shader! {
        ty: "tess_ctrl",
        src: r"
            #version 450
            layout(vertices = 4) out;

            layout(location = 0) in vec2 in_uv[];
            layout(location = 1) in flat uint in_layer[];

            layout(location = 0) out vec2 out_uv[];
            layout(location = 1) out flat uint out_layer[];

            layout(set = 0, binding = 0) uniform Camera {
                mat4 view;
                mat4 proj;
                vec3 pos;
            } ubo;

            void main() {
                gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
                out_uv[gl_InvocationID] = in_uv[gl_InvocationID];
                out_layer[gl_InvocationID] = in_layer[gl_InvocationID];

                if (gl_InvocationID == 0) {
                    // SIMPLIFIED: gl_in is already in World Space!
                    // Just average the 4 corners to find the patch center
                    vec4 center = (gl_in[0].gl_Position + gl_in[1].gl_Position + 
                                   gl_in[2].gl_Position + gl_in[3].gl_Position) / 4.0;
                    
                    float dist = distance(ubo.pos, center.xyz);

                    // LOD Calculation (Same logic)
                    float level = mix(64.0, 2.0, clamp((dist - 200.0) / 1000.0, 0.0, 1.0));

                    gl_TessLevelOuter[0] = level;
                    gl_TessLevelOuter[1] = level;
                    gl_TessLevelOuter[2] = level;
                    gl_TessLevelOuter[3] = level;

                    gl_TessLevelInner[0] = level;
                    gl_TessLevelInner[1] = level;
                }
            }
        ",
    }
}

pub mod tes {
    vulkano_shaders::shader! {
        ty: "tess_eval",
        src: r"
            #version 450
            layout(quads, equal_spacing, ccw) in;

            layout(location = 0) in vec2 in_uv[];
            layout(location = 1) in flat uint in_layer[];

            layout(location = 0) out float height_factor;
            layout(location = 1) out vec3 v_world_pos;
            layout(location = 2) out vec3 v_normal;
            layout(location = 3) out flat uint out_layer;
            layout(location = 4) out vec2 out_uv;

            layout(set = 0, binding = 0) uniform Camera {
                mat4 view;
                mat4 proj;
                vec3 pos;
            } ubo;

            layout(set = 0, binding = 1) uniform sampler2DArray heightMap;

            const float MAX_HEIGHT = 20.0; 

            void main() {
                // 1. Interpolate UVs
                vec2 u1 = mix(in_uv[0], in_uv[1], gl_TessCoord.x);
                vec2 u2 = mix(in_uv[3], in_uv[2], gl_TessCoord.x);
                vec2 tex_coord = mix(u1, u2, gl_TessCoord.y);

                uint layer = in_layer[0];
                out_layer = layer;
                out_uv = tex_coord;

                // 2. Sample Height (Central Difference Method)
                // Get exact texture size for accurate sampling steps
                ivec3 size = textureSize(heightMap, 0);
                float tex_w = 1.0 / float(size.x);
                float tex_h = 1.0 / float(size.y);

                // Sample Center (for position)
                float h_c = texture(heightMap, vec3(tex_coord, layer)).r;

                // Sample Neighbors (Left, Right, Up, Down)
                float h_l = texture(heightMap, vec3(tex_coord + vec2(-tex_w, 0), layer)).r;
                float h_r = texture(heightMap, vec3(tex_coord + vec2( tex_w, 0), layer)).r;
                float h_d = texture(heightMap, vec3(tex_coord + vec2(0, -tex_h), layer)).r; // Down
                float h_u = texture(heightMap, vec3(tex_coord + vec2(0,  tex_h), layer)).r; // Up

                height_factor = h_c;

                // 3. Calculate Normal
                // We span 2 texels (Left to Right), so the world distance is double
                float patch_width = distance(gl_in[0].gl_Position.xyz, gl_in[1].gl_Position.xyz);
                float world_step_x = 2.0 * patch_width * tex_w;
                float world_step_z = 2.0 * patch_width * tex_h;

                // Slope X = (Right - Left) * HeightScale
                // Slope Z = (Up - Down)    * HeightScale
                float dHdX = (h_r - h_l) * MAX_HEIGHT;
                float dHdZ = (h_u - h_d) * MAX_HEIGHT;

                // The normal is the cross product of the tangent vectors.
                // Simplified result of cross((step_x, dHdX, 0), (0, dHdZ, step_z)):
                v_normal = normalize(vec3(-dHdX, world_step_x, -dHdZ));

                // 4. Interpolate Position
                vec4 p1 = mix(gl_in[0].gl_Position, gl_in[1].gl_Position, gl_TessCoord.x);
                vec4 p2 = mix(gl_in[3].gl_Position, gl_in[2].gl_Position, gl_TessCoord.x);
                vec4 pos = mix(p1, p2, gl_TessCoord.y);

                pos.y += h_c * MAX_HEIGHT; 

                v_world_pos = pos.xyz;
                gl_Position = ubo.proj * ubo.view * pos;
            }
        ",
    }
}
pub mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
        #version 450
        // ... inputs ...
        layout(location = 0) in float height_factor;
        layout(location = 1) in vec3 v_world_pos;
        layout(location = 2) in vec3 v_normal;
        layout(location = 3) in flat uint in_layer; // <--- You need to pass the layer from TES to FS!
        layout(location = 4) in vec2 in_uv;

        layout(location = 0) out vec4 f_color;

        layout(set = 0, binding = 2) uniform sampler2DArray colorMap;

        const vec3 LIGHT_DIR = normalize(vec3(0.5, 0.8, 0.5));

        void main() {
            // ... normal calc ...
            vec3 normal = normalize(v_normal);
            float NdotL = max(dot(normal, LIGHT_DIR), 0.0);
            float light = NdotL + 0.8; 

            // SAMPLING
            // We need UVs here. You might need to pass `in_uv` from TES to FS if you haven't already.
            // Assuming you add `layout(location = 4) in vec2 in_uv;`
            vec3 albedo = texture(colorMap, vec3(in_uv, float(in_layer))).rgb;

            f_color = vec4(albedo * light, 1.0);
        }
    ",
    }
}
/*
pub mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
        #version 450
        layout(location = 0) in float height_factor;
        layout(location = 1) in vec3 v_world_pos;
        layout(location = 2) in vec3 v_normal;
        layout(location = 3) in flat uint in_layer;

        layout(set = 0, binding = 2) uniform sampler2DArray colorMap;
        
        layout(location = 0) out vec4 f_color;

        const vec3 LIGHT_DIR = normalize(vec3(0.5, 0.8, 0.5));

        void main() {
            // Use the smooth normal passed from TES
            vec3 normal = normalize(v_normal);

            // 2. Lighting
            float NdotL = max(dot(normal, LIGHT_DIR), 0.0);
            float light = NdotL + 0.3; // Ambient

            // ... (Rest of your color logic remains the same) ...

            vec3 c_water = vec3(0.0, 0.2, 0.6);
            vec3 c_grass = vec3(0.1, 0.5, 0.1); 
            vec3 c_rock  = vec3(0.4, 0.35, 0.3);
            vec3 c_snow  = vec3(1.0, 1.0, 1.0);

            vec3 terrain_color;
            
            if (height_factor < 0.01) {
                terrain_color = c_water;
            } else if (height_factor < 0.2) {
                terrain_color = c_grass;
            } else if (height_factor < 0.4) {
                terrain_color = c_rock;
            } else {
                terrain_color = c_snow;
            }

            f_color = vec4(terrain_color * light, 1.0);
        }
    ",
    }
}
*/
