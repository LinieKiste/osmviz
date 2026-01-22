// src/shaders.rs
pub mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
            #version 450
            layout(location = 0) in vec2 position;      // 0..1 Grid Mesh
            layout(location = 1) in vec2 world_offset;  // Meter offset (Instance)
            layout(location = 2) in uint texture_layer; // Texture ID (Instance)

            layout(location = 0) out vec2 out_uv;       // Pass 0..1 to FS for texture sampling
            layout(location = 1) out flat uint out_layer; 

            // Hardcoded size for now
            const float TILE_SIZE = 1.0;

            void main() {
                out_uv = position; 
                out_layer = texture_layer;

                // TRANSFORM TO WORLD SPACE HERE
                vec2 world_pos = (position * TILE_SIZE) + world_offset;

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
                    // Closer than 100m = Max Detail (64)
                    // Further than 3000m = Min Detail (2)
                    float level = mix(64.0, 2.0, clamp((dist - 3.0) / 15.0, 0.0, 1.0));

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
        // Displaces vertices based on heightmap (LearnOpenGL 'Tessellation Evaluation Shader')
        src: r"
            #version 450
            layout(quads, equal_spacing, ccw) in;

            layout(location = 0) in vec2 in_uv[];
            layout(location = 1) in flat uint in_layer[];

            layout(location = 0) out float height_factor;
            layout(location = 1) out vec3 v_world_pos;

            layout(set = 0, binding = 0) uniform Camera {
                mat4 view;
                mat4 proj;
                vec3 pos;
            } ubo;

            layout(set = 0, binding = 1) uniform sampler2DArray heightMap;

            const float MAX_HEIGHT = 0.5; 

            void main() {
                // 1. Interpolate UVs (Required for texture lookup)
                vec2 u1 = mix(in_uv[0], in_uv[1], gl_TessCoord.x);
                vec2 u2 = mix(in_uv[3], in_uv[2], gl_TessCoord.x);
                vec2 tex_coord = mix(u1, u2, gl_TessCoord.y);

                // 2. Sample Height
                float h_factor = texture(heightMap, vec3(tex_coord, float(in_layer[0]))).r;
                height_factor = h_factor; 

                // 3. Interpolate Position
                // gl_in[].gl_Position is ALREADY in World Space (X, 0, Z)
                vec4 p1 = mix(gl_in[0].gl_Position, gl_in[1].gl_Position, gl_TessCoord.x);
                vec4 p2 = mix(gl_in[3].gl_Position, gl_in[2].gl_Position, gl_TessCoord.x);
                vec4 pos = mix(p1, p2, gl_TessCoord.y);

                // 4. Apply Height
                pos.y += h_factor * MAX_HEIGHT; 

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
        layout(location = 0) in float height_factor; // Now receiving 0.0 to 1.0
        layout(location = 1) in vec3 v_world_pos;
        
        layout(location = 0) out vec4 f_color;

        const vec3 LIGHT_DIR = normalize(vec3(0.5, 0.8, 0.5));

        void main() {
            // 1. Calculate Normal (Same as before)
            vec3 dx = dFdx(v_world_pos);
            vec3 dy = dFdy(v_world_pos);
            vec3 normal = normalize(cross(dy, dx)); 

            // 2. Lighting
            float NdotL = max(dot(normal, LIGHT_DIR), 0.0);
            float light = NdotL + 0.3; // Ambient

            // 3. Colors based on Percentage (0.0 to 1.0)
            vec3 c_water = vec3(0.0, 0.2, 0.6);
            vec3 c_grass = vec3(0.1, 0.5, 0.1); 
            vec3 c_rock  = vec3(0.4, 0.35, 0.3);
            vec3 c_snow  = vec3(1.0, 1.0, 1.0);

            vec3 terrain_color;
            
            // Adjust these thresholds to match your specific image contrast
            if (height_factor < 0.01) {
                terrain_color = c_water;
            } else if (height_factor < 0.2) {
                // Lowlands (Grass)
                terrain_color = c_grass;
            } else if (height_factor < 0.4) {
                // Mid-Mountain (Grass -> Rock)
                float t = (height_factor - 0.2) / 0.5;
                // terrain_color = mix(c_grass, c_rock, t);
                terrain_color = c_rock;
            } else {
                // Peaks (Rock -> Snow)
                float t = clamp((height_factor - 0.7) / 0.3, 0.0, 1.0);
                // terrain_color = mix(c_rock, c_snow, t);
                terrain_color = c_snow;
            }

            f_color = vec4(terrain_color * light, 1.0);
        }
    ",
    }
}
