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

// --- NEW: Helper to decode Mapbox Terrain-RGB ---
float get_height(vec2 uv, uint layer) {
    // 1. Sample the RGB values (0.0 to 1.0)
    vec3 rgb = texture(heightMap, vec3(uv, layer)).rgb;

    // 2. Denormalize to Integers (0 to 255)
    float r = rgb.r * 255.0;
    float g = rgb.g * 255.0;
    float b = rgb.b * 255.0;

    // 3. Decode: height = -10000 + ((R * 256 * 256 + G * 256 + B) * 0.1)
    float height_meters = -10000.0 + ((r * 65536.0 + g * 256.0 + b) * 0.1);

    // 4. Convert Meters to Kilometers (Your World Unit)
    return height_meters / 1000.0;
}

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

    // --- NEW: Use helper function instead of .r access ---
    float h_c = get_height(tex_coord, layer);

    // Sample Neighbors (Left, Right, Up, Down) for Normal Calc
    float h_l = get_height(tex_coord + vec2(-tex_w, 0), layer);
    float h_r = get_height(tex_coord + vec2( tex_w, 0), layer);
    float h_d = get_height(tex_coord + vec2(0, -tex_h), layer);
    float h_u = get_height(tex_coord + vec2(0,  tex_h), layer);

    height_factor = h_c;

    // 3. Calculate Normal
    // We span 2 texels (Left to Right), so the world distance is double
    float patch_width = distance(gl_in[0].gl_Position.xyz, gl_in[1].gl_Position.xyz);
    float world_step_x = 2.0 * patch_width * tex_w;
    float world_step_z = 2.0 * patch_width * tex_h;

    // --- NEW: Remove MAX_HEIGHT multiplication ---
    // The height values are already in World Units (km), so the difference is correct as-is.
    float dHdX = h_r - h_l;
    float dHdZ = h_u - h_d;

    // The normal is the cross product of the tangent vectors.
    vec3 tangent_x = vec3(world_step_x, dHdX, 0.0);
    vec3 tangent_z = vec3(0.0, dHdZ, -world_step_z); // Z is negative 'up' in texture space usually

    v_normal = normalize(cross(tangent_z, tangent_x));

    // 4. Calculate Final Position
    // Interpolate grid position from control points
    vec4 p1 = mix(gl_in[0].gl_Position, gl_in[1].gl_Position, gl_TessCoord.x);
    vec4 p2 = mix(gl_in[3].gl_Position, gl_in[2].gl_Position, gl_TessCoord.x);
    vec4 pos = mix(p1, p2, gl_TessCoord.y);

    // Apply height
    pos.y = h_c;
    v_world_pos = pos.xyz;

    gl_Position = ubo.proj * ubo.view * pos;
}
