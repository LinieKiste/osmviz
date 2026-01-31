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

    pos.y += h_c * 65.; 

    v_world_pos = pos.xyz;
    gl_Position = ubo.proj * ubo.view * pos;
}
