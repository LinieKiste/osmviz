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
