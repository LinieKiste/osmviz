#version 450

layout(location = 0) in vec2 position; // Your flat plane (0,0) to (1,1)
layout(location = 1) in vec2 tex_coords;

layout(location = 0) out vec2 out_tex_coords;

void main() {
    // Pass coordinates through to the Control Shader
    out_tex_coords = tex_coords;
    
    // Just pass the local object position (x, 0, z)
    // We don't apply View/Proj matrices here!
    gl_Position = vec4(position.x, 0.0, position.y, 1.0);
}
