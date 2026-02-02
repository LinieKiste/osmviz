#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
layout(location = 2) in float scale;

layout(location = 0) out vec3 v_color;
layout(location = 1) out vec3 v_normal;
layout(location = 2) out vec2 v_uv; // Pass UV to fragment

layout(set = 0, binding = 0) uniform CameraUniform {
    mat4 view;
    mat4 proj;
    vec3 cam_pos;
    float padding;
} camera;

// Structure to hold Pos + UV
struct Vertex {
    vec3 pos;
    vec2 uv;
};

// two quads
const Vertex VERTICES[12] = Vertex[12](
    Vertex(vec3(-0.5, 0.0, 0.0), vec2(0.0, 1.0)), 
    Vertex(vec3( 0.5, 0.0, 0.0), vec2(1.0, 1.0)), 
    Vertex(vec3( 0.5, 2.0, 0.0), vec2(1.0, 0.0)),
    Vertex(vec3(-0.5, 0.0, 0.0), vec2(0.0, 1.0)), 
    Vertex(vec3( 0.5, 2.0, 0.0), vec2(1.0, 0.0)), 
    Vertex(vec3(-0.5, 2.0, 0.0), vec2(0.0, 0.0)),
    Vertex(vec3(0.0, 0.0, -0.5), vec2(0.0, 1.0)), 
    Vertex(vec3(0.0, 0.0,  0.5), vec2(1.0, 1.0)), 
    Vertex(vec3(0.0, 2.0,  0.5), vec2(1.0, 0.0)),
    Vertex(vec3(0.0, 0.0, -0.5), vec2(0.0, 1.0)), 
    Vertex(vec3(0.0, 2.0,  0.5), vec2(1.0, 0.0)), 
    Vertex(vec3(0.0, 2.0, -0.5), vec2(0.0, 0.0))
);

void main() {
    Vertex v = VERTICES[gl_VertexIndex];
    vec3 local_pos = v.pos;

    vec3 world_pos = position + (local_pos * scale * 0.005);
    
    gl_Position = camera.proj * camera.view * vec4(world_pos, 1.0);
    v_color = color;
    v_normal = normalize(local_pos + vec3(0.0, 0.5, 0.0));
    v_uv = v.uv; // Pass through
}
