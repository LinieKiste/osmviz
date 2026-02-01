#version 450
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 color;

layout(location = 0) out vec3 v_color;
layout(location = 1) out vec3 v_normal;

layout(set = 0, binding = 0) uniform CameraUniform {
    mat4 view;
    mat4 proj;
    vec3 view_pos;
    float padding;
} camera;

void main() {
    v_color = color;
    v_normal = normal;
    
    gl_Position = camera.proj * camera.view * vec4(position, 1.0);
}
