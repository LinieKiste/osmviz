#version 450
layout(location = 0) in vec3 position;

layout(set = 0, binding = 0) uniform CameraUniform {
    mat4 view;
    mat4 proj;
    vec3 view_pos;
    float padding;
} camera;

void main() {
    // Basic MVP transform
    gl_Position = camera.proj * camera.view * vec4(position, 1.0);
}
