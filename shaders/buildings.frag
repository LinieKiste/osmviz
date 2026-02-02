#version 450

layout(location = 0) in vec3 v_color;
layout(location = 1) in vec3 v_world_pos;

layout(location = 0) out vec4 f_color;

void main() {
    // Calculate normal using derivatives
    // This creates a flat-shaded look (perfect for low-poly buildings)
    vec3 dx = dFdx(v_world_pos);
    vec3 dy = dFdy(v_world_pos);
    vec3 normal = normalize(cross(dy, dx));

    // Basic lighting setup
    vec3 light_dir = normalize(vec3(0.5, -1.0, -0.5)); // Arbitrary sun direction
    float diff = max(dot(normal, -light_dir), 0.0);
    
    // Add some ambient light so shadows aren't pitch black
    vec3 ambient = vec3(0.3);
    vec3 final_color = v_color * (diff + ambient);

    f_color = vec4(final_color, 1.0);
}
