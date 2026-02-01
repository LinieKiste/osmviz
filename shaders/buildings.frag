#version 450
layout(location = 0) in vec3 v_color;
layout(location = 1) in vec3 v_normal;

layout(location = 0) out vec4 f_color;

void main() {
    vec3 light_dir = normalize(vec3(0.5, 1.0, 0.2));

    float diff = max(dot(v_normal, light_dir), 0.0);

    float ambient = 0.4;

    vec3 final_color = v_color * (diff + ambient);

    f_color = vec4(final_color, 1.0);
}
