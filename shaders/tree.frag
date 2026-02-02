#version 450
#extension GL_EXT_nonuniform_qualifier : enable

layout(location = 0) in vec3 v_color;
layout(location = 1) in vec3 v_normal;
layout(location = 2) in vec2 v_uv;

layout(location = 0) out vec4 f_color;

// Array of separate samplers (Fixed size for simplicity, e.g., 4 types of trees)
layout(set = 0, binding = 1) uniform sampler2D tex_array[];

void main() {
    vec4 tex_color = texture(tex_array[0], v_uv);

    if (tex_color.a < 0.5) {
        discard;
    }

    vec3 light_dir = normalize(vec3(0.4, 1.0, -0.2));
    float diff = max(dot(v_normal, light_dir), 0.4);
    
    f_color = vec4(v_color * tex_color.rgb * diff, 1.0);
}
