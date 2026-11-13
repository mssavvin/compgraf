#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;

layout(location = 0) out vec3 v_world_pos;
layout(location = 1) out vec3 v_world_normal;

// set 0 Ч данные сцены
layout(set = 0, binding = 0) uniform SceneUniforms {
    mat4 view_proj;
    vec4 camera_pos;
    vec4 ambient;        // rgb * intensity
    vec4 dir_direction;  // xyz Ч куда светит направленный источник
    vec4 dir_color;      // rgb * intensity
    ivec4 counts;        // x Ч точечных, y Ч прожекторных
} scene;

// set 1 Ч данные модели
layout(set = 1, binding = 0) uniform ModelUniforms {
    mat4 model;
    mat4 normal_matrix;  // transpose(inverse(model))
    vec4 albedo;
    vec4 specular;       // rgb Ч цвет блика, w Ч коэффициент блика
} object;

void main() {
    vec4 world = object.model * vec4(in_position, 1.0);
    v_world_pos = world.xyz;
    v_world_normal = mat3(object.normal_matrix) * in_normal;
    gl_Position = scene.view_proj * world;
}