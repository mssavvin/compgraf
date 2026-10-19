#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;

layout(push_constant) uniform Push {
	mat4 mvp;
	vec4 tint;
} pc;

layout(location = 0) out vec4 out_color;

void main() {
	gl_Position = pc.mvp * vec4(in_position, 1.0);
	out_color = vec4(in_color, 1.0) * pc.tint;
}