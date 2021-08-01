#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in float aColor;

out vec3 pColor;
out vec3 pPos;
out float pRadius;

uniform float pointRadius; 
uniform float pointScale;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

vec3 unpackColor(float f) 
{
    vec3 color;
    color.b = floor(f / 256.0 / 256.0);
    color.g = floor((f - color.b * 256.0 * 256.0) / 256.0);
    color.r = floor(f - color.b * 256.0 * 256.0 - color.g * 256.0);
    // now we have a vec3 with the 3 components in range [0..255]. Let's normalize it!
    return color / 255.0;
}

void main()
{	
	vec3 posEye = vec3( view * vec4(aPos, 1.0));
	float dist = length(posEye);
	gl_PointSize = pointRadius * (pointScale/dist);
	gl_Position = projection * view * model * vec4(aPos, 1.0f);
	pColor = unpackColor(aColor);
    pPos = vec3(model * vec4(aPos, 1.0));
    pRadius = pointRadius;
}