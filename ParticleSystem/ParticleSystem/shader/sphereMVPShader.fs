#version 330 core
out vec4 FragColor;

uniform vec3 Color;
uniform vec3 lightDir;
float border = 0.1f;

void main()
{
	// calculate normal from texture coordinates
	vec3 N;
	N.xy = gl_PointCoord* 2.0 - vec2(1.0);
	float mag = dot(N.xy, N.xy);
	if(mag > 1.0)
		discard; // kill pixels outside circle
	N.z = sqrt(1.0-mag);
	// calculate lighting
	float diffuse = max(0.0, dot(lightDir, N));
	vec4 ambient = vec4(0.3f, 0.0f, 0.0f, 0.0f);
	
	vec4 result = vec4(0.0f, 0.0f, 0.0f, 0.0f);
	if (mag < 1.0 - border){
		result = ambient + vec4(Color,1) * diffuse;
	}

	FragColor = result;
}