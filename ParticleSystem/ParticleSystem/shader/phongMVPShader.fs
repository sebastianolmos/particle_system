#version 330 core

in vec3 pColor;
in vec3 pPos;
in float pRadius; 


out vec4 FragColor;
uniform vec3 lightDir;
uniform vec3 camPos;
uniform vec3 camR;
uniform vec3 camU;

float border = 0.1f;

void main()
{
	// calculate normal from texture coordinates
	vec3 N;
	N.xy = gl_PointCoord.xy*vec2(2.0, -2.0) + vec2(-1.0, 1.0);
	float mag = dot(N.xy, N.xy);
	if(mag > 1.0)
		discard; // kill pixels outside circle
	N.z = sqrt(1.0-mag);
	vec3 norm = normalize(camR*N.x + camU*N.y + cross(camR, camU)*N.z);
    vec3 FragPos = pPos + norm * pRadius;
	// calculate lighting
    vec3 ka = vec3(0.3f, 0.3f, 0.3f);
    vec3 kd = vec3(0.6f, 0.6f, 0.6f);
    vec3 ks = vec3(0.7f, 0.7f, 0.7f);
    float sh = 32.0f;
    vec3 la = vec3(1.0f, 1.0f, 1.0f);
    vec3 ld = vec3(1.0f, 1.0f, 1.0f);
    vec3 ls = vec3(1.0f, 1.0f, 1.0f);
	// ambient
    vec3 ambient = la * ka;
  	
    // diffuse 
    //vec3 norm = normalize(Normal);
    //vec3 lightDir = normalize(light.position - FragPos);
    vec3 lightDir2 = normalize(-lightDir); 
    float diff = max(dot(norm, lightDir2), 0.0);
    vec3 diffuse = ld * (diff * kd);
    
    // specular
    vec3 viewDir = normalize(camPos - FragPos);
    vec3 reflectDir = reflect(-lightDir2, norm);  
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), sh);
    vec3 specular = ls * (spec * ks);  
        
    vec3 result = (ambient + diffuse + specular) * pColor;
    FragColor = vec4(result, 1.0);
}