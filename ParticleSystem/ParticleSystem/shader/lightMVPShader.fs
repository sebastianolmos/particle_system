#version 330 core
out vec4 FragColor;

vec3 ka = vec3(0.3f, 0.3f, 0.3f);
vec3 kd = vec3(0.6f, 0.6f, 0.6f);
vec3 ks = vec3(0.7f, 0.7f, 0.7f);
float sh = 32.0f;

uniform vec3 lightDirection;
uniform vec3 viewPos;
vec3 la = vec3(1.0f, 1.0f, 1.0f);
vec3 ld = vec3(1.0f, 1.0f, 1.0f);
vec3 ls = vec3(1.0f, 1.0f, 1.0f);

in vec3 FragPos;  
in vec3 Normal;  
in vec3 Color;


void main()
{
    // ambient
    vec3 ambient = la * ka;
  	
    // diffuse 
    vec3 norm = normalize(Normal);
    //vec3 lightDir = normalize(light.position - FragPos);
    vec3 lightDir = normalize(-lightDirection); 
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = ld * (diff * kd);
    
    // specular
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);  
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), sh);
    vec3 specular = ls * (spec * ks);  
        
    vec3 result = ambient + (diffuse + specular) * Color;
    FragColor = vec4(result, 1.0);
}