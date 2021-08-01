#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"


#include "kernelParams.cuh"
#include "system.h"

class Menu {
public:
    ~Menu();
    void setWindow(GLFWwindow* window);
    void setMouseCallback(GLFWcursorposfun callback);
    void initMenu();
    void preRender();
    void render();

    void setGravity(float* gx, float* gy, float* gz)
    {
        gravityX = gx;
        gravityY = gy;
        gravityZ = gz;
    }

    void setGlobalDamping(float* gdamp)
    {
        globalDamping = gdamp;
    }

    void setBoundaryDamping(float* damp)
    {
        boundaryDamping = damp;
    }

    void setParticleDamping(float* damp)
    {
        particleDamping = damp;
    }

    void setSpring(float* value)
    {
        spring = value;
    }

    void setShear(float* value)
    {
        shear = value;
    }

    void setAttraction(float* att)
    {
        attraction = att;
    }

    void setCollideCheck(bool* boolean)
    {
        collideObject = boolean;
    }

    void setObjectSelector(unsigned int* selector)
    {
        objectToCollide = selector;
    }

    void setTimeStep(float* step)
    {
        timeStep = step;
    }

    void setPhongCheck(bool* ref)
    {
        activePhong = ref;
    }

    void setCollideObjectParams(float* pos, float* size)
    {
        pCollideObject = pos;
        sCollideObject = size;
    }

    void setSystem(System* ref)
    {
        pSystem = ref;
    }

    void updateNumParticles(uint num)
    {
        numParticles = num;
    }

private:
    GLFWwindow* appWindow;
    GLFWcursorposfun mouseCallback;
    float* timeStep;
    float* damping;
    float* gravityX;
    float* gravityY;
    float* gravityZ;
    bool* showSphere;
    float* globalDamping;
    float* boundaryDamping;
    float* particleDamping;
    float* spring;
    float* shear;
    float* attraction;
    bool* collideObject;
    unsigned int* objectToCollide;
    bool* activePhong;
    float* sCollideObject;
    float* pCollideObject;

    uint numParticles;
    System* pSystem;
};

Menu::~Menu()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}
void Menu::setWindow(GLFWwindow* window)
{
    appWindow = window;
}

void Menu::setMouseCallback(GLFWcursorposfun callback)
{
    mouseCallback = callback;
}

void Menu::initMenu()
{
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    //ImGui::StyleColorsClassic();

    // GL 3.0 + GLSL 130
    const char* glsl_version = "#version 130";

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(appWindow, true);
    ImGui_ImplOpenGL3_Init(glsl_version);
}

void Menu::preRender()
{
    // Start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void Menu::render()
{
    // Our state
    // ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    if (io.WantCaptureMouse)
    {
        glfwSetCursorPosCallback(appWindow, NULL);
    }
    else
    {
        glfwSetCursorPosCallback(appWindow, mouseCallback);
    }
    // 2. Show a simple window that we create ourselves. We use a Begin/End pair to created a named window.
    {
        static float f = 0.0f;

        ImGui::Begin("Menu");                          // Create a window called "Hello, world!" and append into it.

        ImGui::SliderFloat("TimeStep", timeStep, 0.0f, 1.0f);
        ImGui::Separator();
        // ImGui::Text("Parametros:");               // Display some text (you can use a format strings too)
        if (ImGui::CollapsingHeader("Gravity"))
        {
            ImGui::PushID(1);
            ImGui::SliderFloat("X", gravityX, -0.01f, 0.01f);
            ImGui::SliderFloat("Y", gravityY, -0.01f, 0.01f);
            ImGui::SliderFloat("Z", gravityZ, -0.01f, 0.01f);
            ImGui::PopID();
        }

        ImGui::Separator();
        if (ImGui::CollapsingHeader("Dumping"))
        {
            ImGui::PushID(2);
            ImGui::SliderFloat("Global", globalDamping, 0.0001, 1.0f);
            ImGui::SliderFloat("Box", boundaryDamping, 0.0001, 1.00f);
            ImGui::SliderFloat("Particles", particleDamping, 0.0001, 1.00f);
            ImGui::PopID();
        }
        ImGui::Separator();

        if (ImGui::CollapsingHeader("Collide Params"))
        {
            ImGui::PushID(3);
            ImGui::SliderFloat("Spring", spring, 0.0001, 1.0f);
            ImGui::SliderFloat("Shear", shear, 0.0001, 1.0f);
            ImGui::SliderFloat("Attraction", attraction, 0.000, 1.0f);
            ImGui::PopID();
        }
        ImGui::Separator();

        ImGui::Checkbox("Object to Collide?", collideObject);
        ImGui::SameLine();
        ImGui::Checkbox("Phong llumination?", activePhong);
        if (*collideObject)
        {
            ImGui::PushID(10);
            ImGui::SliderFloat("Height", pCollideObject + 2, 0.0, 4.0f);
            ImGui::SliderFloat("Size", sCollideObject, 0.1, 4.0f);
            ImGui::PopID();
            ImGui::Separator();
        }
        


        if (ImGui::CollapsingHeader("Particle Configuration"))
        {
            ImGui::PushID(4);
            if (ImGui::Button("Plane"))
            {
                pSystem->reset(262144);
            }
            ImGui::SameLine();
            if (ImGui::Button("Cube"))
            {
                pSystem->resetCube(64);
            }
            ImGui::SameLine();
            if (ImGui::Button("Sphere"))
            {
                pSystem->resetSphere(64);
            }
            ImGui::SameLine();
            if (ImGui::Button("Random"))
            {
                pSystem->resetRandom(262144);
            }
            ImGui::SameLine();
            if (ImGui::Button("Obj"))
            {

            }
            ImGui::PopID();
        }
        ImGui::Separator();
        if (ImGui::CollapsingHeader("Instance Particles"))
        {
            ImGui::PushID(5);
            if (ImGui::Button("Plane"))
            {
                pSystem->addPlane(2.5f);
            }
            ImGui::SameLine();
            if (ImGui::Button("Cube"))
            {
                pSystem->addCube(2.5f);
            }
            ImGui::SameLine();
            if (ImGui::Button("Sphere"))
            {
                pSystem->addSphere(2.2f);
            }
            ImGui::SameLine();
            if (ImGui::Button("Obj"))
            {

            }
            ImGui::PopID();
        }
        
        //ImGui::SliderFloat("Damping", damping, 0.0f, 1.0f);
        //ImGui::Checkbox("Show spheres?", showSphere);      
           // Edit 1 float using a slider from 0.0f to 1.0f
        //ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats representing a color

        ImGui::Text("Number of particles: %d", numParticles);
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
        ImGui::End();
    }
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

