#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"

class Menu {
public:
    ~Menu();
    void setWindow(GLFWwindow* window);
    void setMouseCallback(GLFWcursorposfun callback);
    void initMenu();
    void preRender();
    void render();
private:
    GLFWwindow* appWindow;
    GLFWcursorposfun mouseCallback;
    float* damping;
    float* gravity;
    bool* showSphere;
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

        ImGui::Text("Parametros:");               // Display some text (you can use a format strings too)
        ImGui::SliderFloat("Gravity", gravity, 0.0001f, 0.001f);
        ImGui::SliderFloat("Damping", damping, 0.0f, 1.0f);
        ImGui::Checkbox("Show spheres?", showSphere);      
           // Edit 1 float using a slider from 0.0f to 1.0f
        //ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats representing a color

        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
        ImGui::End();
    }
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

