#include <glad/glad.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_gl_interop.h>

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "shader/shader.h"
#include "utils/camera3d.h"
#include "utils/performanceMonitor.h"
#include "menu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <iostream>

#ifndef M_PI
#define M_PI    3.1415926535897932384626433832795
#endif

using namespace std;

extern "C" void runCuda(struct cudaGraphicsResource** vbo_resource, float time, unsigned int width, unsigned int height);
extern "C" void runTest(unsigned int width, unsigned int height);
extern "C" void registerWithCuda(struct cudaGraphicsResource** resource, GLuint vbo);
extern "C" void unRegisterWithCuda(cudaGraphicsResource_t vbo_res);


void runDisplay();
void deleteVBO(GLuint* vbo, struct cudaGraphicsResource* vbo_res);

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow* window, bool* points);

const unsigned int window_width = 1280;
const unsigned int window_height = 720;
const unsigned int mesh_width = 256;
const unsigned int mesh_height = 256;

// camera
Camera3D camera(glm::vec3(0.0f, 0.0f, 0.0f));

//menu
Menu menu = Menu();

// vbo variables
GLuint vbo;
struct cudaGraphicsResource* cuda_vbo_resource;

// timing
float deltaTime = 0.0f;	// time between current frame and last frame


int main()
{
    bool cudaTest = false;

    if (cudaTest)
        runTest(mesh_width, mesh_height);
    else
        runDisplay();
}


void runDisplay()
{
    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    string title = "3D Points interop CUDA";
    // glfw window creation
    // --------------------
    GLFWwindow* window = glfwCreateWindow(window_width, window_height, title.c_str(), NULL, NULL);
    if (window == NULL)
    {
        cout << "Failed to create GLFW window" << endl;
        glfwTerminate();
        return;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetScrollCallback(window, scroll_callback);

    menu.setWindow(window);
    menu.setMouseCallback(mouse_callback);

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        cout << "Failed to initialize GLAD" << endl;
        return;
    }

    // build and compile our shader program
    // ------------------------------------
    Shader sphereShader("shader/sphereMVPShader.vs", "shader/sphereMVPShader.fs");

    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------

    unsigned int VAO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &vbo);
    // bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
    glBindVertexArray(VAO);

    // create buffer object
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    // initialize buffer object
    unsigned int size = mesh_width * mesh_height * 3 * sizeof(float);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

    // register this buffer object with CUDA
    registerWithCuda(&cuda_vbo_resource, vbo);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);


    glBindBuffer(GL_ARRAY_BUFFER, 0);

    runCuda(&cuda_vbo_resource, 0.0f, mesh_width, mesh_height);

    // You can unbind the VAO afterwards so other VAO calls won't accidentally modify this VAO, but this rarely happens. Modifying other
    // VAOs requires a call to glBindVertexArray anyways so we generally don't unbind VAOs (nor VBOs) when it's not directly necessary.
    glBindVertexArray(0);

    float t1 = (float)glfwGetTime();
    float t0 = (float)glfwGetTime();

    float timer = 0.0f;
    bool points = false;

    menu.initMenu();

    PerformanceMonitor pMonitor(glfwGetTime(), 0.5f);

    // render loop
    // -----------
    while (!glfwWindowShouldClose(window))
    {
        t1 = (float)glfwGetTime();
        deltaTime = t1 - t0;
        t0 = t1;

        pMonitor.update(glfwGetTime());
        stringstream ss;
        ss << title << " " << pMonitor;
        glfwSetWindowTitle(window, ss.str().c_str());

        timer += deltaTime * 1.0f;

        // input
        // -----
        processInput(window, &points);

        runCuda(&cuda_vbo_resource, timer, mesh_width, mesh_height);

        // render
        // ------
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        menu.preRender();

        glEnable(GL_PROGRAM_POINT_SIZE);
        glEnable(GL_DEPTH_TEST);
        // render the triangle
        sphereShader.use();
        //glPointSize(100.0f);

        // pass projection matrix to shader (note that in this case it could change every frame)
        glm::mat4 projection = glm::perspective(glm::radians(camera.Fovy), (float)window_width / (float)window_height, 0.1f, 100.0f);
        sphereShader.setMat4("projection", projection);

        // camera/view transformation
        glm::mat4 view = camera.GetViewMatrix();
        sphereShader.setMat4("view", view);

        glm::mat4 model = glm::mat4(1.0f); // make sure to initialize matrix to identity matrix first
        sphereShader.setMat4("model", model);


        if (points)
        {
            sphereShader.setFloat("pointRadius", 1);
            sphereShader.setFloat("pointScale", 1);
        }
        else {
            sphereShader.setFloat("pointRadius", 0.125f * 0.5f);
            sphereShader.setFloat("pointScale", window_height / glm::tan(camera.Fovy * 0.5f * (float)M_PI / 180.0f));
        }
        sphereShader.setVec3("Color", glm::vec3(1.0f, 0.0f, 0.0f));
        sphereShader.setVec3("lightDir", glm::vec3(1.0f, 1.0f, 0.0f));

        glBindVertexArray(VAO);
        glDrawArrays(GL_POINTS, 0, mesh_width * mesh_height);

        menu.render();

        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // optional: de-allocate all resources once they've outlived their purpose:
    // ------------------------------------------------------------------------
    glDeleteVertexArrays(1, &VAO);
    deleteVBO(&vbo, cuda_vbo_resource);

    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    glfwTerminate();

    return;
}

void deleteVBO(GLuint* vbo, struct cudaGraphicsResource* vbo_res)
{

    // unregister this buffer object with CUDA
    unRegisterWithCuda(vbo_res);

    glBindBuffer(1, *vbo);
    glDeleteBuffers(1, vbo);

    *vbo = 0;
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow* window, bool* points)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.ProcessKeyboardMovement(FORWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.ProcessKeyboardMovement(BACKWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.ProcessKeyboardMovement(LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.ProcessKeyboardMovement(RIGHT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
        camera.ProcessKeyboardMovement(ORIGIN, deltaTime);

    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
        camera.ProcessKeyboardRotation(AZIM_UP, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
        camera.ProcessKeyboardRotation(AZIM_DOWN, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
        camera.ProcessKeyboardRotation(ZEN_LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
        camera.ProcessKeyboardRotation(ZEN_RIGHT, deltaTime);

    if (glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS)
        *points = true;

    if (glfwGetKey(window, GLFW_KEY_P) == GLFW_RELEASE)
        *points = false;
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
    {
        camera.SetRotDrag(true);
    }

    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE)
    {
        camera.SetRotDrag(false);
    }

    if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS)
    {
        camera.SetCenterDrag(true);
    }

    if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_RELEASE)
    {
        camera.SetCenterDrag(false);
    }
}


// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
    float posX = 2 * (xpos - window_width / 2) / window_width;
    float posY = 2 * (window_height / 2 - ypos) / window_height;
    camera.SetCurrentMousePos(posX, posY);
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    camera.ProcessMouseScroll(yoffset);
}