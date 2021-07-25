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
#include "system.h"

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <iostream>

#ifndef M_PI
#define M_PI    3.1415926535897932384626433832795
#endif

#define GRID_SIZE       64
#define NUM_PARTICLES   16384

using namespace std;

void runDisplay();
void deleteVBO(GLuint* vbo, struct cudaGraphicsResource* vbo_res);

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow* window, bool* points);

const unsigned int window_width = 1280;
const unsigned int window_height = 720;

uint numParticles = 0;
uint3 gridSize;

float damping = 1.0f;
glm::vec3 gravity = glm::vec3(0.0f, 0.0f, -0.0003f);
int ballr = 10;

// camera
Camera3D camera(glm::vec3(0.0f, 0.0f, 0.0f));

//menu
Menu menu = Menu();

System* psystem = 0;

// vbo variables
GLuint vbo;
struct cudaGraphicsResource* cuda_vbo_resource;

// timing
float deltaTime = 0.0f;	// time between current frame and last frame

// initialize particle system
void initParticleSystem(int numParticles, uint3 gridSize)
{
    psystem = new System(numParticles, gridSize);
    psystem->reset();
}

void cleanup()
{
    if (psystem)
    {
        delete psystem;
    }
    return;
}

int main()
{
    bool cudaTest = false;
    runDisplay();
}

void initParams()
{

}

void runDisplay()
{
    numParticles = NUM_PARTICLES;
    uint gridDim = GRID_SIZE;
    gridSize.x = gridSize.y = gridSize.z = gridDim;

    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    string title = "3D Particle System";
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

    initParticleSystem(numParticles, gridSize);
    initParams();

    camera.setCenter(glm::vec3(psystem->getCenter().x,
        psystem->getCenter().y,
        psystem->getCenter().z/4.0f));
    camera.setRadius(psystem->getCenter().x * 2.0f);

    // Create collision box with lines
    Shader boxShader("shader/basicMVPShader.vs", "shader/basicMVPShader.fs");
    psystem->createBox();

    float gdump = 1.0f;
    float bdump = 1.0f;
    // Set Menu Params
    menu.setGravity(&gravity.x, &gravity.y, &gravity.z);
    menu.setGlobalDamping(&gdump);
    menu.setBoundaryDamping(&bdump);

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

        psystem->setGlobalDamping(gdump);
        psystem->setBoundaryDamping(bdump);
        psystem->setGravity(gravity.x, gravity.y, gravity.z);
        psystem->update(deltaTime);

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
            sphereShader.setFloat("pointRadius", psystem->getParticleRadius());
            sphereShader.setFloat("pointScale", window_height / glm::tan(camera.Fovy * 0.5f * (float)M_PI / 180.0f));
        }
        sphereShader.setVec3("lightDir", glm::vec3(1.0f, 1.0f, 0.0f));

        psystem->renderParticles();

        // Render the collide box
        boxShader.use();
        boxShader.setMat4("projection", projection);
        boxShader.setMat4("view", view);
        glm::mat4 boxModel = glm::mat4(1.0f); // make sure to initialize matrix to identity matrix first
        boxShader.setMat4("model", boxModel);
        psystem->renderBox();

        menu.render();

        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // optional: de-allocate all resources once they've outlived their purpose:
    // ------------------------------------------------------------------------
    //glDeleteVertexArrays(1, &VAO);

    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    glfwTerminate();

    return;
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