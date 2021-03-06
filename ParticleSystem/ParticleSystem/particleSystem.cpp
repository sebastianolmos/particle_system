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

#define GRID_SIZE_X     128
#define GRID_SIZE_Y     128
#define GRID_SIZE_Z     128
#define NUM_PARTICLES   262144
#define MAX_NUM_PARTICLES   524288

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

float timeStep = 1.0f;
glm::vec3 gravity = glm::vec3(0.0f, 0.0f, -0.0003f);
float globalDump = 1.0f;
float boundaryDump = 0.75f;
float particleDump = 0.015f;
float spring = 0.32f;
float shear = 0.09f;
float attraction = 0.0f;
bool collideObject = false;
uint objectToCollide = 1;
bool activePhong = false;
float pCollideObject[3] = { 0.0f, 0.0f, 0.0f };
float sCollideObject = 1.0f;

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
    psystem = new System(numParticles, gridSize, MAX_NUM_PARTICLES);
    psystem->reset(NUM_PARTICLES);
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
    uint gridDim = GRID_SIZE_X;
    gridSize.x = GRID_SIZE_X;
    gridSize.y = GRID_SIZE_Y;
    gridSize.z = GRID_SIZE_Z;

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


    Shader phongSphereShader("shader/phongMVPShader.vs", "shader/phongMVPShader.fs");

    camera.setCenter(glm::vec3(psystem->getCenter().x,
        psystem->getCenter().y,
        psystem->getCenter().z/4.0f));
    camera.setRadius(psystem->getCenter().x * 2.0f);


    // Create collision box with lines
    Shader boxShader("shader/basicMVPShader.vs", "shader/basicMVPShader.fs");
    psystem->createBox();

    // Create collision shapes
    Shader dirLightShader("shader/lightMVPShader.vs", "shader/lightMVPShader.fs");
    psystem->createSphereCollider();

    // Set system params
    psystem->setGravity(gravity.x, gravity.y, gravity.z);
    psystem->setBoundaryDamping(boundaryDump);
    psystem->setGlobalDamping(globalDump);
    psystem->setParticleDamping(particleDump);
    psystem->setSpring(spring);
    psystem->setShear(shear);
    psystem->setAttraction(attraction);
    psystem->setColliderPosRef(pCollideObject);
    // Configuration shapes
    psystem->createVoxelShape1("../assets/cat.txt");
    psystem->createVoxelShape2("../assets/charizard.txt");
    psystem->createVoxelShape3("../assets/dragon.txt");
    psystem->createVoxelShape4("../assets/eiffel.txt");
    psystem->createVoxelShape5("../assets/guardian.txt");
    psystem->createVoxelShape6("../assets/hogwarts.txt");
    psystem->createVoxelShape7("../assets/knight.txt");
    // Shapes to add
    psystem->createVoxelShape8("../assets/spacecraft.txt");
    psystem->createVoxelShape9("../assets/amongUs.txt");
    psystem->createVoxelShape10("../assets/snorlax.txt");
    psystem->createVoxelShape11("../assets/dinosaur.txt");

    // Set Menu Params
    menu.setGravity(&gravity.x, &gravity.y, &gravity.z);
    menu.setGlobalDamping(&globalDump);
    menu.setBoundaryDamping(&boundaryDump);
    menu.setParticleDamping(&particleDump);
    menu.setSpring(&spring);
    menu.setAttraction(&attraction);
    menu.setShear(&shear);
    menu.setCollideCheck(&collideObject);
    menu.setObjectSelector(&objectToCollide);
    menu.updateNumParticles(psystem->getNumParticles());
    menu.setTimeStep(&timeStep);
    menu.setPhongCheck(&activePhong);
    menu.setCollideObjectParams(pCollideObject, &sCollideObject);
    menu.setSystem(psystem);

    float t1 = (float)glfwGetTime();
    float t0 = (float)glfwGetTime();
    float timeCounter = 0.0f;

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
        timeCounter += deltaTime;

        if (timeCounter > 1.0 / 60.0f) {

            pMonitor.update(glfwGetTime());
            stringstream ss;
            ss << title << " " << pMonitor;
            glfwSetWindowTitle(window, ss.str().c_str());

            timer += deltaTime * 1.0f;

            // input
            // -----
            processInput(window, &points);

            psystem->setGravity(gravity.x, gravity.y, gravity.z);
            psystem->setBoundaryDamping(boundaryDump);
            psystem->setGlobalDamping(globalDump);
            psystem->setParticleDamping(particleDump);
            psystem->setSpring(spring);
            psystem->setShear(shear);
            psystem->setAttraction(attraction);
            psystem->setCollideObjectPos(pCollideObject[0], pCollideObject[1], pCollideObject[2]);
            psystem->setCollideObjectSize(sCollideObject);
            psystem->setCollideObjectShape(objectToCollide);

            psystem->update(timeStep);

            // render
            // ------
            glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            menu.preRender();

            glEnable(GL_PROGRAM_POINT_SIZE);
            glEnable(GL_DEPTH_TEST);
            // render the triangle


            //glPointSize(100.0f);

            // pass projection matrix to shader (note that in this case it could change every frame)
            glm::mat4 projection = glm::perspective(glm::radians(camera.Fovy), (float)window_width / (float)window_height, 0.1f, 100.0f);

            // camera/view transformation
            glm::mat4 view = camera.GetViewMatrix();

            glm::mat4 model = glm::mat4(1.0f); // make sure to initialize matrix to identity matrix first
            glm::vec3 lightDirection = glm::vec3(-1.0f, -1.0f, -1.0f);


            if (!activePhong)
            {
                sphereShader.use();
                sphereShader.setMat4("view", view);
                sphereShader.setMat4("projection", projection);
                sphereShader.setMat4("model", model);
                sphereShader.setFloat("pointRadius", psystem->getParticleRadius());
                sphereShader.setFloat("pointScale", window_height / glm::tan(camera.Fovy * 0.5f * (float)M_PI / 180.0f));
                sphereShader.setVec3("lightDir", lightDirection);
            }
            else
            {
                phongSphereShader.use();
                phongSphereShader.setMat4("view", view);
                phongSphereShader.setMat4("projection", projection);
                phongSphereShader.setMat4("model", model);
                phongSphereShader.setFloat("pointRadius", psystem->getParticleRadius());
                phongSphereShader.setFloat("pointScale", window_height / glm::tan(camera.Fovy * 0.5f * (float)M_PI / 180.0f));
                phongSphereShader.setVec3("lightDir", lightDirection);
                phongSphereShader.setVec3("camPos", camera.Position);
                phongSphereShader.setVec3("camR", camera.Right);
                phongSphereShader.setVec3("camU", camera.Up);
            }

            psystem->renderParticles();

            // Render Collider object
            if (collideObject)
            {
                // sphere case
                dirLightShader.use();
                dirLightShader.setMat4("projection", projection);
                dirLightShader.setMat4("view", view);
                float3 sPos = psystem->getCollideObjectPos();
                glm::mat4 cSmodel = glm::mat4(1.0f);
                cSmodel = glm::translate(cSmodel, glm::vec3(pCollideObject[0], pCollideObject[1], pCollideObject[2]));
                float collSize = psystem->getCollideObjectSize();
                cSmodel = glm::scale(cSmodel, glm::vec3(sCollideObject, sCollideObject, sCollideObject));
                dirLightShader.setMat4("model", cSmodel);
                dirLightShader.setVec3("lightDirection", glm::vec3(-1.0f, -1.0f, -1.0f));
                dirLightShader.setVec3("viewPos", camera.Position);
                psystem->renderSphereCollider();
            }
            else {
                pCollideObject[0] = -0.5f;
                pCollideObject[1] = 2.0f;
                pCollideObject[2] = 0.0f;
                sCollideObject = 0.5;
            }


            // Render the collide box
            boxShader.use();
            boxShader.setMat4("projection", projection);
            boxShader.setMat4("view", view);
            glm::mat4 boxModel = glm::mat4(1.0f); // make sure to initialize matrix to identity matrix first
            boxShader.setMat4("model", boxModel);
            psystem->renderBox();

            menu.updateNumParticles(psystem->getNumParticles());
            menu.render();

            // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
            // -------------------------------------------------------------------------------
            glfwSwapBuffers(window);
            glfwPollEvents();

            timeCounter = 0;
        }
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
        if (!collideObject)
        {
            camera.SetCenterDrag(true);
        }
        else {
            psystem->setCoolliderDrag(true);
        }
    }

    if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_RELEASE)
    {
        psystem->setCoolliderDrag(false);
        camera.SetCenterDrag(false);
    }

    if (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS)
    {
        camera.SetVerticalDrag(true);
    }

    if (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_RELEASE)
    {
        camera.SetVerticalDrag(false);
    }
}


// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
    float posX = 2 * (xpos - window_width / 2) / window_width;
    float posY = 2 * (window_height / 2 - ypos) / window_height;
    camera.SetCurrentMousePos(posX, posY);
    if (collideObject)
    {
        psystem->setMousePos(posX, posY, camera.nRight, camera.nFront, camera.Radius);
    }
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    camera.ProcessMouseScroll(yoffset);
}