#ifndef CAMERA3D_H
#define CAMERA3D_H

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vector>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>

// Defines several possible options for camera movement. Used as abstraction to stay away from window-system specific input methods
enum Camera_Movement {
    FORWARD,
    BACKWARD,
    LEFT,
    RIGHT,
    ORIGIN
};

enum Camera_Rotation {
    AZIM_UP,
    AZIM_DOWN,
    ZEN_LEFT,
    ZEN_RIGHT
};

// Default camera values
const float SPEED = 20.5f;
const float SENSITIVITY = 0.1f;
const float ZOOM = 45.0f;
const float FOVY = 45.0f;
const float THETA = 50.0f;
const float PHI = 0.0f;
const float ROTATION = 75.0f;
const float ROTATION_SENS = 60.0f;
const float CENTER_SENS = 2.0f;


// An abstract camera class that processes input and calculates the corresponding Euler Angles, Vectors and Matrices for use in OpenGL
class Camera3D
{
public:
    // camera Attributes
    glm::vec3 Position;
    glm::vec3 Center;
    glm::vec3 Front;
    glm::vec3 Up;
    glm::vec3 Right;
    glm::vec3 WorldUp;
    float Phi;
    float Theta;
    // camera options
    float MovementSpeed;
    float rotationSpeed;
    float MouseSensitivity;
    float Zoom;
    float Fovy;
    float Radius;
    glm::vec2 LastMousePos;
    glm::vec2 CurrentMousePos;
    glm::vec3 lastCenterPos;
    float centerSensitivty;
    bool centerDrag;
    float lastTheta;
    float lastPhi;
    float rotSensitivity;
    bool rotDrag;
    float screenOffset;
    bool vertDrag;

    // constructor with vectors
    Camera3D(glm::vec3 center = glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3 up = glm::vec3(0.0f, 0.0f, 1.0f), float rad = 30.0f, float theta = THETA) :
        MovementSpeed(SPEED),
        rotationSpeed(ROTATION),
        Phi(PHI), Fovy(FOVY),
        Front(glm::vec3(1.0f, 0.0f, 0.0f)),
        Right(glm::vec3(0.0f, 1.0f, 0.0f)),
        LastMousePos(glm::vec2(0.0f)),
        CurrentMousePos(glm::vec2(0.0f)),
        centerDrag(false),
        rotDrag(false),
        vertDrag(false),
        lastTheta(THETA),
        lastPhi(PHI),
        rotSensitivity(ROTATION_SENS),
        centerSensitivty(CENTER_SENS)
    {
        screenOffset = -0.5;
        Center = center;
        WorldUp = up;
        Radius = rad;
        Theta = theta;
        updateCameraVectors();
    }
    // constructor with scalar values
    Camera3D(float posX, float posY, float posZ, float upX, float upY, float upZ, float rad, float theta) : MovementSpeed(SPEED), rotationSpeed(ROTATION), Phi(PHI), Fovy(FOVY)
    {
        Center = glm::vec3(posX, posY, posZ);
        WorldUp = glm::vec3(upX, upY, upZ);
        Radius = rad;
        Theta = theta;
        updateCameraVectors();
    }

    // returns the view matrix calculated using Euler Angles and the LookAt Matrix
    glm::mat4 GetViewMatrix()
    {

        Position.x = Center.x + Radius * cos(glm::radians(Phi)) * sin(glm::radians(Theta));
        Position.y = Center.y + Radius * sin(glm::radians(Phi)) * sin(glm::radians(Theta));
        Position.z = Center.z + Radius * cos(glm::radians(Theta));

        return glm::lookAt(Position, Center, WorldUp);
    }

    // processes input received from any keyboard-like input system. Accepts input parameter in the form of camera defined ENUM (to abstract it from windowing systems)
    void ProcessKeyboardMovement(Camera_Movement direction, float deltaTime)
    {
        float velocity = MovementSpeed * deltaTime;
        if (direction == FORWARD)
            Center += Front * velocity;
        if (direction == BACKWARD)
            Center -= Front * velocity;
        if (direction == LEFT)
            Center -= Right * velocity;
        if (direction == RIGHT)
            Center += Right * velocity;
        if (direction == ORIGIN)
            Center = glm::vec3(0.0f);
    }

    // processes input received from any keyboard-like input system. Accepts input parameter in the form of camera defined ENUM (to abstract it from windowing systems)
    void ProcessKeyboardRotation(Camera_Rotation direction, float deltaTime)
    {
        float velocity = rotationSpeed * deltaTime;
        if (direction == AZIM_UP)
            Theta -= velocity;
        if (direction == AZIM_DOWN)
            Theta += velocity;
        if (direction == ZEN_LEFT)
            Phi -= velocity;
        if (direction == ZEN_RIGHT)
            Phi += velocity;

        if (Theta > 179.0f)
            Theta = 179.0f;
        if (Theta < 01.0f)
            Theta = 01.0f;

        updateCameraVectors();
    }

    void setCenter(glm::vec3 newCenter)
    {
        Center = newCenter;
    }

    void setRadius(float r)
    {
        Radius = r;
    }

    // processes input received from a mouse scroll-wheel event. Only requires input on the vertical wheel-axis
    void ProcessMouseScroll(float yoffset)
    {
        if (CurrentMousePos.x > screenOffset)
        {
            Radius -= (float)yoffset * 0.2f;
            if (Radius < 1.0f)
                Radius = 1.0f;
            if (Radius > 100.0f)
                Radius = 100.0f;
        }

    }

    void SetRotDrag(bool value)
    {
        rotDrag = value;
    }

    void SetCenterDrag(bool value)
    {
        centerDrag = value;
    }

    void SetVerticalDrag(bool value)
    {
        vertDrag = value;
    }

    void SetCurrentMousePos(float xPos, float yPos)
    {
        glm::vec2 pos2d{ xPos, yPos };
        if (rotDrag && (xPos > screenOffset))
        {
            glm::vec2 delta = (pos2d - CurrentMousePos);
            Phi -= delta.x * rotSensitivity;
            Theta += delta.y * rotSensitivity;
        }
        if (Theta > 179.0f)
            Theta = 179.0f;
        if (Theta < 01.0f)
            Theta = 01.0f;

        if (centerDrag && (yPos > screenOffset))
        {
            glm::vec2 delta = (pos2d - CurrentMousePos) * -1.0f;
            Center += Right * delta.x * centerSensitivty * Radius / 4.0f
                + Front * delta.y * centerSensitivty * Radius / 4.0f;
        }
        else if (vertDrag)
        {
            glm::vec2 delta = (pos2d - CurrentMousePos) * -1.0f;
            Center += Up * delta.y * centerSensitivty * Radius / 4.0f;
        }

        CurrentMousePos.x = xPos;
        CurrentMousePos.y = yPos;
        updateCameraVectors();
    }

private:
    // calculates the front vector from the Camera's (updated) Euler Angles
    void updateCameraVectors()
    {
        // calculate the new Front vector
        glm::vec3 front = glm::normalize(Center - Position);
        Front = glm::vec3(front.x, front.y, 0.0f);
        Front = glm::normalize(Front);
        // also re-calculate the Right and Up vector
        Right = glm::normalize(glm::cross(Front, WorldUp));  // normalize the vectors, because their length gets closer to 0 the more you look up or down which results in slower movement.
        Up = glm::normalize(glm::cross(Right, Front));
    }

};
#endif