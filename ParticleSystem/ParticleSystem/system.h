#pragma once

#include <glad/glad.h>

#include "particleSystem.cuh"
#include "vector_functions.h"
#include "kernelParams.cuh"

#include <math.h>
#include <memory.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

using namespace std;

#ifndef M_SQR3
#define M_SQR3    0.57735026918962576450914878050196
#endif
#ifndef M_PI
#define M_PI    3.1415926535897932384626433832795
#endif

class System 
{
public:
	System(uint numParticles, uint3 gridSize);
	~System();
	
	void update(float deltaTime);
	float* getPosArray();
	float* getVelArray();
	void setPosArray(const float* data, int start, int count);
	void setVelArray(const float* data, int start, int count);
	void renderParticles();
	void createBox();
	void renderBox();

	int getNumParticles() const
	{
		return m_numParticles;
	}

	unsigned int getPosVBO() const
	{
		return m_posVBO;
	}

	void* getCudaPosVBO() const
	{
		return (void*)m_cudaPosVBO;
	}

	float getGlobalDamping()
	{
		return m_params.globalDamping;
	}

	void setGlobalDamping(float x)
	{
		m_params.globalDamping = x;
	}

	void setBoundaryDamping(float x)
	{
		m_params.boundaryDamping = x;
	}

	float3 getGravity() 
	{
		return m_params.gravity;
	}

	void setGravity(float x, float y, float z)
	{
		m_params.gravity = make_float3(x, y, z);
	}

	void setParticleDamping(float x)
	{
		m_params.boundaryDamping = x;
	}

	void setSpring(float x)
	{
		m_params.spring = x;
	}

	void setShear(float x)
	{
		m_params.shear = x;
	}

	void setAttraction(float x)
	{
		m_params.attraction = x;
	}

	float3 getCenter()
	{
		return m_params.center;
	}

	float getParticleRadius()
	{
		return m_params.particleRadius;
	}

	uint3 getGridSize()
	{
		return m_params.gridSize;
	}

	float3 getCellSize()
	{
		return m_params.cellSize;
	}

	float3 getCollideObjectPos()
	{
		return m_params.collidePos;
	}

	void setCollideObjectPos(float x, float y, float z)
	{
		m_params.collidePos = make_float3(x, y, z);
	}

	float getCollideObjectSize()
	{
		return m_params.collideSize;
	}

	void setCollideObjectSize(float value)
	{
		m_params.collideSize = value;
	}

	uint getCollideObjectShape()
	{
		return m_params.collideShape;
	}

	void setCollideObjectShape(uint shape)
	{
		m_params.collideShape = shape;
	}

	void setCoolliderDrag(bool value)
	{
		colliderDrag = value;
	}

	void reset();

	void createSphereCollider();
	void renderSphereCollider();
	void createBoxCollider();
	void renderBoxCollider();
	void setMousePos(float xPos, float yPos, glm::vec3 Right, glm::vec3 Front, float radius);

private:
	System() {};
	uint createVBO(uint size);
	uint createBuffers(uint size);

	void _initialize(int numParticles);
	void _finalize();

	void initGrid(uint* size, uint numParticles);

private:
	uint m_numParticles;

	float* m_hostPos;
	float* m_hostVel;

	float* m_devicePos;
	float* m_deviceVel;

	//float* m_deviceSortedVel;
	//uint* m_deviceGridCounters;
	//uint* m_deviceGridCells;
	float* m_deviceSortedPos;
	float* m_deviceSortedVel;
	uint* m_deviceGridParticleHash; // grid hash value for each particle
	uint* m_deviceGridParticleIndex;// particle index for each particle
	uint* m_deviceCellStart;        // index of start of each cell in sorted list
	uint* m_deviceCellEnd;          // index of end of cell

	uint m_posVAO;
	uint m_posVBO;
	float* m_cudaPosVBO;

	struct cudaGraphicsResource* m_cudaPosVboResource;

	kernelParams m_params;
	uint3 m_gridSize;
	uint m_numGridCells;

	uint m_boxVAO;
	uint m_boxVBO;
	uint m_boxEBO;

	// Collide objects params
	uint m_cSphereVAO;
	uint m_cSphereVBO;
	uint m_cSphereEBO;
	uint m_cSphereIndices;
	uint m_cBoxVAO;
	uint m_cBoxVBO;
	uint m_cBoxEBO;

	bool colliderDrag = false;
	glm::vec2 m_mousePos;
};

System::System(uint numParticles, uint3 gridSize) :
	m_numParticles(numParticles),
	m_hostPos(0),
	m_hostVel(0),
	m_devicePos(0),
	m_deviceVel(0),
	m_gridSize(gridSize)
{
	m_numGridCells = m_gridSize.x * m_gridSize.y * m_gridSize.z;
	m_params.particleRadius = 1.0f / 64.0f;
	float cellSize = m_params.particleRadius * 2.0f;  // cell size equal to particle diameter
	m_params.cellSize = make_float3(cellSize, cellSize, cellSize);
	m_params.gridSize = gridSize;
	m_params.center = make_float3(gridSize.x * cellSize / 2,
		gridSize.y * cellSize / 2,
		gridSize.z * cellSize / 2);

	m_params.gravity = make_float3(0.0f, 0.0f, -0.03f);
	m_params.globalDamping = 1.0f;
	m_params.boundaryDamping = 1.0f;
	m_params.boxSize = make_float3(m_params.cellSize.x * m_params.gridSize.x,
		m_params.cellSize.y * m_params.gridSize.y,
		m_params.cellSize.z * m_params.gridSize.z);
	m_params.spring = 0.5f;
	m_params.damping = 0.02f;
	m_params.shear = 0.1f;
	m_params.attraction = 0.0f;
	_initialize(numParticles);
}

System::~System()
{
	_finalize();
	m_numParticles = 0;
}

float* System::getPosArray()
{
	float* hdata = m_hostPos;
	float* ddata = m_devicePos;
	struct cudaGraphicsResource* cuda_vbo_resource = m_cudaPosVboResource;
	copyArrayFromDevice(hdata, ddata, &cuda_vbo_resource, m_numParticles * 4 * sizeof(float));
	return hdata;
}

float* System::getVelArray()
{
	float* hdata = m_hostVel;
	float* ddata = m_deviceVel;
	struct cudaGraphicsResource* cuda_vbo_resource = 0;
	copyArrayFromDevice(hdata, ddata, &cuda_vbo_resource, m_numParticles * 4 * sizeof(float));
	return hdata;
}

void System::setPosArray(const float* data, int start, int count)
{
	unregisterGLBufferObject(m_cudaPosVboResource);
	glBindBuffer(GL_ARRAY_BUFFER, m_posVBO);
	glBufferSubData(GL_ARRAY_BUFFER, start * 4 * sizeof(float), count * 4 * sizeof(float), data);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	registerGLBufferObject(m_posVBO, &m_cudaPosVboResource);
}

void System::setVelArray(const float* data, int start, int count)
{
	copyArrayToDevice(m_deviceVel, data, start * 4 * sizeof(float), count * 4 * sizeof(float));
}

uint System::createVBO(uint size)
{
	GLuint vbo;
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	return vbo;
}

uint System::createBuffers(uint size)
{
	GLuint vbo;
	glGenVertexArrays(1, &m_posVAO);
	glGenBuffers(1, &vbo);
	glBindVertexArray(m_posVAO);

	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	return vbo;
}

void System::reset()
{
	uint gridSize[3];
	gridSize[0] = m_gridSize.x;
	gridSize[1] = m_gridSize.y;
	gridSize[2] = m_gridSize.z;
	initGrid(gridSize, m_numParticles);
	setPosArray(m_hostPos, 0, m_numParticles);
	setVelArray(m_hostVel, 0, m_numParticles);
}

void System::_initialize(int numParticles)
{
    m_numParticles = numParticles;

    // allocate host storage
    m_hostPos = new float[m_numParticles * 4];
    m_hostVel = new float[m_numParticles * 4];
    memset(m_hostPos, 0, m_numParticles * 4 * sizeof(float));
    memset(m_hostVel, 0, m_numParticles * 4 * sizeof(float));

    // allocate GPU data
    unsigned int memSize = sizeof(float) * 4 * m_numParticles;


    m_posVBO = createBuffers(memSize);
    registerGLBufferObject(m_posVBO, &m_cudaPosVboResource);
	// position attribute
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	// color attribute
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// Allocate in device
    allocateArray((void**)&m_deviceVel, memSize);
	
	//allocateArray((void**)&m_deviceSortedVel, memSize);
		//allocateArray((void**)&m_deviceGridCounters, sizeof(uint) * 1 * m_numGridCells);
	//allocateArray((void**)&m_deviceGridCells, sizeof(uint) * 4 * m_numGridCells);

	allocateArray((void**)&m_deviceSortedPos, memSize);
	allocateArray((void**)&m_deviceSortedVel, memSize);

	allocateArray((void**)&m_deviceGridParticleHash, m_numParticles * sizeof(uint));
	allocateArray((void**)&m_deviceGridParticleIndex, m_numParticles * sizeof(uint));

	allocateArray((void**)&m_deviceCellStart, m_numGridCells * sizeof(uint));
	allocateArray((void**)&m_deviceCellEnd, m_numGridCells * sizeof(uint));

    setParameters(&m_params);
}

void System::_finalize()
{
    delete[] m_hostPos;
    delete[] m_hostVel;

    freeArray(m_deviceVel);

	freeArray(m_deviceSortedPos);
	freeArray(m_deviceSortedVel);

	freeArray(m_deviceGridParticleHash);
	freeArray(m_deviceGridParticleIndex);
	freeArray(m_deviceCellStart);
	freeArray(m_deviceCellEnd);

	unregisterGLBufferObject(m_cudaPosVboResource);
	unregisterGLBufferObject(m_cudaPosVboResource);
	glDeleteBuffers(1, (const GLuint*)&m_posVBO);
}

// step the simulation
void System::update(float deltaTime)
{ 
    float* dPos;

	dPos = (float*)mapGLBufferObject(&m_cudaPosVboResource);

    // update constants
    setParameters(&m_params);

    // integrate
	integrateInDevice1(
        dPos,
        m_deviceVel,
        deltaTime,
        m_numParticles);

	// Order grid
	//updateGridInDevice(dPos, m_deviceGridCounters, m_deviceGridCells, m_numParticles, m_numGridCells);
	// Swap
	//swapVelInDevice(m_deviceSortedVel, m_deviceVel, m_numParticles);
	// Collision
	//collideInDevice(m_deviceVel, dPos, m_deviceSortedVel, m_deviceGridCounters, m_deviceGridCells, m_numParticles);

	// calculate grid hash
	calcHashInDevice(
		m_deviceGridParticleHash,
		m_deviceGridParticleIndex,
		dPos,
		m_numParticles);

	// sort particles based on hash
	sortParticlesInDevice(m_deviceGridParticleHash, 
		m_deviceGridParticleIndex, 
		m_numParticles);

	// reorder particle arrays into sorted order and
	// find start and end of each cell
	reorderParticlesInDevice(
		m_deviceCellStart,
		m_deviceCellEnd,
		m_deviceSortedPos,
		m_deviceSortedVel,
		m_deviceGridParticleHash,
		m_deviceGridParticleIndex,
		dPos,
		m_deviceVel,
		m_numParticles,
		m_numGridCells);

	// process collisions
	collideWithHashInDevice(
		m_deviceVel,
		m_deviceSortedPos,
		m_deviceSortedVel,
		m_deviceGridParticleIndex,
		m_deviceCellStart,
		m_deviceCellEnd,
		m_numParticles,
		m_numGridCells);


	unmapGLBufferObject(m_cudaPosVboResource);
}

void System::createBox()
{
	float3 boxSize = make_float3(m_params.cellSize.x * m_params.gridSize.x,
		m_params.cellSize.y * m_params.gridSize.y, 
		m_params.cellSize.z * m_params.gridSize.z);
	float3 color = make_float3(1.0f, 1.0f, 1.0f);

	float vertices[] = {
		 // positions                       // colors
		 0.0f,      0.0f,      0.0f,       color.x, color.y, color.z,  // top right
		 boxSize.x, 0.0f,      0.0f,       color.x, color.y, color.z,  // bottom right
		 boxSize.x, boxSize.y, 0.0f,       color.x, color.y, color.z,  // bottom left
		 0.0f,      boxSize.y, 0.0f,       color.x, color.y, color.z,  // bottom left

		 0.0f,      0.0f,      boxSize.z,  color.x, color.y, color.z,  // top right
		 boxSize.x, 0.0f,      boxSize.z,  color.x, color.y, color.z,  // bottom right
		 boxSize.x, boxSize.y, boxSize.z,  color.x, color.y, color.z,  // bottom left
		 0.0f,      boxSize.y, boxSize.z,  color.x, color.y, color.z  // bottom left
	};

	unsigned int indices[] = {  // note that we start from 0!
		0, 1,   4, 5,
		1, 2,   5, 6,
		2, 3,   6, 7,
		3, 0,   7, 4,

		0, 4,
		1, 5,
		2, 6,
		3, 7
	};

	glGenVertexArrays(1, &m_boxVAO);
	glGenBuffers(1, &m_boxVBO);
	glGenBuffers(1, &m_boxEBO);
	// bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
	glBindVertexArray(m_boxVAO);

	glBindBuffer(GL_ARRAY_BUFFER, m_boxVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_boxEBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

	// position attribute
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	// color attribute
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);
}

void System::renderBox()
{
	glBindVertexArray(m_boxVAO);
	glDrawElements(GL_LINES, 24, GL_UNSIGNED_INT, 0);
}

inline float lerp(float a, float b, float t)
{
	return a + t * (b - a);
}

inline float packColor(int r, int g, int b)
{
	return r + g * 256 + b * 256 * 256;
}
// create a color ramp
void colorRamp(float t, float* r)
{
	const int ncolors = 7;
	float c[ncolors][3] =
	{
		{ 1.0, 0.0, 0.0, },
		{ 1.0, 0.5, 0.0, },
		{ 1.0, 1.0, 0.0, },
		{ 0.0, 1.0, 0.0, },
		{ 0.0, 1.0, 1.0, },
		{ 0.0, 0.0, 1.0, },
		{ 1.0, 0.0, 1.0, },
	};
	t = t * (ncolors - 1);
	int i = (int)t;
	float u = t - floor(t);
	r[0] = lerp(c[i][0], c[i + 1][0], u);
	r[1] = lerp(c[i][1], c[i + 1][1], u);
	r[2] = lerp(c[i][2], c[i + 1][2], u);
}

void System::initGrid(uint* size, uint numParticles)
{
	float height = 1.0f;
	for (uint z = 0; z < size[2]; z++)
	{
		for (uint y = 0; y < size[1]; y++)
		{
			for (uint x = 0; x < size[0]; x++)
			{
				uint i = (z * size[1] * size[0]) + (y * size[0]) + x;

				if (i < numParticles)
				{
					if (z == 0) {
						m_hostPos[i * 4 + 0] = (x * 2.0f * m_params.particleRadius) + m_params.particleRadius + 0.01f ;
						m_hostPos[i * 4 + 1] = (y * 2.0f * m_params.particleRadius) + m_params.particleRadius + 0.01f;
					}
					else
					{
						m_hostPos[i * 4 + 0] = (x * 2.0f * m_params.particleRadius) + m_params.particleRadius;
						m_hostPos[i * 4 + 1] = (y * 2.0f * m_params.particleRadius) + m_params.particleRadius;
					}
					
					m_hostPos[i * 4 + 2] = (z * 2.0f * m_params.particleRadius) + m_params.particleRadius + height;

					float color[3] = {};
					colorRamp((float)y/((float)size[1]), color);
					m_hostPos[i * 4 + 3] = packColor(color[0]*255, color[1] * 255, color[2] * 255);

					m_hostVel[i * 4 + 0] = 0.0f;
					m_hostVel[i * 4 + 1] = 0.0f;
					m_hostVel[i * 4 + 2] = 0.0f;
					m_hostVel[i * 4 + 3] = 0.0f;
				}
			}
		}
	}
}

void System::renderParticles()
{
	glBindVertexArray(m_posVAO);
	glDrawArrays(GL_POINTS, 0, m_numParticles);
}

void System::createSphereCollider()
{
	int N = 20;
	int vertexSize = 9;
	int indexSize = 3;

	m_cSphereIndices = N * ((int)N / 2) * indexSize * 2;
	float* vertices = new float[N * ((int)N/2) * vertexSize * 4];
	uint *indices = new uint[N * ((int)N / 2) * indexSize * 2];
	float dTheta = 2 * M_PI / N;
	float dPhi = 2 * M_PI / N;
	float rho = 0.5;
	float3 color = make_float3(0.7, 0.7, 0.7);
	int c = 0;

	for (int i = 0; i < ((int)N / 2); i++)
	{
		float theta = i * dTheta;
		float theta1 = (i + 1) * dTheta;
		for (int j = 0; j < N; j++)
		{
			float phi = j * dPhi;
			float phi1 = (j + 1) * dPhi;

			float tempv1[] = {
				rho * glm::sin(theta) * glm::cos(phi), rho * glm::sin(theta) * glm::sin(phi), rho * glm::cos(theta),
				color.x, color.y, color.z,
				glm::sin(theta)* glm::cos(phi), glm::sin(theta)* glm::sin(phi), glm::cos(theta)
			};
			float tempv2[] = {
				rho * glm::sin(theta1) * glm::cos(phi), rho * glm::sin(theta1) * glm::sin(phi), rho * glm::cos(theta1),
				color.x, color.y, color.z,
				glm::sin(theta1) * glm::cos(phi), glm::sin(theta1) * glm::sin(phi), glm::cos(theta1)
			};
			float tempv3[] = {
				rho * glm::sin(theta1) * glm::cos(phi1), rho * glm::sin(theta1) * glm::sin(phi1), rho * glm::cos(theta1),
				color.x, color.y, color.z,
				glm::sin(theta1) * glm::cos(phi1), glm::sin(theta1) * glm::sin(phi1), glm::cos(theta1)
			};
			float tempv4[] = {
				rho * glm::sin(theta) * glm::cos(phi1), rho * glm::sin(theta) * glm::sin(phi1), rho * glm::cos(theta),
				color.x, color.y, color.z,
				glm::sin(theta) * glm::cos(phi1), glm::sin(theta) * glm::sin(phi1), glm::cos(theta)
			};
			int offset = (j + i * N) * 4 * 9;
			copy(tempv1, tempv1 + (1 * vertexSize), (vertices + (offset + vertexSize * 0) ));
			copy(tempv2, tempv2 + (1 * vertexSize), (vertices + (offset + vertexSize * 1)));
			copy(tempv3, tempv3 + (1 * vertexSize), (vertices + (offset + vertexSize * 2)));
			copy(tempv4, tempv4 + (1 * vertexSize), (vertices + (offset + vertexSize * 3)));

			uint tempi1[] = { c + 0, c + 1, c + 2 };
			uint tempi2[] = { c + 2, c + 3, c + 0 };
			int idxOffset = (j + i * N) * 2 * 3;
			copy(tempi1, tempi1 + (1 * indexSize), (indices + (idxOffset + indexSize * 0)));
			copy(tempi2, tempi2 + (1 * indexSize), (indices + (idxOffset + indexSize * 1)));

			c = c + 4;
		}
	}


	glGenVertexArrays(1, &m_cSphereVAO);
	glGenBuffers(1, &m_cSphereVBO);
	glGenBuffers(1, &m_cSphereEBO);

	glBindVertexArray(m_cSphereVAO);

	glBindBuffer(GL_ARRAY_BUFFER, m_cSphereVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * N * ((int)N / 2) * vertexSize * 4, vertices, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_cSphereEBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(uint) * m_cSphereIndices, indices, GL_STATIC_DRAW);

	// position attribute
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	// color attribute
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);
	// normal attribute
	glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(6 * sizeof(float)));
	glEnableVertexAttribArray(2);

	glBindVertexArray(0);
}

void System::renderSphereCollider()
{
	glBindVertexArray(m_cSphereVAO);
	glDrawElements(GL_TRIANGLES, m_cSphereIndices, GL_UNSIGNED_INT, 0);
}

void System::setMousePos(float xPos, float yPos, glm::vec3 Right, glm::vec3 Front, float radius)
{
	glm::vec2 pos2d{ xPos, yPos };
	if (colliderDrag)
	{
		glm::vec2 delta = (pos2d - m_mousePos) * -1.0f;
		glm::vec3 center = glm::vec3( m_params.collidePos.x, m_params.collidePos.y, m_params.collidePos.z);
		center += Right * delta.x * 2.0f * radius / 4.0f
			+ Front * delta.y * 2.0f * radius / 4.0f;
		m_params.collidePos = make_float3(center.x, center.y, center.z);
	}

	m_mousePos.x = xPos;
	m_mousePos.y = yPos;
}
