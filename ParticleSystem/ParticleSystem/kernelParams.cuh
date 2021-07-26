
#ifndef KERNEL_PARAMS_H
#define KERNEL_PARAMS_H

#include "vector_types.h"

typedef unsigned int uint;

struct kernelParams
{
    float3 gravity;
    float globalDamping;
    float particleRadius;

    uint3 gridSize;
    uint numCells;
    float3 cellSize;
    float3 center;
    float3 boxSize;

    uint maxParticlesPerCell;

    float spring;
    float damping;
    float shear;
    float attraction;
    float boundaryDamping;
};

#endif