#ifndef KERNEL_SYSTEM_H_
#define KERNEL_SYSTEM_H_

#include <stdio.h>
#include <math.h>

#include "helper_math.h"
#include "kernelParams.cuh"

// Parametros de la simulacion
__constant__ kernelParams params;

struct integrate1
{
    float deltaTime;
    __host__ __device__
        integrate1(float delta_time) : deltaTime(delta_time) {}

    template <typename Tuple>
    __device__
        void operator()(Tuple t)
    {
        volatile float4 posData = thrust::get<0>(t);
        volatile float4 velData = thrust::get<1>(t);
        float3 pos = make_float3(posData.x, posData.y, posData.z);
        float3 vel = make_float3(velData.x, velData.y, velData.z);

        vel += params.gravity * deltaTime;
        vel *= params.globalDamping;

        // new position = old position + velocity * deltaTime
        pos += vel * deltaTime;

        // Check collision in the box boundary
        if (pos.x > params.boxSize.x - params.particleRadius)
        {
            pos.x = params.boxSize.x - params.particleRadius;
            vel.x *= params.boundaryDamping * -1.0f;
        }

        if (pos.x < 0.0f + params.particleRadius)
        {
            pos.x = 0.0f + params.particleRadius;
            vel.x *= params.boundaryDamping * -1.0f;
        }

        if (pos.y > params.boxSize.y - params.particleRadius)
        {
            pos.y = params.boxSize.y - params.particleRadius;
            vel.y *= params.boundaryDamping * -1.0f;
        }

        if (pos.y < 0.0f + params.particleRadius)
        {
            pos.y = 0.0f + params.particleRadius;
            vel.y *= params.boundaryDamping * -1.0f;
        }

        if (pos.z > params.boxSize.z - params.particleRadius)
        {
            pos.z = params.boxSize.z - params.particleRadius;
            vel.z *= params.boundaryDamping * -1.0f;
        }

        if (pos.z < 0.0f + params.particleRadius)
        {
            pos.z = 0.0f + params.particleRadius;
            vel.z *= params.boundaryDamping * -1.0f;
        }

        // store new position and velocity
        thrust::get<0>(t) = make_float4(pos, posData.w);
        thrust::get<1>(t) = make_float4(vel, velData.w);
    }
};

// Aqui iran los kernels
__global__ void integrate2(float4* pos, float4* vel, float deltaTime)
{
    uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    float4 p = pos[index];
    float4 v = vel[index];

    float3 nPos = make_float3(p.x, p.y, p.z);
    float3 nVel = make_float3(v.x, v.y, v.z);

    nVel += params.gravity * deltaTime;

    // new position = old position + velocity * deltaTime
    nPos += nVel * deltaTime;

    // write output vertex
    pos[index] = make_float4(nPos.x, nPos.y, nPos.z, p.w);
    vel[index] = make_float4(nVel.x, nVel.y, nVel.z, 1.0f);
}

#endif