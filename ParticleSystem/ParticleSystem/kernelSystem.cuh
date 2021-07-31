#ifndef KERNEL_SYSTEM_H_
#define KERNEL_SYSTEM_H_

#include <stdio.h>
#include <math.h>

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

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

// calculate position in uniform grid
__device__ int calcGridIndex(float3 p)
{
    int3 gridPos;
    gridPos.x = floor((p.x) / params.cellSize.x);
    gridPos.y = floor((p.y) / params.cellSize.y);
    gridPos.z = floor((p.z) / params.cellSize.z);
    return __umul24(__umul24(gridPos.z, params.gridSize.y), params.gridSize.x) + __umul24(gridPos.y, params.gridSize.x) + gridPos.x;
}

// calculate position in uniform grid
__device__ int3 calcGridPos(float3 p)
{
    int3 gridPos;
    gridPos.x = floor((p.x) / params.cellSize.x);
    gridPos.y = floor((p.y) / params.cellSize.y);
    gridPos.z = floor((p.z) / params.cellSize.z);
    return gridPos;
}

// calculate address in grid from position (clamping to edges)
__device__ uint calcGridHash(int3 gridPos)
{
    gridPos.x = gridPos.x & (params.gridSize.x - 1);  // wrap grid, assumes size is power of 2
    gridPos.y = gridPos.y & (params.gridSize.y - 1);
    gridPos.z = gridPos.z & (params.gridSize.z - 1);
    return __umul24(__umul24(gridPos.z, params.gridSize.y), params.gridSize.x) + __umul24(gridPos.y, params.gridSize.x) + gridPos.x;
}

__global__ void updateGrid(float4* pos, uint* gridCounters, uint* gridCells, uint numParticles)
{
    uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (index >= numParticles) return;

    volatile float4 p = pos[index];
    int cellIdx = calcGridIndex(make_float3(p.x, p.y, p.z));
    int amount = atomicAdd(&gridCounters[cellIdx], 1.0f);

    __syncthreads();
    gridCounters[cellIdx] = min(gridCounters[cellIdx], 4);

    if (amount < 4)
    {
        gridCells[4 * cellIdx + amount] = index;
    }
}

// collide two spheres using DEM method
__device__
float3 collideSpheres(float3 posA, float3 posB,
    float3 velA, float3 velB,
    float radiusA, float radiusB,
    float attraction)
{
    // calculate relative position
    float3 relPos = posB - posA;

    float dist = length(relPos);
    float collideDist = radiusA + radiusB;

    float3 force = make_float3(0.0f);

    if (dist < collideDist)
    {
        float3 norm = relPos / dist;

        // relative velocity
        float3 relVel = velB - velA;

        // relative tangential velocity
        float3 tanVel = relVel - (dot(relVel, norm) * norm);

        // spring force
        force = -params.spring * (collideDist - dist) * norm;
        // dashpot (damping) force
        force += params.damping * relVel;
        // tangential shear force
        force += params.shear * tanVel;
        // attraction
        force += attraction * relPos;
    }

    return force;
}

__device__ float3 collideWithCell(int3 gridPos, uint index, float3 pos, float3 vel, float4* oldPos, float4* oldVel, uint* gridCounters, uint* gridCells)
{
    uint gridHash = calcGridHash(gridPos);
    float3 force = make_float3(0.0f);

    // get start of bucket for this cell
    uint cellIdx = gridCounters[cellIdx];
    if (cellIdx != 0x00000000)          // cell is not empty
    {
        // iterate over particles in this cell
        for (uint j = 0; j < cellIdx; j++)
        {
            uint cell = gridCells[4 * gridHash + j];
            if (cell != index)                // check not colliding with self
            {
                float3 pos2 = make_float3(oldPos[cell]);
                float3 vel2 = make_float3(oldVel[cell]);

                // collide two spheres
                force += collideSpheres(pos, pos2, vel, vel2, params.particleRadius, params.particleRadius, params.attraction);
            }
        }
    }
    return force;
}

__global__ void swapVel(float4* newVel, float4* oldVel, uint numParticles)
{
    uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (index >= numParticles) return;
    newVel[index] = oldVel[index];
}


__global__ void collideParticles(float4* newVel, float4* oldPos, float4* oldVel, uint* gridCounters, uint* gridCells, uint numParticles)
{
    uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (index >= numParticles) return;
    // read particle data from sorted arrays
    float3 pos = make_float3(oldPos[index]);
    float3 vel = make_float3(oldVel[index]);
    // get address in grid
    int3 gridPos = calcGridPos(pos);

    // examine neighbouring cells
    float3 force = make_float3(0.0f);

    for (int z = -1; z <= 1; z++)
    {
        for (int y = -1; y <= 1; y++)
        {
            for (int x = -1; x <= 1; x++)
            {
                int3 neighbourPos = gridPos + make_int3(x, y, z);
                force += collideWithCell(neighbourPos, index, pos, vel, oldPos, oldVel, gridCounters, gridCells);
            }
        }
    }
    newVel[index] = make_float4(vel + force, 0.0f);
}

// Functions to collide with hash sort

// calculate grid hash value for each particle
__global__
void calcHashD(uint* gridParticleHash,  // output
               uint* gridParticleIndex, // output
               float4* pos,               // input: positions
               uint    numParticles)
{
    uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (index >= numParticles) return;

    volatile float4 p = pos[index];

    // get address in grid
    int3 gridPos = calcGridPos(make_float3(p.x, p.y, p.z));
    uint hash = calcGridHash(gridPos);

    // store grid hash and particle index
    gridParticleHash[index] = hash;
    gridParticleIndex[index] = index;
}

// rearrange particle data into sorted order, and find the start of each cell
// in the sorted hash array
__global__
void reorderDataAndFindCellStartD(uint* cellStart,          // output: cell start index
                                  uint* cellEnd,            // output: cell end index
                                  float4* sortedPos,        // output: sorted positions
                                  float4* sortedVel,        // output: sorted velocities
                                  uint* gridParticleHash,   // input: sorted grid hashes
                                  uint* gridParticleIndex,  // input: sorted particle indices
                                  float4* oldPos,           // input: sorted position array
                                  float4* oldVel,           // input: sorted velocity array
                                  uint    numParticles)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    extern __shared__ uint sharedHash[];    // blockSize + 1 elements
    uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    uint hash;

    // handle case when no. of particles not multiple of block size
    if (index < numParticles)
    {
        hash = gridParticleHash[index];

        // Load hash data into shared memory so that we can look
        // at neighboring particle's hash value without loading
        // two hash values per thread
        sharedHash[threadIdx.x + 1] = hash;

        if (index > 0 && threadIdx.x == 0)
        {
            // first thread in block must load neighbor particle hash
            sharedHash[0] = gridParticleHash[index - 1];
        }
    }

    cg::sync(cta);

    if (index < numParticles)
    {
        // If this particle has a different cell index to the previous
        // particle then it must be the first particle in the cell,
        // so store the index of this particle in the cell.
        // As it isn't the first particle, it must also be the cell end of
        // the previous particle's cell

        if (index == 0 || hash != sharedHash[threadIdx.x])
        {
            cellStart[hash] = index;

            if (index > 0)
                cellEnd[sharedHash[threadIdx.x]] = index;
        }

        if (index == numParticles - 1)
        {
            cellEnd[hash] = index + 1;
        }

        // Now use the sorted index to reorder the pos and vel data
        uint sortedIndex = gridParticleIndex[index];
        float4 pos = oldPos[sortedIndex];
        float4 vel = oldVel[sortedIndex];

        sortedPos[index] = pos;
        sortedVel[index] = vel;
    }

}

// collide a particle against all other particles in a given cell
__device__
float3 collideWithCellHash(int3  gridPos,
                           uint    index,
                           float3  pos,
                           float3  vel,
                           float4* oldPos,
                           float4* oldVel,
                           uint* cellStart,
                           uint* cellEnd)
{
    uint gridHash = calcGridHash(gridPos);

    // get start of bucket for this cell
    uint startIndex = cellStart[gridHash];

    float3 force = make_float3(0.0f);

    if (startIndex != 0xffffffff)          // cell is not empty
    {
        // iterate over particles in this cell
        uint endIndex = cellEnd[gridHash];

        for (uint j = startIndex; j < endIndex; j++)
        {
            if (j != index)                // check not colliding with self
            {
                float3 pos2 = make_float3(oldPos[j]);
                float3 vel2 = make_float3(oldVel[j]);

                // collide two spheres
                force += collideSpheres(pos, pos2, vel, vel2, params.particleRadius, params.particleRadius, params.attraction);
            }
        }
    }

    return force;
}


__global__
void collideParticlesHash(float4* newVel,               // output: new velocity
                          float4* oldPos,               // input: sorted positions
                          float4* oldVel,               // input: sorted velocities
                          uint* gridParticleIndex,      // input: sorted particle indices
                          uint* cellStart,
                          uint* cellEnd,
                          uint    numParticles)
{
    uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (index >= numParticles) return;

    // read particle data from sorted arrays
    float3 pos = make_float3(oldPos[index]);
    float3 vel = make_float3(oldVel[index]);

    // get address in grid
    int3 gridPos = calcGridPos(pos);

    // examine neighbouring cells
    float3 force = make_float3(0.0f);

    for (int z = -1; z <= 1; z++)
    {
        for (int y = -1; y <= 1; y++)
        {
            for (int x = -1; x <= 1; x++)
            {
                int3 neighbourPos = gridPos + make_int3(x, y, z);
                force += collideWithCellHash(neighbourPos, index, pos, vel, oldPos, oldVel, cellStart, cellEnd);
            }
        }
    }

    // collide with cursor sphere
    // force += collideSpheres(pos, params.colliderPos, vel, make_float3(0.0f, 0.0f, 0.0f), params.particleRadius, params.colliderRadius, 0.0f);

    // write new velocity back to original unsorted location
    uint originalIndex = gridParticleIndex[index];
    newVel[originalIndex] = make_float4(vel + force, 0.0f);
}


#endif