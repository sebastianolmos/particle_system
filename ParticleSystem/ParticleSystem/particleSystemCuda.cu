#include <glad/glad.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_gl_interop.h>

#include <stdio.h>

#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"

#include "kernelSystem.cuh"

extern "C"
{
    void allocateArray(void** devPtr, size_t size)
    {
        cudaMalloc(devPtr, size);
    }

    void freeArray(void* devPtr)
    {
        cudaFree(devPtr);
    }

    void threadSync()
    {
        cudaDeviceSynchronize();
    }

    void copyArrayToDevice(void* device, const void* host, int offset, int size)
    {
        cudaMemcpy((char*)device + offset, host, size, cudaMemcpyHostToDevice);
    }

    void registerGLBufferObject(uint vbo, struct cudaGraphicsResource** cuda_vbo_resource)
    {
        cudaGraphicsGLRegisterBuffer(cuda_vbo_resource, vbo, cudaGraphicsMapFlagsNone);
    }

    void unregisterGLBufferObject(struct cudaGraphicsResource* cuda_vbo_resource)
    {
        cudaGraphicsUnregisterResource(cuda_vbo_resource);
    }

    void* mapGLBufferObject(struct cudaGraphicsResource** cuda_vbo_resource)
    {
        void* ptr;
        cudaGraphicsMapResources(1, cuda_vbo_resource, 0);
        size_t num_bytes;
        cudaGraphicsResourceGetMappedPointer((void**)&ptr, &num_bytes, *cuda_vbo_resource);
        return ptr;
    }

    void unmapGLBufferObject(struct cudaGraphicsResource* cuda_vbo_resource)
    {
        cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);
    }

    void copyArrayFromDevice(void* host, const void* device,
        struct cudaGraphicsResource** cuda_vbo_resource, int size)
    {
        if (cuda_vbo_resource)
        {
            device = mapGLBufferObject(cuda_vbo_resource);
        }

        cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost);

        if (cuda_vbo_resource)
        {
            unmapGLBufferObject(*cuda_vbo_resource);
        }
    }

    void setParameters(kernelParams* hostParams)
    {
        // copy parameters to constant memory
        cudaMemcpyToSymbol(params, hostParams, sizeof(kernelParams));
    }

    //Round a / b to nearest higher integer value
    uint iDivUp(uint a, uint b)
    {
        return (a % b != 0) ? (a / b + 1) : (a / b);
    }

    // compute grid and thread block size for a given number of elements
    void computeGridSize(uint n, uint blockSize, uint& numBlocks, uint& numThreads)
    {
        numThreads = min(blockSize, n);
        numBlocks = iDivUp(n, numThreads);
    }

    void integrateInDevice1(float* pos, float* vel, float deltaTime, uint nParticles)
    {
        thrust::device_ptr<float4> d_pos((float4*)pos);
        thrust::device_ptr<float4> d_vel((float4*)vel);

        thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(d_pos, d_vel)),
            thrust::make_zip_iterator(thrust::make_tuple(d_pos + nParticles, d_vel + nParticles)),
            integrate1(deltaTime));
    }

    void integrateInDevice2(float* pos, float* vel, float deltaTime, uint nParticles)
    {
        // thread per particle
        uint numThreads, numBlocks;
        computeGridSize(nParticles, 256, numBlocks, numThreads);

        // execute the kernel
        integrate2 << < numBlocks, numThreads >> > ((float4*)pos,
            (float4*)vel,
            deltaTime);
    }

    void swapVelInDevice(float* newVel, float* oldVel, uint nParticles)
    {
        // thread per particle
        uint numThreads, numBlocks;
        computeGridSize(nParticles, 256, numBlocks, numThreads);

        // execute the kernel
        swapVel << < numBlocks, numThreads >> > (
            (float4*)newVel,
            (float4*)oldVel,
            nParticles);
    }

    void updateGridInDevice(float* pos, uint* gridCounters, uint* gridCells, uint nParticles, uint numCells)
    {
        uint numThreads, numBlocks;
        computeGridSize(nParticles, 256, numBlocks, numThreads);

        // set all cells to empty
        cudaMemset(gridCounters, 0x00000000, numCells * sizeof(uint));
        uint smemSize = sizeof(uint) * (numThreads + 1);
        updateGrid << < numBlocks, numThreads, smemSize >> >(
            (float4*)pos,
            gridCounters,
            gridCells,
            nParticles
            );
    }

    void collideInDevice(float* newVel, float* oldPos, float* oldVel, uint* gridCounters, uint* gridCells, uint nParticles)
    {
        // thread per particle
        uint numThreads, numBlocks;
        computeGridSize(nParticles, 256, numBlocks, numThreads);

        // execute the kernel
        collideParticles << < numBlocks, numThreads >> > (
            (float4*)newVel,
            (float4*)oldPos,
            (float4*)oldVel,
            gridCounters,
            gridCells,
            nParticles);
    }
}