#include <glad/glad.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_gl_interop.h>

#include <stdio.h>

extern "C"
{
    __global__ void points3dKernel(float3* pos, unsigned int width, unsigned int height, float time)
    {
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

        // calculate uv coordinates
        float u = x / (float)width;
        float v = y / (float)height;
        u = u * 2.0f - 1.0f;
        v = v * 2.0f - 1.0f;
        u = u * 7.0f;
        v = v * 7.0f;

        // calculate simple sine wave pattern
        float freq = 2.0f;
        float w = sinf(u * freq + time) * cosf(v * freq + time) * 2.0f;

        // write output vertex
        pos[y * width + x] = make_float3(u, v, w);
    }

    void runKernel(float3* pos, unsigned int mesh_width, unsigned int mesh_height, float time)
    {
        // execute the kernel
        dim3 block(16, 16, 1);
        dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
        points3dKernel << < grid, block >> > (pos, mesh_width, mesh_height, time);
    }

    void runCuda(struct cudaGraphicsResource** vbo_resource, float time, unsigned int width, unsigned int height)
    {
        // map OpenGL buffer object for writing from CUDA
        float3* dptr;
        cudaGraphicsMapResources(1, vbo_resource, 0);
        size_t num_bytes;
        cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes,
            *vbo_resource);
        //printf("CUDA mapped VBO: May access %ld bytes\n", num_bytes);

        runKernel(dptr, width, height, time);

        // unmap buffer object
        cudaGraphicsUnmapResources(1, vbo_resource, 0);
    }


    void runTest(unsigned int width, unsigned int height)
    {
        void* returnData = malloc(width * height * sizeof(float));
        void* d_vbo_buffer = NULL;
        // create VBO
        cudaMalloc((void**)&d_vbo_buffer, width * height * 3 * sizeof(float));

        // execute the kernel
        runKernel((float3*)d_vbo_buffer, width, height, 1.0f);

        cudaDeviceSynchronize();
        cudaMemcpy(returnData, d_vbo_buffer, width * height * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_vbo_buffer);
        d_vbo_buffer = NULL;

        free(returnData);
        printf("Test passed");

    }

    void registerWithCuda(struct cudaGraphicsResource** resource, GLuint vbo) {
        cudaGraphicsGLRegisterBuffer(resource, vbo, cudaGraphicsMapFlagsWriteDiscard);
    }

    void unRegisterWithCuda(cudaGraphicsResource_t vbo_res) {
        cudaGraphicsUnregisterResource(vbo_res);

    }
}