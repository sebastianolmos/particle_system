#include "kernelParams.cuh"

extern "C"
{
    void allocateArray(void** devPtr, size_t size);

    void freeArray(void* devPtr);

    void threadSync();

    void copyArrayToDevice(void* device, const void* host, int offset, int size);

    void registerGLBufferObject(uint vbo, struct cudaGraphicsResource** cuda_vbo_resource);

    void unregisterGLBufferObject(struct cudaGraphicsResource* cuda_vbo_resource);

    void* mapGLBufferObject(struct cudaGraphicsResource** cuda_vbo_resource);

    void unmapGLBufferObject(struct cudaGraphicsResource* cuda_vbo_resource);

    void copyArrayFromDevice(void* host, const void* device, struct cudaGraphicsResource** cuda_vbo_resource, int size);

    void setParameters(kernelParams* hostParams);

    // compute grid and thread block size for a given number of elements
    void computeGridSize(uint n, uint blockSize, uint& numBlocks, uint& numThreads);

    void integrateInDevice1(float* pos, float* vel, float deltaTime, uint nParticles);

    void integrateInDevice2(float* pos, float* vel, float deltaTime, uint nParticles);

    void swapVelInDevice(float* newVel, float* oldVel, uint nParticles);

    void updateGridInDevice(float* pos, uint* gridCounters, uint* gridCells, uint nParticles, uint numCells);

    void collideInDevice(float* newVel, float* oldPos, float* oldVel, uint* gridCounters, uint* gridCells, uint nParticles);

    void calcHashInDevice(uint* gridParticleHash, uint* gridParticleIndex, float* pos, int nParticles);

    // sort particles based on hash
    void sortParticlesInDevice(uint* dGridParticleHash, uint* dGridParticleIndex, uint nParticles);

    // reorder particle arrays into sorted order and find start and end of each cell
    void reorderParticlesInDevice(uint* cellStart, uint* cellEnd, float* sortedPos, float* sortedVel, uint* gridParticleHash, uint* gridParticleIndex,
        float* oldPos, float* oldVel, uint numParticles, uint numCells);

    // process collisions with hash
    void collideWithHashInDevice(float* newVel, float* sortedPos, float* sortedVel, uint* gridParticleIndex, uint* cellStart, uint* cellEnd,
        uint   numParticles, uint   numCells);
}