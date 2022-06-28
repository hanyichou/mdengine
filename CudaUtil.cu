#include "CudaUtil.cuh"

/*#if __CUDA_ARCH__ < 300
volatile extern __shared__ int sh[];
__device__ int warp_bcast(int v, int leader) {
	// WARNING: might not be safe to call in divergent branches 
	const int tid = threadIdx.x;
	const int warpLane = tid % WARPSIZE;
	if (warpLane == leader)
		sh[tid/WARPSIZE] = v;
	return sh[tid/WARPSIZE];		
}	
#elif __CUDA_ARCH__ < 700
__device__ int warp_bcast(int v, int leader) {return __shfl(v, leader); }
#else*/
//__device__ int warp_bcast(int v, int leader, int srcLane) {return __shfl_sync(v, leader, srcLane); }
__inline__ __device__ unsigned __lanemask_lt()
{
    unsigned mask;
    asm volatile ( "mov.u32 %0, %lanemask_lt;" : "=r"( mask ) );
    return mask;
}
/*
__forceinline__ __device__ unsigned lane_id()
{
    unsigned ret; 
    asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
    return ret;
}
*/
//#endif


/*#if __CUDA_ARCH__ < 700
__device__ int atomicAggInc(int *ctr, int warpLane) {
	// https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/
	int mask = __ballot(1);
	int leader = __ffs(mask)-1;

	int res;
	if ( warpLane == leader )
		res = atomicAdd(ctr, __popc(mask));
	res = warp_bcast(res,leader);
	return res + __popc( mask & ((1 << warpLane) - 1) );
}
#else*/
#if 0
__device__ int atomicAggInc(int *ptr) 
{
    int pred;
    int mask = __match_all_sync(__activemask(), ptr, &pred);
    int leader = __ffs(mask) - 1;    // select a leader
    int res;
    if(lane_id() == leader)                  // leader does the update
        res = atomicAdd(ptr, __popc(mask));
    res = __shfl_sync(mask, res, leader);    // get leader’s old value
    return res + __popc(mask & ((1 << lane_id()) - 1)); //compute old value
}
#endif

__device__ int atomicAggInc(int *ctr) 
{
    unsigned int active = __activemask();
    int leader = __ffs(active) - 1;
    int change = __popc(active);
    unsigned int rank = __popc(active & __lanemask_lt());
    int warp_res;
    if(rank == 0)
        warp_res = atomicAdd(ctr, change);
    warp_res = __shfl_sync(active, warp_res, leader);
    return warp_res + rank;
}
//#endif
/*
__global__
void reduceVector(const int num, Vector3* __restrict__ vector, Vector3* netVector) {
	extern __shared__ Vector3 blockVector[];
	const int tid = threadIdx.x;

	// grid-stride loop over vector[]
	for (int i = threadIdx.x+blockIdx.x*blockDim.x; i < num; i+=blockDim.x*gridDim.x) {
		// assign vector to shared memory
		blockVector[tid] = vector[i];
		// blockVector[tid] = Vector3(0.0f);
		__syncthreads();
		
		
		// Reduce vectors in shared memory
		// http://www.cuvilib.com/Reduction.pdf
		for (int offset = blockDim.x/2; offset > 0; offset >>= 1) {
			if (tid < offset) {
				int oid = tid + offset;
				blockVector[tid] = blockVector[tid] + blockVector[oid];
			}
			__syncthreads();
		}

		if (tid == 0)
			atomicAdd( netVector, blockVector[0] );
	}
}*/
