#include <math_constants.h>

#include "BC.h"

/**
 * Calculates the next finite difference step given a
 * grid point and step lengths.
 *
 * @param curr Pointer to the grid point that should be updated.
 * @param width Number of grid points in the x dimension.
 * @param xcfl Courant number for x dimension.
 * @param ycfl Courant number for y dimension.
 * @returns Grid value of next timestep.
 */
template<int order>
__device__
float Stencil(const float* curr, int width, float xcfl, float ycfl) {
    switch(order) {
        case 2:
            return curr[0] + xcfl * (curr[-1] + curr[1] - 2.f * curr[0]) +
                   ycfl * (curr[width] + curr[-width] - 2.f * curr[0]);

        case 4:
            return curr[0] + xcfl * (-curr[2] + 16.f * curr[1] - 30.f * curr[0]
                                     + 16.f * curr[-1] - curr[-2])
                           + ycfl * (- curr[2 * width] + 16.f * curr[width]
                                     - 30.f * curr[0] + 16.f * curr[-width]
                                     - curr[-2 * width]);

        case 8:
            return curr[0] + xcfl * (-9.f * curr[4] + 128.f * curr[3]
                                     - 1008.f * curr[2] + 8064.f * curr[1]
                                     - 14350.f * curr[0] + 8064.f * curr[-1]
                                     - 1008.f * curr[-2] + 128.f * curr[-3]
                                     - 9.f * curr[-4])
                           + ycfl * (-9.f * curr[4 * width]
                                     + 128.f * curr[3 * width]
                                     - 1008.f * curr[2 * width]
                                     + 8064.f * curr[width]
                                     - 14350.f * curr[0]
                                     + 8064.f * curr[-width]
                                     - 1008.f * curr[-2 * width]
                                     + 128.f * curr[-3 * width]
                                     - 9.f * curr[-4 * width]);

        default:
            printf("ERROR: Order %d not supported", order);
            return CUDART_NAN_F;
    }
}

/**
 * Kernel to propagate finite difference grid from the current
 * time point to the next.
 *
 * This kernel should be very simple and only use global memory
 * and 1d threads and blocks.
 *
 * @param next[out] Next grid state.
 * @param curr Current grid state.
 * @param gx Number of grid points in the x dimension.
 * @param nx Number of grid points in the x dimension to which the full
 *           stencil can be applied (ie the number of points that are at least
 *           order/2 grid points away from the boundary).
 * @param ny Number of grid points in the y dimension to which th full
 *           stencil can be applied.
 * @param xcfl Courant number for x dimension.
 * @param ycfl Courant number for y dimension.
 */
template<int order>
__global__
void gpuStencilGlobal(float* next, const float* __restrict__ curr, int gx, int nx, int ny,
                float xcfl, float ycfl) {
    // TODO
    // Assign a thread-block to a row of the mesh


    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int borderSize = order / 2;

    int x = borderSize + id % nx;
    int y = borderSize + id / nx;

    // apply stencil
    // if you do go over the limit you will need to map one thread to multiple executions via a for loop (not necessary here)
    //  for (i = blockIdx.x * blockDim.x + threadIdx.x; i<array_length; i+= gridDim.x * blockDim.x){ 
    if (id < nx*ny)
    {
    next[x + gx * y] = Stencil<order>(&curr[x + gx * y], gx, xcfl, ycfl);
    }
}

/**
 * Propagates the finite difference 2D heat diffusion solver
 * using the gpuStencilGlobal kernel.
 *
 * Use this function to do necessary setup and propagate params.iters()
 * number of times.
 *
 * @param curr_grid The current state of the grid.
 * @param params Parameters for the finite difference computation.
 * @returns Time required for computation.
 */
double gpuComputationGlobal(Grid& curr_grid, const simParams& params) {

    boundary_conditions BC(params);

    Grid next_grid(curr_grid);

    // TODO: Declare variables/Compute parameters.
    // variables
    float xcfl = params.xcfl();
    float ycfl = params.ycfl();

    int nx = params.nx();
    int ny = params.ny();

    int gx = params.gx();
    int borderSize = params.borderSize();

    int order = params.order();

    // compute parameters
    const int block_size = 512; // because this is the max, but 512 would also work
    int num_nodes = nx*ny;
    int blocks_per_grid = (num_nodes + block_size - 1) / block_size; //  formula for when num_nodes not divisible by block_size

    // synchronizing the threads  - this is probably done by the compiler, but we should do it after each iteration otherwise

    // copy the data
    //curr_grid.toGPU();
    //next_grid.toGPU();

    event_pair timer;
    start_timer(&timer);

    for(int i = 0; i < params.iters(); ++i) {
        // update the values on the boundary only
        BC.updateBC(next_grid.dGrid_, curr_grid.dGrid_);

        // TODO: Apply stencil.

        // thread divergence within a warp, mask all the other threads -> never put an if inside a kernel
        // even if it is if(true), all the threads will have to set the mask and time is lost
        if (order==2)
            gpuStencilGlobal<2><<<blocks_per_grid,block_size>>>(next_grid.dGrid_, curr_grid.dGrid_, gx, nx, ny, xcfl, ycfl);
        if (order==4)
            gpuStencilGlobal<4><<<blocks_per_grid,block_size>>>(next_grid.dGrid_, curr_grid.dGrid_, gx, nx, ny, xcfl, ycfl);
        if (order==8)
            gpuStencilGlobal<8><<<blocks_per_grid,block_size>>>(next_grid.dGrid_, curr_grid.dGrid_, gx, nx, ny, xcfl, ycfl);

        Grid::swap(curr_grid, next_grid);
    }

    check_launch("gpuStencilGlobal");

    //curr_grid.fromGPU();
    // curr_grid.saveStateToFile("out1.csv");

    return stop_timer(&timer);
}


/**
 * Kernel to propagate finite difference grid from the current
 * time point to the next.
 *
 * This kernel should be optimized to compute finite difference updates
 * in blocks of size (blockDim.y * numYPerStep) * blockDim.x. Each thread
 * should calculate at most numYPerStep updates. It should still only use
 * global memory.
 *
 * @param next[out] Next grid state.
 * @param curr Current grid state.
 * @param gx Number of grid points in the x dimension.
 * @param nx Number of grid points in the x dimension to which the full
 *           stencil can be applied (ie the number of points that are at least
 *           order/2 grid points away from the boundary).
 * @param ny Number of grid points in the y dimension to which th full
 *           stencil can be applied.
 * @param xcfl Courant number for x dimension.
 * @param ycfl Courant number for y dimension.
 */
template<int order, int numYPerStep>
__global__
void gpuStencilBlock(float* next, const float* __restrict__ curr, int gx, int nx, int ny,
                    float xcfl, float ycfl) {
    // TODO
    // A thread block should work on a mesh block.

    int borderSize = order / 2;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = ( blockIdx.y * blockDim.y + threadIdx.y )*numYPerStep;

    int iters;
    iters = min(numYPerStep, ny-j);
    int y;
    int x = i + borderSize;  
    if (i < nx) {
      // we do less than numYPerStep if grid % numYPerStep != 0
        for (int k = 0; k < iters; k++) { // see the 1
            y = j + borderSize + k;
            next[x + gx * y] = Stencil<order>(&curr[x + gx * y], gx, xcfl, ycfl);
        }
    }
}

/**
 * Propagates the finite difference 2D heat diffusion solver
 * using the gpuStencilBlock kernel.
 *
 * Use this function to do necessary setup and propagate params.iters()
 * number of times.
 *
 * @param curr_grid The current state of the grid.
 * @param params Parameters for the finite difference computation.
 * @returns Time required for computation.
 */
double gpuComputationBlock(Grid& curr_grid, const simParams& params) {

    boundary_conditions BC(params);

    Grid next_grid(curr_grid);

    // TODO: Declare variables/Compute parameters.
    // variables
    float xcfl = params.xcfl();
    float ycfl = params.ycfl();

    int nx = params.nx();
    int ny = params.ny();

    int gx = params.gx();
    int borderSize = params.borderSize();

    int order = params.order();

    // compute parameters
    const int numYPerStep = 4;
    int xthread = 64;
    int ythread = xthread/numYPerStep; 
    dim3 threads(xthread, ythread);

    int xblock = (nx + threads.x - 1)/threads.x; 
    int yblock = (ny + threads.y * numYPerStep- 1)/threads.y / numYPerStep; 
    dim3 blocks(xblock, yblock);

    event_pair timer;
    start_timer(&timer);

    for(int i = 0; i < params.iters(); ++i) {
        // update the values on the boundary only
        BC.updateBC(next_grid.dGrid_, curr_grid.dGrid_);

        // TODO: Apply stencil.
        // <<number of blocks, number of threads>>
        if (order==2)
            gpuStencilBlock<2, numYPerStep><<<blocks,threads>>>(next_grid.dGrid_, curr_grid.dGrid_, gx, nx, ny, xcfl, ycfl);
        else if (order==4)
            gpuStencilBlock<4, numYPerStep><<<blocks,threads>>>(next_grid.dGrid_, curr_grid.dGrid_, gx, nx, ny, xcfl, ycfl);
        else if (order==8)
            gpuStencilBlock<8, numYPerStep><<<blocks,threads>>>(next_grid.dGrid_, curr_grid.dGrid_, gx, nx, ny, xcfl, ycfl);


        Grid::swap(curr_grid, next_grid);
    }

    check_launch("gpuStencilBlock");

    //curr_grid.saveStateToFile("out2.csv");
    return stop_timer(&timer);
}


/**
 * Kernel to propagate finite difference grid from the current
 * time point to the next.
 *
 * This kernel should be optimized to compute finite difference updates
 * in blocks of size side * side using shared memory.
 *
 * @param next[out] Next grid state.
 * @param curr Current grid state.
 * @param gx Number of grid points in the x dimension.
 * @param gy Number of grid points in the y dimension.
 * @param xcfl Courant number for x dimension.
 * @param ycfl Courant number for y dimension.
 */
template<int side, int order>
__global__
void gpuStencilShared(float* next, const float* __restrict__ curr, int gx, int gy,
               float xcfl, float ycfl) {
    // TODO

    // Since shared memory is local to a block, it cannot be allocated in the main
    // function. Instead, it needs to be declared inside the CUDA kernel.
    int borderSize = order / 2;
    int numYPerStep = side / blockDim.y;

    int small_side = side - order; 
    int side_border = side - borderSize;
    int i = blockIdx.x * small_side + threadIdx.x;
    int j =  blockIdx.y * small_side + threadIdx.y  *numYPerStep;

    //int nx = gx - borderSize; /// the actual nx is gx - order
    //int ny = gy - borderSize;

    const int lane = threadIdx.x;

    __shared__ float block[side][side]; 

    // Load block into shared memory
    // each thread is going to load a coalesced part of the memory
    int iters;
    if (i < gx){
    iters = min(numYPerStep, gy-j);
    for (int k = 0; k < iters; k++){
        block[threadIdx.y*numYPerStep+k][lane] = curr[i + gx * (j+k)];

    }
    }
    __syncthreads();

    //  update next from the shared memory
    iters = min(numYPerStep, gy -j); 
    if (threadIdx.x >= borderSize && threadIdx.x < side_border && i < gx - borderSize && j < gy - borderSize) {
        for (int k = 0; k < iters; k++) { 
            if (k+threadIdx.y*numYPerStep >= borderSize && threadIdx.y*numYPerStep+k < side_border && j+k < gy - borderSize) {
            // printf("next[x + gx * y]\n" );
            // printf('%a \n', *next[x + gx * y]);
            // printf("block[threadIdx.y*numYPerStep+k][lane] \n");
            // printf('%a \n', *block[threadIdx.y*numYPerStep+k][lane]);
            // printf("stencil computation\n");
            // printf('%a \n', Stencil<order>(&block[threadIdx.y*numYPerStep+k][lane], side, xcfl, ycfl));
            next[i + gx * (j+k)] = Stencil<order>(&block[threadIdx.y*numYPerStep+k][lane], side, xcfl, ycfl);
            }
        }
    }
}

/**
 * Propagates the finite difference 2D heat diffusion solver
 * using the gpuStencilShared kernel.
 *
 * Use this function to do necessary setup and propagate params.iters()
 * number of times.
 *
 * @param curr_grid The current state of the grid.
 * @param params Parameters for the finite difference computation.
 * @returns Time required for computation.
 */
template<int order>
double gpuComputationShared(Grid& curr_grid, const simParams& params) {

    boundary_conditions BC(params);

    Grid next_grid(curr_grid);

    // TODO: Declare variables/Compute parameters.
    // variables
    float xcfl = params.xcfl();
    float ycfl = params.ycfl();

    int nx = params.nx();
    int ny = params.ny();

    int gx = params.gx();
    int gy = params.gy();
    int borderSize = params.borderSize();

    // compute parameters
    const int numYPerStep = 8;
    int xthread = 64;
    const int side = 64; // xthread
    int ythread = xthread/numYPerStep;
    dim3 threads(xthread, ythread);

    int small_side = xthread - order;
    int xblock = (gx + small_side - 1)/small_side; 
    int yblock = (gy + small_side  - 1)/small_side; 
    dim3 blocks(xblock, yblock);

    event_pair timer;
    start_timer(&timer);

    for(int i = 0; i < params.iters(); ++i) {
        // update the values on the boundary only
        BC.updateBC(next_grid.dGrid_, curr_grid.dGrid_);

        // TODO: Apply stencil.
        // <<number of blocks, number of threads>>
        if (order==2)
            gpuStencilShared<side, 2><<<blocks,threads>>>(next_grid.dGrid_, curr_grid.dGrid_, gx, gy, xcfl, ycfl);
        if (order==4)
            gpuStencilShared<side, 4><<<blocks,threads>>>(next_grid.dGrid_, curr_grid.dGrid_, gx, gy, xcfl, ycfl);
        if (order==8)
            gpuStencilShared<side, 8><<<blocks,threads>>>(next_grid.dGrid_, curr_grid.dGrid_, gx, gy, xcfl, ycfl);

        Grid::swap(curr_grid, next_grid);
    }

    check_launch("gpuStencilShared");
    return stop_timer(&timer);
}
