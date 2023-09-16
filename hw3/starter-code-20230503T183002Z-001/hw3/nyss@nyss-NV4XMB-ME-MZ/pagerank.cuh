#ifndef _PAGERANK_CUH
#define _PAGERANK_CUH

#include "util.cuh"

/* 
 * Each kernel handles the update of one pagerank score. In other
 * words, each kernel handles one row of the update:
 *
 *      pi(t+1) = A pi(t) + (1 / (2N))
 *
 */
__global__ void device_graph_propagate( // only called on one thread
    const uint *graph_indices,
    const uint *graph_edges,
    const float *graph_nodes_in,
    float *graph_nodes_out,
    const float *inv_edges_per_node,
    int num_nodes
) {
    // TODO: fill in the kernel code here
    // A num_node x num_node , pi num_node x 1
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j;
    float tot=0.f;
    if (i<num_nodes){
        //Specifically, node i is adjacent to all nodes in the range h_graph_edges[h_graph_indices[i] ... h_graph_indices[i+1]].
        for (j = graph_indices[i]; j < graph_indices[i+1]; j++)
            tot+= graph_nodes_in[graph_edges[j]] * inv_edges_per_node[graph_edges[j]];
    tot = 0.5f * tot; // formula
    tot += 0.5f / num_nodes; // + (1 / (2N))
    graph_nodes_out[i] = tot;
    }
}

/* 
 * This function executes a specified number of iterations of the
 * pagerank algorithm. The variables are:
 *
 * h_graph_indices, h_graph_edges:
 *     These arrays describe the indices of the neighbors of node i.
 *     Specifically, node i is adjacent to all nodes in the range
 *     h_graph_edges[h_graph_indices[i] ... h_graph_indices[i+1]].
 *
 * h_node_values_input:
 *     An initial guess of pi(0).
 *
 * h_gpu_node_values_output:
 *     Output array for the pagerank vector.
 *
 * h_inv_edges_per_node:
 *     The i'th element in this array is the reciprocal of the
 *     out degree of the i'th node.
 *
 * nr_iterations:
 *     The number of iterations to run the pagerank algorithm for.
 *
 * num_nodes:
 *     The number of nodes in the whole graph (ie N).
 *
 * avg_edges:
 *     The average number of edges in the graph. You are guaranteed
 *     that the whole graph has num_nodes * avg_edges edges.
 */
double device_graph_iterate(
    const uint *h_graph_indices,
    const uint *h_graph_edges,
    const float *h_node_values_input,
    float *h_gpu_node_values_output,
    const float *h_inv_edges_per_node,
    int nr_iterations,
    int num_nodes,
    int avg_edges
) {
    // TODO: allocate GPU memory

    // pointers
    uint *d_graph_indices = nullptr;
    uint *d_graph_edges = nullptr;
    float *d_node_values_input = nullptr;
    float *d_gpu_node_values_output = nullptr;
    float *d_inv_edges_per_node = nullptr;

    //malloc
    int size_graph_indices = sizeof(int) * (num_nodes+1) ; // common way of storing graphes, we go h_graph_edges[h_graph_indices[i] ... h_graph_indices[i+1]]
    int size_graph_edges = sizeof(int) * avg_edges * num_nodes; // so there is a +1
    int size_node_values_input = sizeof(float) * num_nodes;
    int size_gpu_node_values_output = sizeof(float) * num_nodes;
    int size_inv_edges_per_node = sizeof(float) * num_nodes;

    cudaMalloc(&d_graph_indices, size_graph_indices);  
    cudaMalloc(&d_graph_edges, size_graph_edges); 
    cudaMalloc(&d_node_values_input, size_node_values_input); 
    cudaMalloc(&d_gpu_node_values_output, size_gpu_node_values_output); 
    cudaMalloc(&d_inv_edges_per_node, size_inv_edges_per_node); 

    // TODO: check for allocation failure
    if (!d_graph_indices || !d_graph_edges ||
    !d_node_values_input || !d_gpu_node_values_output ||
    !d_inv_edges_per_node) {
    std::cerr << "Couldn't allocate memory!" << std::endl; 
    return 1;
    }

    // TODO: copy data to the GPU
    // dst, src, count, kind
    cudaMemcpy(d_graph_indices, &h_graph_indices[0], size_graph_indices,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_graph_edges, &h_graph_edges[0], size_graph_edges,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_node_values_input, &h_node_values_input[0], size_node_values_input,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_gpu_node_values_output, &h_gpu_node_values_output[0], size_gpu_node_values_output,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_inv_edges_per_node, &h_inv_edges_per_node[0], size_inv_edges_per_node,
               cudaMemcpyHostToDevice);

    // launch kernels
    event_pair timer;
    start_timer(&timer);

    const int block_size = 192;
    int blocks_per_grid = (num_nodes + block_size - 1) / block_size; //  formula for when num_nodes not divisible by block_size

    // TODO: launch your kernels the appropriate number of iterations
    // each kernel operates on one row
    for (int iter = 0; iter<nr_iterations; iter++){  // number of blocks, threads per blocks
        device_graph_propagate<<<blocks_per_grid,block_size>>>(d_graph_indices,
        d_graph_edges,d_node_values_input,d_gpu_node_values_output,d_inv_edges_per_node,num_nodes);

        std::swap(d_node_values_input, d_gpu_node_values_output); // we swap every time
        // works because all the threads are synchronized, that way we don't modify input of other threads (inplace modification)
    }

    if (nr_iterations % 2 != 0){
        std::swap(d_node_values_input, d_gpu_node_values_output);
    }

    check_launch("gpu graph propagate");
    double gpu_elapsed_time = stop_timer(&timer);

    // TODO: copy final data back to the host for correctness checking
    cudaMemcpy(&h_gpu_node_values_output[0], d_node_values_input,
	       size_gpu_node_values_output, cudaMemcpyDeviceToHost);

    // TODO: free the memory you allocated!
    cudaFree(d_graph_indices);
    cudaFree(d_graph_edges);
    cudaFree(d_node_values_input);
    cudaFree(d_gpu_node_values_output);
    cudaFree(d_inv_edges_per_node);

    return gpu_elapsed_time;
}

/**
 * This function computes the number of bytes read from and written to
 * global memory by the pagerank algorithm.
 * 
 * nodes:
 *      The number of nodes in the graph
 *
 * edges: 
 *      The average number of edges in the graph
 *
 * iterations:
 *      The number of iterations the pagerank algorithm was run
 */
uint get_total_bytes(uint nodes, uint edges, uint iterations)
{
    // TODO
    // we ignore cache and consider that all the tables are stored in main memory
    // the 4 tables we read in
    int read = iterations * (sizeof(int) * edges * nodes + sizeof(float) * nodes + sizeof(float) * nodes + 2 * sizeof(int) * (nodes+1));
    // the table we write in
    int written = iterations * sizeof(float) * nodes;
    return read + written;
}

#endif
