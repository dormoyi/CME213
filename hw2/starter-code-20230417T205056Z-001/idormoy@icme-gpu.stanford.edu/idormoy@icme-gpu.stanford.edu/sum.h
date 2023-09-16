#include <vector>

// import algorithm to be able to do the reduction?
// https://stackoverflow.com/questions/43168661/openmp-and-reduction-on-stdvector



std::vector<uint> serialSum(const std::vector<uint> &v) {
    std::vector<uint> sums(2);
    // sums[0]  = evem
    // sums[1] = odd
    sums[0]=0;
    sums[1]=0;
    for (uint i = 0; i < v.size(); i++) {
        if (v[i] % 2 == 0)
            sums[0] += v[i];
        else
            sums[1] += v[i];
    }
    return sums;
}

std::vector<uint> parallelSum(const std::vector<uint> &v) {
    std::vector<uint> sums(2);
    uint sum_even = 0;
    uint sum_odd = 0;
    #pragma omp parallel for reduction(+:sum_even, sum_odd)
    for (uint i = 0; i < v.size(); i++) {
        if (v[i] % 2 == 0)
            sum_even += v[i];
        else
            sum_odd += v[i];
    }
    sums[0] = sum_even;
    sums[1] = sum_odd;
    return sums;
}