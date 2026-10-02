// Run the production compact-buffer constructor without CUDA driver interposition.
#include "mercury_cuda_memory.h"
#include <cassert>
#include <cstring>
int main() {
    for(std::size_t n:{1,4,33,257,4097}) {
        auto zero=mercury_zero_force_parameters(n);
        std::vector<double> uploaded(5*n);
        std::memcpy(uploaded.data(),zero.data(),5*n*sizeof(double));
        assert(zero.size()==5*n);
        for(double coefficient:uploaded) assert(coefficient==0);
    }
}
