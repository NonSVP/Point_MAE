#include <vector>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// Updated macros to use TORCH_CHECK (replacement for AT_ASSERTM)
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_TYPE(x, t) TORCH_CHECK(x.dtype() == t, #x " must be " #t)
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be on CUDA")
#define CHECK_INPUT(x, t) CHECK_CONTIGUOUS(x); CHECK_TYPE(x, t); CHECK_CUDA(x)

// The kernel wrapper uses long*, which we link to int64_t on Windows
void knn_device(
    float* ref_dev, 
    int ref_nb, 
    float* query_dev, 
    int query_nb, 
    int dim, 
    int k, 
    float* dist_dev, 
    int64_t* ind_dev, 
    cudaStream_t stream
    );

std::vector<at::Tensor> knn(
    at::Tensor & ref, 
    at::Tensor & query, 
    const int k
    ){

    CHECK_INPUT(ref, at::kFloat);
    CHECK_INPUT(query, at::kFloat);
    
    int dim = ref.size(0);
    int ref_nb = ref.size(1);
    int query_nb = query.size(1);
    
    // Updated .data<float>() to .data_ptr<float>()
    float * ref_dev = ref.data_ptr<float>();
    float * query_dev = query.data_ptr<float>();
    
    auto dist = at::empty({ref_nb, query_nb}, query.options().dtype(at::kFloat));
    auto ind = at::empty({k, query_nb}, query.options().dtype(at::kLong));
    
    float * dist_dev = dist.data_ptr<float>();
    // Updated long to int64_t to match the linker's expectation for 64-bit integers
    int64_t * ind_dev = ind.data_ptr<int64_t>();
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    knn_device(
        ref_dev,
        ref_nb,
        query_dev,
        query_nb,
        dim,
        k,
        dist_dev,
        ind_dev,
        stream
    );

    return {dist.slice(0, 0, k), ind};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("knn", &knn, "KNN cuda version");
}