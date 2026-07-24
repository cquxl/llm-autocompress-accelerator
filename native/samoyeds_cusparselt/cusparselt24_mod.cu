// Samoyeds cuSPARSELt 2:4 PyTorch binding.
// Source: local Samoyeds codex/kernel integration, Apache-2.0.
// This binding calls the NVIDIA cuSPARSELt C API directly; it does not use
// torch._cslt_sparse_mm or torch.sparse semi-structured operators.

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cuda_fp16.h>
#include <cusparseLt.h>

#include <memory>
#include <sstream>
#include <unordered_map>

namespace py = pybind11;

namespace {

__global__ void permute_activation_kernel(
    const __half* __restrict__ input,
    const int32_t* __restrict__ permutation,
    __half* __restrict__ output,
    int64_t tokens,
    int64_t channels) {
    const int64_t pair = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t token = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const int64_t channel0 = pair * 2;
    if (token >= tokens || channel0 >= channels) {
        return;
    }

    const int64_t row_offset = token * channels;
    const int32_t source0 = __ldg(permutation + channel0);
    const __half value0 = __ldg(input + row_offset + source0);
    const int64_t channel1 = channel0 + 1;
    if (channel1 < channels) {
        const int32_t source1 = __ldg(permutation + channel1);
        const __half value1 = __ldg(input + row_offset + source1);
        *reinterpret_cast<__half2*>(output + row_offset + channel0) =
            __halves2half2(value0, value1);
    } else {
        output[row_offset + channel0] = value0;
    }
}

torch::Tensor permute_activation_out(
    torch::Tensor input,
    torch::Tensor permutation,
    torch::Tensor output) {
    TORCH_CHECK(input.is_cuda(), "activation must be on CUDA");
    TORCH_CHECK(input.scalar_type() == torch::kFloat16, "activation must be FP16");
    TORCH_CHECK(input.dim() == 2 && input.is_contiguous(),
                "activation must be a contiguous [tokens, channels] tensor");
    TORCH_CHECK(permutation.is_cuda(), "permutation must be on CUDA");
    TORCH_CHECK(permutation.scalar_type() == torch::kInt32,
                "permutation must have dtype torch.int32");
    TORCH_CHECK(permutation.dim() == 1 && permutation.is_contiguous(),
                "permutation must be a contiguous vector");
    TORCH_CHECK(permutation.size(0) == input.size(1),
                "permutation length must equal the activation channel count");
    TORCH_CHECK(input.size(1) % 2 == 0,
                "half2 activation permutation requires an even channel count");
    TORCH_CHECK(output.is_cuda() && output.scalar_type() == torch::kFloat16,
                "permutation output must be a CUDA FP16 tensor");
    TORCH_CHECK(output.sizes() == input.sizes() && output.is_contiguous(),
                "permutation output must be contiguous and match the input shape");
    TORCH_CHECK(output.device() == input.device(),
                "permutation input and output must be on the same CUDA device");

    const int64_t tokens = input.size(0);
    const int64_t channels = input.size(1);
    const dim3 block(256, 2);
    const dim3 grid(
        static_cast<unsigned int>((channels + 2 * block.x - 1) / (2 * block.x)),
        static_cast<unsigned int>((tokens + block.y - 1) / block.y));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    permute_activation_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half*>(input.data_ptr()),
        permutation.data_ptr<int32_t>(),
        reinterpret_cast<__half*>(output.data_ptr()),
        tokens,
        channels);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "activation permutation kernel launch failed");
    return output;
}

torch::Tensor permute_activation(torch::Tensor input, torch::Tensor permutation) {
    return permute_activation_out(input, permutation, torch::empty_like(input));
}

void check_cusparselt(cusparseStatus_t status, const char* expression) {
    if (status == CUSPARSE_STATUS_SUCCESS) {
        return;
    }
    std::ostringstream message;
    message << expression << " failed with cuSPARSELt status " << static_cast<int>(status);
    throw std::runtime_error(message.str());
}

#define CHECK_CUSPARSELT(expr) check_cusparselt((expr), #expr)

struct MatmulPlan {
    cusparseLtMatDescriptor_t mat_a{};
    cusparseLtMatDescriptor_t mat_b{};
    cusparseLtMatDescriptor_t mat_c{};
    cusparseLtMatmulDescriptor_t matmul{};
    cusparseLtMatmulAlgSelection_t algorithm{};
    cusparseLtMatmulPlan_t plan{};
    torch::Tensor compressed_weight;
    torch::Tensor workspace;
    bool mat_a_initialized = false;
    bool mat_b_initialized = false;
    bool mat_c_initialized = false;
    bool plan_initialized = false;

    ~MatmulPlan() {
        if (mat_a_initialized) cusparseLtMatDescriptorDestroy(&mat_a);
        if (mat_b_initialized) cusparseLtMatDescriptorDestroy(&mat_b);
        if (mat_c_initialized) cusparseLtMatDescriptorDestroy(&mat_c);
        if (plan_initialized) cusparseLtMatmulPlanDestroy(&plan);
    }
};

class CuSparseLtLinear {
public:
    explicit CuSparseLtLinear(torch::Tensor weight)
        : weight_(weight.contiguous()), out_features_(weight.size(0)), in_features_(weight.size(1)) {
        TORCH_CHECK(weight_.is_cuda(), "cuSPARSELt weight must be on CUDA");
        TORCH_CHECK(weight_.scalar_type() == torch::kFloat16, "cuSPARSELt supports FP16 weights only");
        TORCH_CHECK(weight_.dim() == 2, "cuSPARSELt weight must be 2D");
        TORCH_CHECK(in_features_ % 16 == 0 && out_features_ % 16 == 0,
                    "cuSPARSELt dimensions must be multiples of 16");
        CHECK_CUSPARSELT(cusparseLtInit(&handle_));
    }

    ~CuSparseLtLinear() {
        plans_.clear();
        cusparseLtDestroy(&handle_);
    }

    torch::Tensor forward(torch::Tensor input) {
        TORCH_CHECK(input.is_cuda(), "cuSPARSELt input must be on CUDA");
        TORCH_CHECK(input.scalar_type() == torch::kFloat16, "cuSPARSELt supports FP16 inputs only");
        TORCH_CHECK(input.dim() == 2 && input.size(1) == in_features_,
                    "cuSPARSELt input must have shape [tokens, in_features]");
        TORCH_CHECK(input.is_contiguous(), "cuSPARSELt input must be contiguous");
        TORCH_CHECK(input.size(0) % 16 == 0,
                    "cuSPARSELt token dimension must be padded to a multiple of 16");

        const int64_t tokens = input.size(0);
        auto iterator = plans_.find(tokens);
        if (iterator == plans_.end()) {
            iterator = plans_.emplace(tokens, create_plan(tokens, input)).first;
        }
        auto& cached = *iterator->second;
        // [tokens, out_features] row-major has the same byte layout as the
        // cuSPARSELt column-major C[out_features, tokens].
        auto output = torch::empty({tokens, out_features_}, input.options());
        float alpha = 1.0f;
        float beta = 0.0f;
        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
        cudaStream_t streams[] = {stream};
        CHECK_CUSPARSELT(cusparseLtMatmul(
            &handle_, &cached.plan, &alpha, cached.compressed_weight.data_ptr(), input.data_ptr(),
            &beta, output.data_ptr(), output.data_ptr(),
            cached.workspace.numel() ? cached.workspace.data_ptr() : nullptr, streams, 1));
        return output;
    }

private:
    std::shared_ptr<MatmulPlan> create_plan(int64_t tokens, const torch::Tensor& search_input) {
        auto cached = std::make_shared<MatmulPlan>();
        constexpr uint32_t alignment = 16;
        CHECK_CUSPARSELT(cusparseLtStructuredDescriptorInit(
            &handle_, &cached->mat_a, out_features_, in_features_, in_features_, alignment,
            CUDA_R_16F, CUSPARSE_ORDER_ROW, CUSPARSELT_SPARSITY_50_PERCENT));
        cached->mat_a_initialized = true;
        CHECK_CUSPARSELT(cusparseLtDenseDescriptorInit(
            &handle_, &cached->mat_b, in_features_, tokens, in_features_, alignment,
            CUDA_R_16F, CUSPARSE_ORDER_COL));
        cached->mat_b_initialized = true;
        CHECK_CUSPARSELT(cusparseLtDenseDescriptorInit(
            &handle_, &cached->mat_c, out_features_, tokens, out_features_, alignment,
            CUDA_R_16F, CUSPARSE_ORDER_COL));
        cached->mat_c_initialized = true;
        // cuSPARSELt 0.7+ rejects CUSPARSE_COMPUTE_16F for this FP16
        // sparse-dense matmul path on L40/SM89. Older 0.4.x builds used by
        // the original benchmark only expose/use CUSPARSE_COMPUTE_16F.
#if CUSPARSELT_VERSION >= 521
        constexpr auto compute_type = CUSPARSE_COMPUTE_32F;
#else
        constexpr auto compute_type = CUSPARSE_COMPUTE_16F;
#endif
        CHECK_CUSPARSELT(cusparseLtMatmulDescriptorInit(
            &handle_, &cached->matmul, CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE, &cached->mat_a, &cached->mat_b,
            &cached->mat_c, &cached->mat_c, compute_type));
        CHECK_CUSPARSELT(cusparseLtMatmulAlgSelectionInit(
            &handle_, &cached->algorithm, &cached->matmul, CUSPARSELT_MATMUL_ALG_DEFAULT));
        CHECK_CUSPARSELT(cusparseLtMatmulPlanInit(
            &handle_, &cached->plan, &cached->matmul, &cached->algorithm));
        cached->plan_initialized = true;

        size_t compressed_size = 0;
        size_t compression_buffer_size = 0;
        CHECK_CUSPARSELT(cusparseLtSpMMACompressedSize(
            &handle_, &cached->plan, &compressed_size, &compression_buffer_size));
        auto byte_options = torch::TensorOptions().dtype(torch::kUInt8).device(weight_.device());
        cached->compressed_weight = torch::empty({static_cast<int64_t>(compressed_size)}, byte_options);
        auto compression_buffer = torch::empty(
            {static_cast<int64_t>(compression_buffer_size)}, byte_options);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
        CHECK_CUSPARSELT(cusparseLtSpMMACompress(
            &handle_, &cached->plan, weight_.data_ptr(), cached->compressed_weight.data_ptr(),
            compression_buffer_size ? compression_buffer.data_ptr() : nullptr, stream));

        auto search_output = torch::zeros({tokens, out_features_}, search_input.options());
        float alpha = 1.0f;
        float beta = 0.0f;
        cudaStream_t streams[] = {stream};
        CHECK_CUSPARSELT(cusparseLtMatmulSearch(
            &handle_, &cached->plan, &alpha, cached->compressed_weight.data_ptr(),
            search_input.data_ptr(), &beta, search_output.data_ptr(), search_output.data_ptr(),
            nullptr, streams, 1));

        size_t workspace_size = 0;
        CHECK_CUSPARSELT(cusparseLtMatmulGetWorkspace(&handle_, &cached->plan, &workspace_size));
        cached->workspace = torch::empty({static_cast<int64_t>(workspace_size)}, byte_options);
        return cached;
    }

    torch::Tensor weight_;
    int64_t out_features_;
    int64_t in_features_;
    cusparseLtHandle_t handle_{};
    std::unordered_map<int64_t, std::shared_ptr<MatmulPlan>> plans_;
};

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    py::class_<CuSparseLtLinear>(module, "CuSparseLtLinear")
        .def(py::init<torch::Tensor>())
        .def("forward", &CuSparseLtLinear::forward);
    module.def("permute_activation", &permute_activation,
               "Permute the channels of a contiguous FP16 activation");
    module.def("permute_activation_out", &permute_activation_out,
               "Permute FP16 activation channels into a reusable output tensor");
}
