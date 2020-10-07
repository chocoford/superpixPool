#include <torch/torch.h>
#include <vector>

// CUDA forward declarations

std::vector<at::Tensor> suppixpool_max_cuda_forward(
    at::Tensor img,
    at::Tensor spx_labels,
    at::Tensor output, 
    at::Tensor outIdx,
    const int K);

std::vector<at::Tensor> suppixpool_max_cuda_backward(
    at::Tensor grad_outputs,
    at::Tensor img,
    at::Tensor spx_labels,
    at::Tensor max_indices,
    const int K);

std::vector<at::Tensor> suppixpool_avg_cuda_forward(
    at::Tensor img,
    at::Tensor spx_labels,
    at::Tensor output,
    at::Tensor pool_size,
    const int K);

at::Tensor get_adjacent_matrix_cuda(
    at::Tensor spx_labels,
    at::Tensor output,
    const int K);

// C++ interface

#define CHECK_CUDA(x) AT_ASSERT(x.type().is_cuda()) //, #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERT(x.is_contiguous()) //, #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> suppixpool_max_forward(
    at::Tensor img,
    at::Tensor spx_labels,
    const int K) {
  
  CHECK_INPUT(img);
  CHECK_INPUT(spx_labels);
  // img + img; // breaks code

  const int batch_size = img.size(0);
  const int channels_size = img.size(1);

  at::Tensor output = torch::zeros({batch_size, channels_size, K}, torch::CUDA(at::kInt));
  output = output.type_as(img);
  // torch::set_requires_grad(output, true);
  // 初始化为了-1
  at::Tensor outIdx = -torch::ones({batch_size, channels_size, K}, torch::CUDA(at::kInt)); 
  return suppixpool_max_cuda_forward(img, spx_labels, output, outIdx, K);
  // return {output, outIdx};
  // return {img, spx_labels};
}

std::vector<at::Tensor> suppixpool_max_backward(
    at::Tensor grad_outputs,
    at::Tensor img,
    at::Tensor spx_labels,
    at::Tensor max_indices,
    const int K) {
  
  CHECK_INPUT(grad_outputs);
  CHECK_INPUT(img);
  CHECK_INPUT(spx_labels);
  CHECK_INPUT(max_indices);

  // at::Tensor output =  torch::CUDA(at::kFloat).zeros({batch_size, channels_size, K});
  // at::Tensor outIdx = -torch::CUDA(at::kInt).ones({batch_size, channels_size, K});

  return suppixpool_max_cuda_backward(
      grad_outputs,
      img,
      spx_labels,
      max_indices,
      K);
}

std::vector<at::Tensor> suppixpool_avg_forward(
    at::Tensor img,
    at::Tensor spx_labels,
    const int K) {

  CHECK_INPUT(img);
  CHECK_INPUT(spx_labels);
  // img + img; // breaks code

  const int batch_size = img.size(0);
  const int channels_size = img.size(1);

  at::Tensor output = torch::zeros({batch_size, channels_size, K}, torch::CUDA(at::kInt));
  output = output.type_as(img);
  // torch::set_requires_grad(output, true);
  // at::Tensor outIdx = -at::ones(torch::CUDA(at::kInt), {batch_size, channels_size, K}); // save max index of each superpixel
  // aveNum: save the size of each superpixel
  at::Tensor pool_size = torch::zeros({batch_size, channels_size, K}, torch::CUDA(at::kInt));
  // std::cout << aveNum;
  suppixpool_avg_cuda_forward(img, spx_labels, output, pool_size, K);
  // std::cout<<aveNum;
  return {output, pool_size};
  // return {img, spx_labels};
}

// K: superpixels num
at::Tensor get_adjacent_matrix(
    at::Tensor spx_labels,
    const int K) {

  CHECK_INPUT(spx_labels);

  const int batch_size = spx_labels.size(0);

  at::Tensor output = torch::zeros({batch_size, K, K}, torch::CUDA(at::kInt));
  output = output.type_as(spx_labels);
  
  get_adjacent_matrix_cuda(spx_labels, output, K);
  // std::cout<<aveNum;
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("max_forward", &suppixpool_max_forward, "Superpixel max pooling forward (CUDA)");
  m.def("max_backward", &suppixpool_max_backward, "Superpixel max pooling backward (CUDA)");
  m.def("avg_forward", &suppixpool_avg_forward, "Superpixel avg pooling forward (CUDA)");
  m.def("get_adjacent_matrix", &get_adjacent_matrix, "Superpixel get adjacent matrix (CUDA)");
}