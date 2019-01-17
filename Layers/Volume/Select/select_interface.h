#include <torch/extension.h>

void SelectVolume_forward(  at::Tensor volume,
                            at::Tensor coords,
                            at::Tensor num_atoms,
                            at::Tensor features,
                            float res
                        );