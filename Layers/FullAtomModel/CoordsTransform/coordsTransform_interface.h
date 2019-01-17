#include <torch/extension.h>
void CoordsTranslate_forward(   at::Tensor input_coords, 
                                at::Tensor output_coords,
                                at::Tensor T,
                                at::Tensor num_atoms
                                );

void CoordsRotate_forward(  at::Tensor input_coords, 
                            at::Tensor output_coords,
                            at::Tensor R,
                            at::Tensor num_atoms
                            );

void CoordsRotate_backward( at::Tensor grad_output_coords, 
                            at::Tensor grad_input_coords,
                            at::Tensor R,
                            at::Tensor num_atoms);

void getBBox(   at::Tensor input_coords,
                at::Tensor a, at::Tensor b,
                at::Tensor num_atoms);

void getRandomRotation( at::Tensor R);
void getRandomTranslation( at::Tensor T, at::Tensor a, at::Tensor b, int volume_size);
