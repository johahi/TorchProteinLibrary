#include <torch/extension.h>
void PDB2CoordsOrdered(at::Tensor filenames, at::Tensor coords, at::Tensor res_names, at::Tensor atom_names);
void PDB2CoordsUnordered(at::Tensor filenames, at::Tensor coords, at::Tensor res_names, at::Tensor atom_names, at::Tensor num_atoms);
