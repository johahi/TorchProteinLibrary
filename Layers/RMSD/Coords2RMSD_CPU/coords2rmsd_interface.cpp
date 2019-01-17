#include "coords2rmsd_interface.h"
#include <cRMSD.h>
#include <iostream>
#include <string>


void Coords2RMSD_CPU_forward(   at::Tensor src, at::Tensor dst, at::Tensor rmsd,
                            at::Tensor ce_src, at::Tensor ce_dst,
                            at::Tensor U_ce_src, at::Tensor UT_ce_dst,
                            at::Tensor num_atoms
                        ){
    
    
    int batch_size = src.size(0);
    auto num_atoms_acc = num_atoms.accessor<int,1>();
    // #pragma omp parallel for num_threads(10)
    for(int i=0; i<batch_size; i++){
        at::Tensor single_src = src[i];
        at::Tensor single_dst = dst[i];
        at::Tensor single_ce_src = ce_src[i];
        at::Tensor single_ce_dst = ce_dst[i];
        at::Tensor single_U_ce_src = U_ce_src[i];
        at::Tensor single_UT_ce_dst = UT_ce_dst[i];

        cRMSD crmsd( single_ce_src.data<double>(), single_ce_dst.data<double>(), 
                    single_U_ce_src.data<double>(), single_UT_ce_dst.data<double>(), 
                    num_atoms_acc[i]);
        rmsd[i] = crmsd.compute(single_src.data<double>(), single_dst.data<double>());
    }
}
void Coords2RMSD_CPU_backward(  at::Tensor grad_atoms, at::Tensor grad_output,
                            at::Tensor ce_src, at::Tensor ce_dst,
                            at::Tensor U_ce_src, at::Tensor UT_ce_dst,
                            at::Tensor num_atoms
                        ){
    
    if(grad_atoms.ndimension() != 2){
        std::cout<<"Incorrect input ndim"<<std::endl;
        throw("Incorrect input ndim");
    }
    int batch_size = grad_atoms.size(0);
    auto grad_output_acc = grad_output.accessor<double, 1>();
    auto num_atoms_acc = num_atoms.accessor<int,1>();

    // #pragma omp parallel for num_threads(10)
    for(int i=0; i<batch_size; i++){
        at::Tensor single_grad_atoms = grad_atoms[i];
        at::Tensor single_ce_src = ce_src[i];
        at::Tensor single_ce_dst = ce_dst[i];
        at::Tensor single_U_ce_src = U_ce_src[i];
        at::Tensor single_UT_ce_dst = UT_ce_dst[i];
        
        double single_grad_output = grad_output_acc[i];
        
        cRMSD rmsd( single_ce_src.data<double>(), single_ce_dst.data<double>(), 
                    single_U_ce_src.data<double>(), single_UT_ce_dst.data<double>(), 
                    num_atoms_acc[i]);

        rmsd.grad(single_grad_atoms.data<double>(), &single_grad_output);
    }
}
