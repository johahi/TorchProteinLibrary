#include "angles2backbone_interface.h"
#include <cBackboneProteinCUDAKernels.h>
#include <iostream>

int Angles2Backbone_forward(at::Tensor input_angles, 
                            at::Tensor output_coords, 
                            at::Tensor angles_length, 
                            at::Tensor A
                        ){
    if( (!input_angles.type().is_cuda()) || (!output_coords.type().is_cuda()) || (!angles_length.type().is_cuda()) 
    || (!A.type().is_cuda()) ){
        throw("Incorrect device");
    }
    if(input_angles.ndimension() != 3){
        throw("Incorrect input ndim");
    }
    
    cpu_computeCoordinatesBackbone(	input_angles.data<float>(), 
                                    output_coords.data<float>(), 
                                    A.data<float>(), 
                                    angles_length.data<int>(),
                                    input_angles.size(0),
                                    input_angles.size(2));
}
int Angles2Backbone_backward(   at::Tensor gradInput,
                                at::Tensor gradOutput,
                                at::Tensor input_angles, 
                                at::Tensor angles_length, 
                                at::Tensor A,   
                                at::Tensor dr_dangle
                            ){
    
    if( (!gradInput.type().is_cuda()) || (!gradOutput.type().is_cuda()) || (!input_angles.type().is_cuda()) 
    || (!A.type().is_cuda()) || (!angles_length.type().is_cuda()) || (!dr_dangle.type().is_cuda())){
        throw("Incorrect device");
    }
    if(gradOutput.ndimension() != 2){
        throw("Incorrect input ndim");
    }
    cpu_computeDerivativesBackbone( input_angles.data<float>(),
                                    dr_dangle.data<float>(),
                                    A.data<float>(),
                                    angles_length.data<int>(),
                                    input_angles.size(0),
                                    input_angles.size(2));
    
    cpu_backwardFromCoordsBackbone( gradInput.data<float>(),
                                    gradOutput.data<float>(),
                                    dr_dangle.data<float>(),
                                    angles_length.data<int>(),
                                    input_angles.size(0),
                                    input_angles.size(2));
}