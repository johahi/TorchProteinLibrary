#include "volumeConvolution_interface.h"
#include <VolumeConv.h>
#include <iostream>


void VolumeConvolution_forward( at::Tensor volume1, 
                                at::Tensor volume2, 
                                at::Tensor output){
    if( (!volume1.type().is_cuda()) || (!volume2.type().is_cuda()) || (!output.type().is_cuda()) ){
        throw("Incorrect device");
    }
    if(volume1.ndimension()!=4){
        throw("incorrect input dimension");
    }
    cpu_VolumeConv(	volume1.data<float>(), 
                    volume2.data<float>(), 
                    output.data<float>(), 
                    volume1.size(0),
                    volume1.size(1));
}
void VolumeConvolution_backward(    at::Tensor gradOutput,
                                    at::Tensor gradVolume1,
                                    at::Tensor gradVolume2,
                                    at::Tensor volume1, 
                                    at::Tensor volume2){
    if( (!gradOutput.type().is_cuda()) || (!gradVolume1.type().is_cuda()) || (!gradVolume2.type().is_cuda())
        || (!volume1.type().is_cuda()) || (!volume2.type().is_cuda()) ){
        throw("Incorrect device");
    }
    if(gradOutput.ndimension()!=4){
        throw("incorrect input dimension");
    }
        
    cpu_VolumeConv(	gradOutput.data<float>(), 
                    volume2.data<float>(), 
                    gradVolume1.data<float>(), 
                        volume1.size(0),
                        volume1.size(1));
    
    cpu_VolumeConv(	gradOutput.data<float>(), 
                    volume1.data<float>(), 
                    gradVolume2.data<float>(), 
                    volume1.size(0),
                    volume1.size(1));
    
}


