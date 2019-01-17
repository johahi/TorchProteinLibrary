#include "volume2xplor_interface.h"
#include <stdio.h>
#include <string>

void Volume2Xplor(  at::Tensor volume, const char *filename){
    if(volume.ndimension() != 3){
        std::cout<<"Incorrect input ndim"<<std::endl;
        throw("Incorrect input ndim");
    }
    auto V = volume.accessor<float, 3>();
    int size = volume.size(0);
    float mean=0.5, std=0.5;
    
    FILE *fout = fopen(filename, "w");
    fprintf(fout, "\n");
    fprintf(fout, " Density map\n");
    fprintf(fout, " 1\n");
    fprintf(fout, " 4\n");
    fprintf(fout, "%8d%8d%8d%8d%8d%8d%8d%8d%8d\n",size-1,0,size-1,size-1,0,size-1,size-1,0,size-1);
    fprintf(fout, "%12.5E%12.5E%12.5E%12.5E%12.5E%12.5E\n",float(size),float(size),float(size),90.,90.,90.);
    fprintf(fout, "ZYX\n");
    for(int z=0; z<size; z++){
        fprintf(fout, "%8d\n", z);
        for(int y=0; y<size; y++){
            for(int x=0; x<size; x+=6){
                fprintf(fout, "%12.5E%12.5E%12.5E%12.5E%12.5E%12.5E\n", V[x][y][z], V[x+1][y][z], V[x+2][y][z], V[x+3][y][z], V[x+4][y][z], V[x+5][y][z]);
            }
        }
    }
    fprintf(fout, "%8d\n", -9999);
    fprintf(fout, "%12.5E%12.5E\n", mean, std);
    fclose(fout);

}