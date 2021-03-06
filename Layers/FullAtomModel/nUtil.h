#ifndef NUTIL_H_
#define NUTIL_H_
#include <string>
#include <algorithm>
#include <memory>
#include <ATen/ATen.h>
#include <cVector3.h>
#include <cMatrix33.h>
#include <cConformation.h>

namespace StringUtil{
    //string utils
    std::string trim(const std::string &s);

    // template<typename ... Args>
    // inline std::string string_format( const std::string& format, Args ... args ){
    //     size_t size = snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
    //     std::unique_ptr<char[]> buf( new char[ size ] ); 
    //     std::snprintf( buf.get(), size, format.c_str(), args ... );
    //     return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
    // };
    std::string string_format(const std::string fmt, ...);
    at::Tensor string2Tensor(std::string s);
    void string2Tensor(std::string s, at::Tensor T);
    std::string tensor2String(at::Tensor T);
};

namespace ProtUtil{
    // atom indexing common for cConformation and cPDBLoader
    uint getAtomIndex(std::string &res_name, std::string &atom_name);
    
    // number of atoms in a sequence
    uint getNumAtoms(std::string &sequence, bool add_terminal);
    
    // heavy atoms
    bool isHeavyAtom(std::string &atom_name);

    // convert 1-letter aa code to 3-letter code
    std::string convertRes1to3(char resName);

    // assign atom type from 11 possible
    uint get11AtomType(std::string res_name, std::string atom_name, bool terminal);

    void rotate(at::Tensor &input_coords, cMatrix33 &R, at::Tensor &output_coords, int num_atoms);
    // void rotate(THDoubleTensor *coords, cMatrix33 R);
    void translate(at::Tensor &input_coords, cVector3 &T, at::Tensor &output_coords, int num_atoms);
    // void translate(THDoubleTensor *coords, cVector3 T);
    void computeBoundingBox(at::Tensor &input_coords, int num_atoms, cVector3 &b0, cVector3 &b1);

    cMatrix33 getRandomRotation();
    cVector3 getRandomTranslation(uint spatial_dim, cVector3 &b0, cVector3 &b1);
    
    cMatrix33 tensor2Matrix33(at::Tensor T);
    void matrix2Tensor(cMatrix33 &mat, at::Tensor &T);

};

#endif