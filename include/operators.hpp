#ifndef OPERATORS_HPP
#define OPERATORS_HPP

#include <iostream>

// tokenizer
class Tokenizer {
    public:
        Tokenizer(const std::string& text);
        ~Tokenizer();

        // string token to int token id
        void encode();

        // int token id to string token
        void decode();
};

// model
class Model {

};


//

void embeddng();

void rmsnorm(int batch_size, int seq_len, int vec_size, float* input, float* output,);

void matmul();

void matmul_silu();



#endif