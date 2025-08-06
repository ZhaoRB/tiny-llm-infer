#ifndef OPERATORS_HPP
#define OPERATORS_HPP

#include <string>
#include <vector>

void token_encode(const std::string &text, std::vector<int> &token_ids);
void token_decode(const std::vector<int> &token_ids, std::string &text);

void embedding(int len, int token_id, float *output, float *weight);

void rmsnorm(int len, float *input, float *output, float *weight);

// Q K V O projection
// prefill
void matmul(int M, int N, int K, const float *A, const float *B, float *C);

void gemv(int M, int N, const float *A, const float *B, float *C);

// position embedding
void rope(int len, float *input, float *output, int pos);


// attention
// prefill 


#endif // OPERATORS_HPP