#ifndef MODEL_HPP
#define MODEL_HPP

#include <string>

class Model {
  public:
    Model(const std::string &model_path, const std::string &tokenizer_path);
    ~Model();

    
};

#endif // MODEL_HPP