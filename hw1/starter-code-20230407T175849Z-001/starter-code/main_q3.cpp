#include <iostream>
#include <memory>
#include <string>
#include <vector>

/* TODO: Make Matrix a pure abstract class with the
 * public method:
 *     std::string repr()
 */
class Matrix {
 public:
  virtual ~Matrix() = 0; //special type of method that is called when an object of this class is destroyed. 
  virtual std::string repr() = 0; //  QUES why destruction thing
};

Matrix::~Matrix() {}

// The empty implementation of the destructor in this line {}
// provides a default implementation that does nothing

/* TODO: Modify the following classes so that the code runs as expected */

class SparseMatrix : public Matrix{
 public:
  std::string repr() {
    return "sparse";
  }
};

class ToeplitzMatrix : public Matrix{
 public:
  std::string repr() { return "toeplitz"; }
};

/* TODO: This function should accept a vector of Matrices and call the repr
 * function on each matrix, printing the result to the standard output.
 */

// shared pointers provide a mechanism for runtime polymorphism, allowing a single function to work with objects of different types
void PrintRepr(const std::vector<std::shared_ptr<Matrix>> &vec) {
  for (auto matrix : vec) {
    std::cout << matrix->repr() << std::endl;
  }
}

// -> necessary because m is a shared pointer, not a raw pointer to 
// the Matrix object, and so we must use the arrow operator to access the object's methods

/* This fills a vector with an instance of SparseMatrix
 * and an instance of ToeplitzMatrix and passes the resulting vector
 * to the PrintRepr function.
 */
int main() {
  std::vector<std::shared_ptr<Matrix>> vec;
  vec.push_back(std::make_shared<SparseMatrix>());
  vec.push_back(std::make_shared<ToeplitzMatrix>());
  PrintRepr(vec);
}
