#ifndef _MATRIX_HPP
#define _MATRIX_HPP

#include <ostream>
#include <vector>

/*
This is the pure abstract base class specifying general set of functions for a
square matrix.

Concrete classes for specific types of matrices, like MatrixSymmetric, should
implement these functions.
*/
template <typename T>
class Matrix {
  // Returns reference to matrix element (i, j).
  virtual T& operator()(int i, int j) = 0; //  virtual: it can be overridden by a derived class.
  // The = 0 at the end of the function declaration indicates that this is a pure virtual function
  // Number of non-zero elements in matrix.
  virtual unsigned NormL0() const = 0;
  // Enables printing all matrix elements using the overloaded << operator
  virtual void Print(std::ostream& ostream) = 0;

  template <typename U>
  friend std::ostream& operator<<(std::ostream& stream, Matrix<U>& m);
  // friend: it has access to the private and protected members of the Matrix class
};

/* TODO: Overload the insertion operator by modifying the ostream object */
template <typename T>
std::ostream& operator<<(std::ostream& stream, Matrix<T>& m) {
  m.Print(stream);
  return stream;
}

/* MatrixDiagonal Class is a subclass of the Matrix class */
template <typename T>
class MatrixDiagonal : public Matrix<T> {
 private:
  // Matrix Dimension. Equals the number of columns and the number of rows.
  unsigned int n_;

  // Elements of the matrix. You get to choose how to organize the matrix
  // elements into this vector.
  std::vector<T> data_;
  T zero = 0; //  this element stores the zero of the matrix

 public:
  // TODO: Default constructor
  MatrixDiagonal() : n_(0) {
    n_ = 0;
    // allocating memory for the empty vector
    data_.resize(0);
  }

  // TODO: Constructor that takes matrix dimension as argument
  MatrixDiagonal(const int n) : n_(n) {
    // throw an error if the argument is negative
    if (n < 0) {
      throw std::invalid_argument("Matrix dimension must be non-negative");
    }
    n_ = n;
    // allocating memory for the vector
    data_.resize(n_);
  }

  // TODO: Function that returns the matrix dimension
  //The "const" keyword at the end of the function declaration 
  //indicates that this function is a "const" member function, 
  //which means it can be called on a const object of the class
  unsigned int size() const { 
    return n_;
  }

  // TODO: Function that returns reference to matrix element (i, j).
  // The "override" keyword at the end of the function declaration indicates 
  // that this function overrides a virtual function declared in a base class.
  T& operator()(int i, int j) override { 
    if (i<0 || i>=(int)n_ || j>=(int)n_ || j<0)
      throw std::invalid_argument("Index out of bounds");
    if (i == j) {
      return data_[i]; // QUES reference
    } else {
      zero = 0; //  we set to zero before returning to make sure no one modifies it
      return zero; //  QUES why don't we return *zero?
    }
  }

  // TODO: Function that returns number of non-zero elements in matrix.
  unsigned NormL0() const override { 
    unsigned int count = 0;
    for (unsigned int i =0; i<n_; i++) {
      if (data_[i]!= 0) {
        count++;
      }
    }
    return count;
  }

  // TODO: Function that modifies the ostream object so that
  // the "<<" operator can print the matrix (one row on each line).
  void Print(std::ostream& ostream) override {
    for (unsigned int i=0; i<n_; i++) {
      for (unsigned int j=0; j<n_; j++) {
        if (i==j) {
          ostream << data_[i] << " ";
        } else {
          ostream << 0 << " ";
        } 
      } ostream << std::endl;
    }
  }
};

/* MatrixSymmetric Class is a subclass of the Matrix class */
template <typename T>
class MatrixSymmetric : public Matrix<T> {
 private:
  // Matrix Dimension. Equals the number of columns and the number of rows.
  unsigned int n_;
  // Elements of the matrix. You get to choose how to organize the matrix
  // elements into this vector.
  std::vector<T> data_;

 public:
  // TODO: Default constructor
  MatrixSymmetric() {
    n_ = 0;
    // allocating memory for the empty vector
    data_.resize(0);
  }

  // TODO: Constructor that takes matrix dimension as argument
  MatrixSymmetric(const int n) {
    // throw an error if the argument is negative
    if (n < 0) {
      throw std::invalid_argument("Matrix dimension must be non-negative");
    }
    n_ = n;
    // allocating memory for the vector
    data_.resize(n_*(n_+1)/2);
  }

  // TODO: Function that returns the matrix dimension
  unsigned int size() const { return n_; }

  //  0 1 2 3
  //    4 5 6
  //      7 8
  //        9
  // i*n + j  when squared

  // [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

  // TODO: Function that returns reference to matrix element (i, j).
  T& operator()(int i, int j) override {
    if (i<0 || i>=(int)n_ || j>=(int)n_ || j<0)
      throw std::invalid_argument("Index out of bounds");
    if (i <= j)
        return data_[i*n_-(i-1)*i/2+j-i];
      else
        return data_[j*n_-(j-1)*j/2+i-j];
  
  }

  // TODO: Function that returns number of non-zero elements in matrix.
  unsigned NormL0() const override {
    unsigned int count = 0;
    for (unsigned int j =0; j<n_; j++) {
      for (unsigned int i=0; i<=j; i++) {
        if (i==j && data_[i*n_-(i-1)*i/2+j-i]!=0){
          count++;
        } else if ( i<j && data_[i*n_-(i-1)*i/2+j-i]!=0) { // other formula test
          count = count + 2;}
      }
      }

    return count;
  }

  // TODO: Function that modifies the ostream object so that
  // the "<<" operator can print the matrix (one row on each line).
  void Print(std::ostream& ostream) override {
    for (unsigned int i=0; i<n_; i++){
      for (unsigned int j=0; j<n_; j++){
        if (i <= j)
        ostream <<  data_[i*n_-(i-1)*i/2+j-i] << " ";
      else
        ostream <<  data_[j*n_-(j-1)*j/2+i-j] << " ";
      } ostream << std::endl;
    }
  }
};

#endif /* MATRIX_HPP */