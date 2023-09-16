#ifndef MATRIX_RECT
#define MATRIX_RECT

#include <algorithm>
#include <iomanip>
#include <numeric>
#include <ostream>
#include <vector>

template <typename T>
class Matrix2D;

template <typename T>
bool Broadcastable(Matrix2D<T>& A, Matrix2D<T>& B) {
  // TODO: Write a function that returns true if either of the matrices can be
  // broadcast to be compatible with the other for elementwise multiplication.

  // Two dimensions are compatible when
  //   they are equal, or
  //   one of them is 1.
  // We need to test every dimension

  if ((A.size_rows() == B.size_rows() || A.size_rows()==1 || B.size_rows()==1) && 
  (A.size_cols() == B.size_cols()|| A.size_cols()==1 || B.size_cols()==1)) 
    return true;
 else {
    return false;
  }
}

template <typename T>
class Matrix2D {
 private:
  // The size of the matrix is (n_rows, n_cols)
  unsigned int n_rows;
  unsigned int n_cols;

  // Vector storing the data in row major order. Element (i,j) for 0 <= i <
  // n_rows and 0 <= j < n_cols is stored at data[i * n_cols + j].
  std::vector<T> data_;

 public:
  // Empty matrix
  Matrix2D() { 
    // TODO
    n_rows = 0;
    n_cols = 0;
    data_.resize(0);
  }

  // Constructor takes argument (m,n) = matrix dimension.
  Matrix2D(const int m, const int n) {
      if (n < 0 || m < 0) {
      throw std::invalid_argument("Matrix dimension must be non-negative");
    }
      n_rows = m;
      n_cols = n;
      data_.resize(m*n);
  }

  unsigned int size_rows() const { return n_rows; } // TODO
  unsigned int size_cols() const { return n_cols; } // TODO

  // Returns reference to matrix element (i, j).
  T& operator()(int i, int j) {
    // TODO: Hint: Element (i,j) for 0 <= i < n_rows and 0 <= j < n_cols 
    // is stored at data[i * n_cols + j]. 
    if (i<0 || i>=(int)n_rows || j>=(int)n_cols || j<0)
      throw std::invalid_argument("Index out of bounds");
    return data_[i * n_cols + j];
  }
    
  void Print(std::ostream& ostream) {
      // TODO
      for (unsigned int i=0;i<n_rows; i++) {
        for (unsigned int j=0; j<n_cols; j++) {
          ostream << (*this)(i, j)<< " ";
        } ostream << std::endl;}
  }

  Matrix2D<T> dot(Matrix2D<T>& mat) {
    if (n_rows == mat.size_rows() && n_cols == mat.size_cols()) {
      Matrix2D<T> ret(n_rows, n_cols);
      for (unsigned int i = 0; i < n_rows; i++) {
        for (unsigned int j = 0; j < n_cols; j++) {
          ret(i, j) = (*this)(i, j) * mat(i, j);
        }
      }
      return ret;
    } else if (Broadcastable<T>(*this, mat)) {
      // TODO: Replace the code in this scope.
      // Compute and return the elementwise product of the two Matrix2D's
      // "*this" and "mat" after appropriate broadcasting.
      Matrix2D<T> ret;
      // there are 6 cases
      // 1. left is 1x1
      if (n_rows == 1 && n_cols == 1) {
        ret = Matrix2D<T>(mat.size_rows(), mat.size_cols());
        for (unsigned int i = 0; i< mat.size_rows(); i++)
          for (unsigned int j = 0; j < mat.size_cols(); j++)
            ret(i, j) = (*this)(0, 0) * mat(i, j);
      }
      // 2. right is 1x1
      else if (mat.size_rows() == 1 && mat.size_cols() == 1) {
        ret = Matrix2D<T>(n_rows, n_cols);
        for (unsigned int i = 0; i< n_rows; i++)
          for (unsigned int j = 0; j < n_cols; j++)
            ret(i, j) = (*this)(i, j) * mat(0, 0);
      }
      // 3. left is 1xn
      else if (n_rows == 1) {
        ret = Matrix2D<T>(mat.size_rows(), mat.size_cols());
        for (unsigned int i = 0; i< mat.size_rows(); i++)
          for (unsigned int j = 0; j < mat.size_cols(); j++)
            ret(i, j) = (*this)(0, j) * mat(i, j);
      }

      // 4. right is 1xn
      else if (mat.size_rows() == 1) {
        ret = Matrix2D<T>(n_rows, n_cols);
        for (unsigned int i = 0; i< n_rows; i++)
          for (unsigned int j = 0; j < n_cols; j++)
            ret(i, j) = (*this)(i, j) * mat(0, j);
      }

      // 5. left is nx1
      else if (n_cols == 1) {
        ret = Matrix2D<T>(mat.size_rows(), mat.size_cols());
        for (unsigned int i = 0; i< mat.size_rows(); i++)
          for (unsigned int j = 0; j < mat.size_cols(); j++)
            ret(i, j) = (*this)(i, 0) * mat(i, j);
      }

      // 6. right is nx1  
      else if (mat.size_cols() == 1) {
        ret = Matrix2D<T>(n_rows, n_cols);
        for (unsigned int i = 0; i< n_rows; i++)
          for (unsigned int j = 0; j < n_cols; j++)
            ret(i, j) = (*this)(i, j) * mat(i, 0);
      }  

      return ret;
    } else {
      throw std::invalid_argument("Incompatible shapes of the two matrices.");
    }
  }

  template <typename U>
  friend std::ostream& operator<<(std::ostream& stream, Matrix2D<U>& m);
};

template <typename T>
std::ostream& operator<<(std::ostream& stream, Matrix2D<T>& m) {
  // TODO
  m.Print(stream);
  return stream;
}

#endif /* MATRIX_RECT */
