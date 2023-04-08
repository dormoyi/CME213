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
  }

  // Constructor takes argument (m,n) = matrix dimension.
  Matrix2D(const int m, const int n) {
      // TODO: Hint: The data_ should be resized to have m * n elements
  }

  unsigned int size_rows() const { return 0; } // TODO
  unsigned int size_cols() const { return 0; } // TODO

  // Returns reference to matrix element (i, j).
  T& operator()(int i, int j) {
    // TODO: Hint: Element (i,j) for 0 <= i < n_rows and 0 <= j < n_cols 
    // is stored at data[i * n_cols + j]. 
    return data_[0];
  }
    
  void Print(std::ostream& ostream) {
      // TODO
  }

  Matrix2D<T> dot(Matrix2D<T>& mat) {
    if (n_rows == mat.size_rows() && n_cols == mat.size_cols()) {
      Matrix2D<T> ret(n_rows, n_cols);
      for (int i = 0; i < n_rows; i++) {
        for (int j = 0; j < n_cols; j++) {
          ret(i, j) = (*this)(i, j) * mat(i, j);
        }
      }
      return ret;
    } else if (Broadcastable<T>(*this, mat)) {
      // TODO: Replace the code in this scope.
      // Compute and return the elementwise product of the two Matrix2D's
      // "*this" and "mat" after appropriate broadcasting.
      Matrix2D<T> ret;

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
  return stream;
}

#endif /* MATRIX_RECT */
