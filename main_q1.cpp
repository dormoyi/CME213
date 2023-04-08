#include <cassert>
#include <cstring>
#include <iostream>
#include <sstream>

#include "gtest/gtest.h"
#include "matrix.hpp"

TEST(testMatrix, sampleTest) {
  ASSERT_EQ(1000, 1000)
      << "This does not fail, hence this message is not printed.";
  EXPECT_EQ(2000, 2000)
      << "This does not fail, hence this message is not printed.";
  // If uncommented, the following line will make this test fail.
  // EXPECT_EQ(2000, 3000) << "This expect statement fails, and this message
  // will be printed.";
}

/*
TODO:

For both the MatrixDiagonal and the MatrixSymmetric classes, do the following:

Write at least the following tests to get full credit here:*/

//1. Declare an empty matrix with the default constructor for MatrixSymmetric.
MatrixSymmetric<int> symmetric_matrix;
MatrixDiagonal<int> diagonal_matrix;
// Assert that the NormL0 and size functions return appropriate values for these.
TEST(testMatrix, emptyMatrix) {
  EXPECT_EQ(symmetric_matrix.NormL0(), 0) << "Wrong NormL0 value";
  EXPECT_EQ(symmetric_matrix.size(), 0) << "Wrong size value";
  EXPECT_EQ(diagonal_matrix.NormL0(), 0) << "Wrong NormL0 value";
  EXPECT_EQ(diagonal_matrix.size(), 0) << "Wrong size value";
}

// 2. Using the second constructor that takes size as argument, create a matrix of
// size zero. Repeat the assertions from (1).
MatrixSymmetric<int> symmetric_matrix2(0);
MatrixDiagonal<int> diagonal_matrix2(0);
TEST(testMatrix, emptyMatrix2) {
  EXPECT_EQ(symmetric_matrix2.NormL0(), 0) << "Wrong NormL0 value";
  EXPECT_EQ(symmetric_matrix2.size(), 0) << "Wrong size value";
  EXPECT_EQ(diagonal_matrix2.NormL0(), 0) << "Wrong NormL0 value";
  EXPECT_EQ(diagonal_matrix2.size(), 0) << "Wrong size value";
}

// // 3. Provide a negative argument to the second constructor and assert that the
// // constructor throws an exception.
TEST(testMatrix, invalidArgument) {
  EXPECT_THROW(MatrixSymmetric<int> symmetric_matrix3(-1), std::invalid_argument);
  EXPECT_THROW(MatrixDiagonal<int> diagonal_matrix3(-1), std::invalid_argument);
}

// // 4. Create and initialize a matrix of some size, and verify that the NormL0
// // function returns the correct value.
MatrixSymmetric<int> symmetric_matrix4(4);
//symmetric_matrix4(0,0) = 0;
MatrixDiagonal<int> matrix(4);
matrix(0,0) = 0;
// symmetric_matrix4(0,1) = 0;
// symmetric_matrix4(0,2) = 3;
// symmetric_matrix4(0,3) = 4;
// symmetric_matrix4(1,1) = 5;
// symmetric_matrix4(1,2) = 6;
// symmetric_matrix4(1,3) = 7;
// symmetric_matrix4(2,2) = 8;
// symmetric_matrix4(2,3) = 9;
// symmetric_matrix4(3,3) = 10;
TEST(testMatrix, symmetric_NormL0) {
  EXPECT_EQ(symmetric_matrix4.NormL0(), 3) << "Wrong NormL0 value";
}

// 5. Create a matrix, initialize some or all of its elements, then retrieve and
// check that they are what you initialized them to.

// 6. Create a matrix of some size. Make an out-of-bounds access into it and check
// that an exception is thrown.

// 7. Test the stream operator using std::stringstream and using the "<<" operator.
