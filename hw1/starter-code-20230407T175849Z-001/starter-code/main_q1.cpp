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

unsigned int populate_symmetric1(MatrixSymmetric<int> &matrix) {
  // we need the & to actually modify the matrix
  matrix(0,0) = 0;
  matrix(0,1) = 0;
  matrix(0,2) = 3;
  matrix(0,3) = 4;
  matrix(1,1) = 5;
  matrix(2,1) = 6;
  matrix(1,3) = 7;
  matrix(2,2) = 8;
  matrix(2,3) = 0;
  matrix(3,3) = 10;
  return matrix.NormL0();
}

unsigned int populate_diagonal1(MatrixDiagonal<int> &matrix) {
  matrix(0,0) = 0;
  matrix(1,1) = 1;
  matrix(2,2) = 2;
  matrix(3,3) = 3;
  return matrix.NormL0();
}

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
TEST(testMatrix, NormL0) {
  MatrixSymmetric<int> symmetric_matrix4(4);
  unsigned int res = populate_symmetric1(symmetric_matrix4);
  EXPECT_EQ(res, 11) << "Wrong NormL0 value";

  MatrixDiagonal<int> diagonal_matrix4(4);
  unsigned int res2 = populate_diagonal1(diagonal_matrix4);
  EXPECT_EQ(res2, 3) << "Wrong NormL0 value";
}

// 5. Create a matrix, initialize some or all of its elements, then retrieve and
// check that they are what you initialized them to.
TEST(testMatrix, elements){
  MatrixSymmetric<int> symmetric_matrix5(4);
  unsigned int res = populate_symmetric1(symmetric_matrix5);
  res++; // to avoid the warning
  EXPECT_EQ(symmetric_matrix5(0,0), 0) << "Wrong value";
  EXPECT_EQ(symmetric_matrix5(0,1), 0) << "Wrong value";
  EXPECT_EQ(symmetric_matrix5(0,2), 3) << "Wrong value";
  EXPECT_EQ(symmetric_matrix5(3,0), 4) << "Wrong value";
  EXPECT_EQ(symmetric_matrix5(1,2), 6) << "Wrong value";
  EXPECT_EQ(symmetric_matrix5(1,3), 7) << "Wrong value";
  EXPECT_EQ(symmetric_matrix5(3,2), 0) << "Wrong value";

  MatrixDiagonal<int> diagonal_matrix5(4);
  unsigned int res2 = populate_diagonal1(diagonal_matrix5);
  res2++; // to avoid the warning
  EXPECT_EQ(diagonal_matrix5(0,0), 0) << "Wrong value";
  EXPECT_EQ(diagonal_matrix5(1,1), 1) << "Wrong value";
  EXPECT_EQ(diagonal_matrix5(2,2), 2) << "Wrong value";
  EXPECT_EQ(diagonal_matrix5(3,3), 3) << "Wrong value";
  EXPECT_EQ(diagonal_matrix5(0,3), 0) << "Wrong value";
  EXPECT_EQ(diagonal_matrix5(1,2), 0) << "Wrong value";
  EXPECT_EQ(diagonal_matrix5(2,1), 0) << "Wrong value";
}

// 6. Create a matrix of some size. Make an out-of-bounds access into it and check
// that an exception is thrown.
TEST(testMatrix, out_of_bounds){
  MatrixSymmetric<int> symmetric_matrix6(4);
  unsigned int res = populate_symmetric1(symmetric_matrix6); // QUES: better way to do that?
  res++; // to avoid the warning
  EXPECT_THROW(symmetric_matrix6(4,0), std::invalid_argument); // or std::out_of_range
  EXPECT_THROW(symmetric_matrix6(6,6), std::invalid_argument);

  MatrixDiagonal<int> diagonal_matrix6(4);
  unsigned int res2 = populate_diagonal1(diagonal_matrix6);
  res2++; // to avoid the warning
  EXPECT_THROW(diagonal_matrix6(4,0), std::invalid_argument);
  EXPECT_THROW(diagonal_matrix6(6,6), std::invalid_argument);
}

// 7. Test the stream operator using std::stringstream and using the "<<" operator.
TEST(testMatrix, stream_operator){
  MatrixSymmetric<int> symmetric_matrix7(4);
  unsigned int res = populate_symmetric1(symmetric_matrix7);
  res++; // to avoid the warning
  std::stringstream string_stream;
  string_stream << symmetric_matrix7;
  std::string output = string_stream.str();
  std::string expected = "0 0 3 4 \n0 5 6 7 \n3 6 8 0 \n4 7 0 10 \n"; 
  EXPECT_EQ(output, expected);

  MatrixDiagonal<int> diagonal_matrix7(4);
  unsigned int res2 = populate_diagonal1(diagonal_matrix7);
  res2++; // to avoid the warning
  std::stringstream string_stream2;
  string_stream2 << diagonal_matrix7;
  std::string output2 = string_stream2.str(); 
  std::string expected2 = "0 0 0 0 \n0 1 0 0 \n0 0 2 0 \n0 0 0 3 \n";
  EXPECT_EQ(output2, expected2);
}