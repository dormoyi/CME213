#include <cassert>
#include <cstring>
#include <iostream>
#include <sstream>

#include "gtest/gtest.h"
#include "matrix_rect.hpp"

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
Test your implementation by writing tests that cover most scenarios of 2D matrix
broadcasting. Say you are testing the result C = A * B, test with:*/


void populate2x3(Matrix2D<int> &matrix) {
  // we need the & to actually modify the matrix
  matrix(0,0) = 1;
  matrix(0,1) = 2;
  matrix(0,2) = 3;
  matrix(1,0) = 4;
  matrix(1,1) = 5;
  matrix(1,2) = 6;
  }

void populate1x3(Matrix2D<int> &matrix) {
  // we need the & to actually modify the matrix
  matrix(0,0) = 1;
  matrix(0,1) = 2;
  matrix(0,2) = 3;
  }

void populate2x1(Matrix2D<int> &matrix) {
  // we need the & to actually modify the matrix
  matrix(0,0) = 1;
  matrix(1,0) = 2;
  }

// 1. A of shape (m != 1, n != 1), B of shape (m != 1, n != 1)
TEST(testMatrix, shape1){
  Matrix2D<int> matrix1(2, 3);
  populate2x3(matrix1);
  Matrix2D<int> res = matrix1.dot(matrix1);
  EXPECT_EQ(res(0,0), 1) << "Wrong value";
  EXPECT_EQ(res(0,1), 4) << "Wrong value";
  EXPECT_EQ(res(0,2), 9) << "Wrong value";
  EXPECT_EQ(res(1,0), 16) << "Wrong value";
  EXPECT_EQ(res(1,1), 25) << "Wrong value";
  EXPECT_EQ(res(1,2), 36) << "Wrong value";
}

// 2. A of shape (1, n != 1), B of shape (m != 1, n != 1)
// 2bis. A of shape (m != 1, n != 1), B of shape (1, n != 1)
TEST(testMatrix, shape2){
  Matrix2D<int> matrix1(1, 3);
  populate1x3(matrix1);
  Matrix2D<int> matrix2(2,3);
  populate2x3(matrix2);
  Matrix2D<int> res = matrix1.dot(matrix2);
  EXPECT_EQ(res(0,0), 1) << "Wrong value";
  EXPECT_EQ(res(0,1), 4) << "Wrong value";
  EXPECT_EQ(res(0,2), 9) << "Wrong value";
  EXPECT_EQ(res(1,0), 4) << "Wrong value";
  EXPECT_EQ(res(1,1), 10) << "Wrong value";
  EXPECT_EQ(res(1,2), 18) << "Wrong value";
  Matrix2D<int> res2 = matrix2.dot(matrix1);
  EXPECT_EQ(res2(0,0), 1) << "Wrong value";
  EXPECT_EQ(res2(0,1), 4) << "Wrong value";
  EXPECT_EQ(res2(0,2), 9) << "Wrong value";
  EXPECT_EQ(res2(1,0), 4) << "Wrong value";
  EXPECT_EQ(res2(1,1), 10) << "Wrong value";
  EXPECT_EQ(res2(1,2), 18) << "Wrong value";
}

// 3. A of shape (m != 1, n != 1), B of shape (m != 1, 1)
// 3bis. A of shape (m != 1, 1), B of shape (m != 1, n != 1)
TEST(testMatrix, shape3){
  Matrix2D<int> matrix1(2, 3);
  populate2x3(matrix1);
  Matrix2D<int> matrix2(2,1);
  populate2x1(matrix2);
  Matrix2D<int> res = matrix1.dot(matrix2);
  EXPECT_EQ(res(0,0), 1) << "Wrong value";
  EXPECT_EQ(res(0,1), 2) << "Wrong value";
  EXPECT_EQ(res(0,2), 3) << "Wrong value";
  EXPECT_EQ(res(1,0), 8) << "Wrong value";
  EXPECT_EQ(res(1,1), 10) << "Wrong value";
  EXPECT_EQ(res(1,2), 12) << "Wrong value";
  Matrix2D<int> res2 = matrix2.dot(matrix1);
  EXPECT_EQ(res2(0,0), 1) << "Wrong value";
  EXPECT_EQ(res2(0,1), 2) << "Wrong value";
  EXPECT_EQ(res2(0,2), 3) << "Wrong value";
  EXPECT_EQ(res2(1,0), 8) << "Wrong value";
  EXPECT_EQ(res2(1,1), 10) << "Wrong value";
  EXPECT_EQ(res2(1,2), 12) << "Wrong value";
}

// 4. A of shape (1, 1), B of shape (m != 1, n != 1)
// 4bis. A of shape (m != 1, n != 1), B of shape (1, 1)
TEST(testMatrix, shape4){
  Matrix2D<int> matrix1(1, 1);
  matrix1(0,0) = 3;
  Matrix2D<int> matrix2(2,3);
  populate2x3(matrix2);
  Matrix2D<int> res = matrix1.dot(matrix2);
  EXPECT_EQ(res(0,0), 3) << "Wrong value";
  EXPECT_EQ(res(0,1), 6) << "Wrong value";
  EXPECT_EQ(res(0,2), 9) << "Wrong value";
  EXPECT_EQ(res(1,0), 12) << "Wrong value";
  EXPECT_EQ(res(1,1), 15) << "Wrong value";
  EXPECT_EQ(res(1,2), 18) << "Wrong value";
  Matrix2D<int> res2 = matrix2.dot(matrix1);
  EXPECT_EQ(res2(0,0), 3) << "Wrong value";
  EXPECT_EQ(res2(0,1), 6) << "Wrong value";
  EXPECT_EQ(res2(0,2), 9) << "Wrong value";
  EXPECT_EQ(res2(1,0), 12) << "Wrong value";
  EXPECT_EQ(res2(1,1), 15) << "Wrong value";
  EXPECT_EQ(res2(1,2), 18) << "Wrong value";
}
