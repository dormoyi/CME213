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
broadcasting. Say you are testing the result C = A * B, test with:
1. A of shape (m != 1, n != 1), B of shape (m != 1, n != 1)
2. A of shape (1, n != 1), B of shape (m != 1, n != 1)
3. A of shape (m != 1, n != 1), B of shape (m != 1, 1)
4. A of shape (1, 1), B of shape (m != 1, n != 1)
Please test any more cases that you can think of.
*/
