#include <cassert>
#include <iostream>
#include <random>
#include <set>
#include <stdexcept>

// TODO: add your function here. The function should count the number of
// entries between lb and ub.
// You need to use the std::set::lower_bound and std::set::upper_bound functions
// to find the first and last elements in the range.
unsigned int count_range(const std::set<double>& data, const double lb,
                         const double ub) {
  if (lb > ub) 
    throw std::invalid_argument("Lower bound higher than upper bound");
  
  unsigned int entries_nb = 0;
  auto start = data.lower_bound(lb); // Returns an iterator pointing to the first element in the set
  // that is not inferior to lb
  auto stop = data.upper_bound(ub); // Returns an iterator pointing to the first element in the set
  // that is greater than ub
  // The auto keyword directs the compiler to use the initialization expression of a declared variable, 
  // or lambda expression parameter, to deduce its type.
  for (auto it = start; it != stop; it++) {
      entries_nb++;
  }
  return entries_nb;
}

int main() {
  std::set<double> data_simple{0, 1, 2, 3, 4, 5, 6};

  // Range test
  try {
    count_range(data_simple, 1, 0);
    std::cout << "Error: range test." << std::endl;
  } catch (const std::exception& error) {
    // This line is expected to be run
    std::cout << "Range test passed." << std::endl;
  }

  // Count test
  assert(count_range(data_simple, 3, 6) == 4);

  // Test with N(0,1) data.
  std::set<double> data_rng;
  std::default_random_engine generator;
  std::normal_distribution<double> distribution(0.0, 1.0);
  unsigned int n = 10000;
  for (unsigned int i = 0; i < n; ++i) data_rng.insert(distribution(generator));

  std::cout << "Number of elements in range [-1, 1]: "
            << count_range(data_rng, -1, 1) << " (est. = " << 0.683 * n << ")"
            << std::endl;

  return 0;
}
