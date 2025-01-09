#include <iostream>

void print_fuzz();

void print_full() {
  print_fuzz();
  std::cout << "Hello from full within library depending on library with "
               "public kokkos dependency \n";
}
