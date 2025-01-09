#include <iostream>

void print_furr();

void print_full() {
  print_furr();
  std::cout << "Hello from full within library depending on library with "
               "private kokkos dependency \n";
}
