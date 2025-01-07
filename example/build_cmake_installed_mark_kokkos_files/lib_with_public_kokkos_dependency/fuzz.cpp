#include <Kokkos_Core.hpp>
#include <iostream>

void print_fuzz(Kokkos::View<int*> a)
{
 std::cout << "Hello from fuzz within library with public kokkos dependency \n"; 
}
