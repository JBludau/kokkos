#include <Kokkos_Core.hpp>
#include <iostream>

void print_furr()
{
 Kokkos::initialize();
 Kokkos::print_configuration(std::cout);

 std::cout << "Hello from furr within library with private kokkos dependency \n"; 

 Kokkos::finalize();
}
