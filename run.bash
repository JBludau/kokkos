#!/bin/bash

for contention in 1000000 100000 10000 1000 100 50 20 10 5
  do
  ./Kokkos_atomic 1000000 ${contention} 1 1 1 10 4
  echo "Ran Kokkos_atomic 1000000 ${contention} 1 1 1 10 4"
done
