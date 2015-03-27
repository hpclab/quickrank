/*
 * QuickRank - A C++ suite of Learning to Rank algorithms
 * Webpage: http://quickrank.isti.cnr.it/
 * Contact: quickrank@isti.cnr.it
 *
 * Unless explicitly acquired and licensed from Licensor under another
 * license, the contents of this file are subject to the Reciprocal Public
 * License ("RPL") Version 1.5, or subsequent versions as allowed by the RPL,
 * and You may not copy or use this file in either source code or executable
 * form, except in compliance with the terms and conditions of the RPL.
 *
 * All software distributed under the RPL is provided strictly on an "AS
 * IS" basis, WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESS OR IMPLIED, AND
 * LICENSOR HEREBY DISCLAIMS ALL SUCH WARRANTIES, INCLUDING WITHOUT
 * LIMITATION, ANY WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
 * PURPOSE, QUIET ENJOYMENT, OR NON-INFRINGEMENT. See the RPL for specific
 * language governing rights and limitations under the RPL.
 *
 * Contributor:
 *   HPC. Laboratory - ISTI - CNR - http://hpc.isti.cnr.it/
 */
#define BOOST_TEST_SORTING_TOOLS
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <fstream>

#include "utils/radix.h"

#include "omp.h"
#include <chrono>

#include <cmath>

BOOST_AUTO_TEST_CASE( test_sort ) {

  // 3000 on Mac os
  int sizes[] = { 1000, 2000, 3000, 4000, 5000 };
  int num_exp = sizeof(sizes) / sizeof(int);
  int rounds = 10;

  std::cout << "# Float Sorting" << std::endl;
  std::cout << "# Init data..." << std::endl;
  float* orig = new float[sizes[num_exp - 1]];
  for (int i = 0; i < sizes[num_exp - 1]; i++)
    orig[i] = static_cast<float>(rand())
        / (static_cast<float>(RAND_MAX / 10.0));

  // ---------
  // STD::SORT
  for (int s = 0; s < num_exp; s++) {
    // prepare data for all the rounds
    float* data = new float[sizes[s] * rounds];
    for (int r = 0; r < rounds; r++) {
      for (int i = 0; i < sizes[s]; i++)
        data[i + r * sizes[s]] = orig[i];
    }
    // execute rounds
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < rounds; r++) {
      std::sort(data + r * sizes[s], data + (r + 1) * sizes[s]);
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(
        end_time - start_time).count();
    elapsed /= rounds;
    std::cout << "std::sort on " << sizes[s] << " elements: " << elapsed
              << std::endl;

    delete[] data;
  }

  // ----------
  // RADIX SORT
  for (int s = 0; s < num_exp; s++) {
    // prepare data for all the rounds
    float* data = new float[sizes[s] * rounds];
    for (int r = 0; r < rounds; r++) {
      for (int i = 0; i < sizes[s]; i++)
        data[i + r * sizes[s]] = orig[i];
    }
    // execute rounds
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < rounds; r++) {
      delete [] idxfloat_radixsort(data + r * sizes[s], sizes[s]);
      //float_radixsort<ascending>(data + r * sizes[s], sizes[s]);
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(
        end_time - start_time).count();
    elapsed /= rounds;
    std::cout << "Radix-sort on " << sizes[s] << " elements: " << elapsed
              << std::endl;

    delete[] data;
  }

  delete[] orig;
}
