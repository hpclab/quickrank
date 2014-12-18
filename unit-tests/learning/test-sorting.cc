/*
 * QuickRank - A C++ suite of Learning to Rank algorithms
 * Webpage: http://quickrank.isti.cnr.it/
 * Contact: quickrank@isti.cnr.it
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
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
