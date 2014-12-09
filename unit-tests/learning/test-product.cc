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
#define BOOST_TEST_XML_LOAD
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <fstream>

#include "utils/radix.h"
#include "utils/qsort.h"
#include "utils/mergesorter.h"

#include "omp.h"
#include <chrono>

#include <cmath>

BOOST_AUTO_TEST_CASE( test_product ) {

  float* scores = new float[5000]();

  std::cout << "init data" << std::endl;
  float* data = new float[5000 * 136];
  for (unsigned int i = 0; i < 5000 * 136; i++)
    data[i] = static_cast<float>(rand())
        / (static_cast<float>(RAND_MAX / 10.0));
  std::cout << "init weights" << std::endl;
  float* w = new float[136];
  for (unsigned int i = 0; i < 136; i++)
    w[i] = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 10.0));

  auto start_serial = std::chrono::high_resolution_clock::now();
  for (unsigned int r = 0; r < 100; r++) {
    for (unsigned int d = 0; d < 5000; d++) {
      for (unsigned int f = 0; f < 136; f++) {
        scores[d] += w[f]*data[d*136 + f];
      }
    }
  }
  auto end_serial = std::chrono::high_resolution_clock::now();
  double time_serial = std::chrono::duration_cast<std::chrono::duration<double>>(
      end_serial - start_serial).count();

  auto start_parallel = std::chrono::high_resolution_clock::now();
  for (unsigned int r = 0; r < 100; r++) {
#pragma omp parallel for num_threads(2)
    for (unsigned int d = 0; d < 5000; d++) {
      for (unsigned int f = 0; f < 136; f++) {
        scores[d] += w[f]*data[d*136 + f];
      }
    }
  }
  auto end_parallel = std::chrono::high_resolution_clock::now();
  double time_parallel = std::chrono::duration_cast<std::chrono::duration<double>>(
      end_parallel - start_parallel).count();


  std::cout
      << "Random score: "
      << scores[static_cast<int>(static_cast<float>(rand())
          / (static_cast<float>(RAND_MAX / 5000.0)))]
      << std::endl;

  std::cout << "Serial time: " << time_serial << std::endl;
  std::cout << "Parallel time: " << time_parallel << std::endl;


  delete[] scores;
  delete[] w;
  delete[] data;
}
