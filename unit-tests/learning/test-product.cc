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

  unsigned int cache_size = 3000000/sizeof(float);
  unsigned int num_rounds = 10;
  unsigned int num_docs = 10*cache_size; // this is more than 3MB
  unsigned int num_fx = 136;

  std::cout << "Num docs: " << num_docs << std::endl;

  float* scores = new float[num_docs]();

  std::cout << "init data" << std::endl;
  float* data = new float[num_docs * num_fx];
  for (unsigned int i = 0; i < num_docs * num_fx; i++)
    data[i] = static_cast<float>(rand())
        / (static_cast<float>(RAND_MAX / 10.0));
  std::cout << "init weights" << std::endl;
  float* w = new float[num_fx];
  for (unsigned int i = 0; i < num_fx; i++)
    w[i] = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 10.0));

  auto start_serial = std::chrono::high_resolution_clock::now();
  for (unsigned int r = 0; r < num_rounds; r++) {
    for (unsigned int d = 0; d < num_docs; d++) {
      for (unsigned int f = 0; f < num_fx; f++) {
        scores[d] += w[f]*data[d*num_fx + f];
      }
    }
  }
  auto end_serial = std::chrono::high_resolution_clock::now();
  double time_serial = std::chrono::duration_cast<std::chrono::duration<double>>(
      end_serial - start_serial).count();

  auto start_parallel = std::chrono::high_resolution_clock::now();
  for (unsigned int r = 0; r < num_rounds; r++) {
#pragma omp parallel for //num_threads(2)
    for (unsigned int d = 0; d < num_docs; d++) {
      for (unsigned int f = 0; f < num_fx; f++) {
        scores[d] += w[f]*data[d*num_fx + f];
      }
    }
  }
  auto end_parallel = std::chrono::high_resolution_clock::now();
  double time_parallel = std::chrono::duration_cast<std::chrono::duration<double>>(
      end_parallel - start_parallel).count();


  std::cout
      << "Random score: "
      << scores[static_cast<int>(static_cast<float>(rand())
          / (static_cast<float>(RAND_MAX) / num_docs))]
      << std::endl;

  std::cout << "Serial time: " << time_serial << std::endl;
  std::cout << "Parallel time: " << time_parallel << std::endl;


  delete[] scores;
  delete[] w;
  delete[] data;
}
