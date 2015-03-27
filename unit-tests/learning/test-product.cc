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
#define BOOST_TEST_OPENMP_DOT_PRODUCT
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <fstream>

#include "omp.h"
#include <chrono>

#include <cmath>


void precompute(unsigned int num_docs, unsigned int num_fx, double* scores, double* w, float* data, unsigned int skip) {
#pragma omp parallel for //num_threads(2)
  for (unsigned int d = 0; d < num_docs; d++) {
    for (unsigned int f = 0; f < num_fx; f++) {
      if (f!=skip)
	scores[d] += w[f]*data[d*num_fx + f];
    }
  }
}

BOOST_AUTO_TEST_CASE( test_product ) {

  unsigned int cache_size = 2000000/sizeof(float);
  unsigned int num_rounds = 10;
  unsigned int num_docs = 10*cache_size; // this is more than 3MB
  unsigned int num_fx = 136;

  std::cout << "Num docs: " << num_docs << std::endl;

  double* scores = new double[num_docs]();

  std::cout << "init data" << std::endl;
  float* data = new float[num_docs * num_fx];
  for (unsigned int i = 0; i < num_docs * num_fx; i++)
    data[i] = static_cast<float>(rand())
        / (static_cast<float>(RAND_MAX / 10.0));
  std::cout << "init weights" << std::endl;
  double* w = new double[num_fx];
  for (unsigned int i = 0; i < num_fx; i++)
    w[i] = static_cast<double>(rand()) / (static_cast<double>(RAND_MAX / 10.0));

  /*
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
  */

  auto start_parallel = std::chrono::high_resolution_clock::now();
  for (unsigned int r = 0; r < num_rounds; r++) {
    precompute(num_docs, num_fx, scores, w, data, r);
    //#pragma omp parallel for //num_threads(2)
    //for (unsigned int d = 0; d < num_docs; d++) {
    //  for (unsigned int f = 0; f < num_fx; f++) {
    //    scores[d] += w[f]*data[d*num_fx + f];
    //  }
    //}
  }
  auto end_parallel = std::chrono::high_resolution_clock::now();
  double time_parallel = std::chrono::duration_cast<std::chrono::duration<double>>(
      end_parallel - start_parallel).count();


  std::cout
      << "Random score: "
      << scores[static_cast<int>(static_cast<float>(rand())
          / (static_cast<float>(RAND_MAX) / num_docs))]
      << std::endl;

  //std::cout << "Serial time: " << time_serial << std::endl;
  std::cout << "Parallel time: " << time_parallel << std::endl;


  delete[] scores;
  delete[] w;
  delete[] data;
}
