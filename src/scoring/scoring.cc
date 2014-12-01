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
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>

#include "data/dataset.h"
#include "io/svml.h"

double ranker(float* v);

int main(int argc, char *argv[]) {
  std::cout << "# ## ========================== ## #" << std::endl
            << "# ##          QuickRank         ## #" << std::endl
            << "# ## -------------------------- ## #" << std::endl
            << "# ## developed by the HPC. Lab. ## #" << std::endl
            << "# ##  http://hpc.isti.cnr.it/   ## #" << std::endl
            << "# ##  quickrank@.isti.cnr.it    ## #" << std::endl
            << "# ## ========================== ## #" << std::endl;

  std::cout << "usage: quickscore <dataset> [output scores file]" << std::endl;
  char* data_file = argv[1];

  unsigned int rounds = 10;

  // read dataset
  quickrank::io::Svml reader;
  auto dataset = reader.read_horizontal(data_file);
  std::cout << *dataset;

  // score dataset
  double* scores = new double[dataset->num_instances()];
  auto start_scoring = std::chrono::high_resolution_clock::now();

  for (unsigned int r = 0; r < rounds; r++) {
    float* document = dataset->at(0, 0);
    for (unsigned int i = 0; i < dataset->num_instances(); i++) {
      scores[i] = ranker(document);
      document += dataset->num_features();
    }
  }

  auto end_scoring = std::chrono::high_resolution_clock::now();

  double scoring_time =
      std::chrono::duration_cast<std::chrono::duration<double>>(
          end_scoring - start_scoring).count();

  std::cout << "       Total scoring time: " << scoring_time << " s."
            << std::endl;
  std::cout << "Avg. Dataset scoring time: " << scoring_time / rounds << " s."
            << std::endl;
  std::cout << "Avg.    Doc. scoring time: "
            << scoring_time / dataset->num_instances() / rounds << " s."
            << std::endl;

  // potentially save scores
  if (argc > 2) {
    std::fstream output;
    output.open(argv[2], std::ofstream::out);
    output << std::setprecision(15);
    for (unsigned int i = 0; i < dataset->num_instances(); i++) {
      output << scores[i] << std::endl;
    }
    output.close();
  }

  delete[] scores;
  return EXIT_SUCCESS;
}

