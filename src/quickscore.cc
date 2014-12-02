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
#include <string>

#include <boost/program_options.hpp>

#include "data/dataset.h"
#include "io/svml.h"

double ranker(float* v);

int main(int argc, char *argv[]) {
  std::cout << "# ## ================================== ## #" << std::endl
            << "# ##              QuickRank             ## #" << std::endl
            << "# ## ---------------------------------- ## #" << std::endl
            << "# ##     developed by the HPC. Lab.     ## #" << std::endl
            << "# ##      http://hpc.isti.cnr.it/       ## #" << std::endl
            << "# ##      quickrank@.isti.cnr.it        ## #" << std::endl
            << "# ## ================================== ## #" << std::endl;

  namespace po = boost::program_options;

  // parameters
  std::string dataset_file;
  unsigned int rounds;
  std::string scores_file;

  // prepare options
  po::options_description options("Options");
  options.add_options()("help,h", "Print help messages");
  options.add_options()("dataset,d",
                        po::value<std::string>(&dataset_file)->required(),
                        "Input dataset in SVML format");
  options.add_options()("rounds,r",
                        po::value<unsigned int>(&rounds)->default_value(10),
                        "Number of test repetitions");
  options.add_options()(
      "scores,s",
      po::value<std::string>(&scores_file)->default_value(std::string()),
      "File where scores are saved");

  // parse command line
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, options), vm);

  // print help
  if (vm.count("help")) {
    std::cout << options << "\n";
    return EXIT_FAILURE;
  }

  // raise any error
  po::notify(vm);

  // read dataset
  quickrank::io::Svml reader;
  auto dataset = reader.read_horizontal(dataset_file);
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
  if (!scores_file.empty()) {
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

