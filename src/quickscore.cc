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
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <string>

#include <boost/program_options.hpp>

#include "data/dataset.h"
#include "io/svml.h"

void print_logo() {
  if (isatty(fileno(stdout))) {
    std::string color_reset = "\033[0m";
    std::string color_logo = "\033[1m\033[32m";
    std::cout
        << color_logo
        << std::endl
        << "      _____  _____"
        << std::endl
        << "     /    / /____/"
        << std::endl
        << "    /____\\ /    \\          QuickRank has been developed by hpc.isti.cnr.it"
        << std::endl
        << "    ::Quick:Rank::                                   quickrank@isti.cnr.it"
        << std::endl << color_reset << std::endl;
  } else {
    std::cout
        << std::endl
        << "      _____  _____"
        << std::endl
        << "     /    / /____/"
        << std::endl
        << "    /____\\ /    \\          QuickRank has been developed by hpc.isti.cnr.it"
        << std::endl
        << "    ::Quick:Rank::                                   quickrank@isti.cnr.it"
        << std::endl << std::endl;
  }
}

double ranker(float *v);

int main(int argc, char *argv[]) {
  print_logo();

  namespace po = boost::program_options;

  // parameters
  std::string dataset_file;
  size_t rounds = 10;
  std::string scores_file;

  // prepare options
  po::options_description options("Options");
  options.add_options()("help,h", "Print help messages");
  options.add_options()("dataset,d",
                        po::value<std::string>(&dataset_file)->required(),
                        "Input dataset in SVML format");
  options.add_options()("rounds,r",
                        po::value<size_t>(&rounds)->default_value(rounds),
                        "Number of test repetitions");
  options.add_options()(
      "scores,s",
      po::value<std::string>(&scores_file)->default_value(scores_file),
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
  double *scores = new double[dataset->num_instances()];
  auto start_scoring = std::chrono::high_resolution_clock::now();

  for (size_t r = 0; r < rounds; r++) {
    float *document = dataset->at(0, 0);
    for (size_t i = 0; i < dataset->num_instances(); i++) {
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
    output.open(scores_file, std::ofstream::out);
    output << std::setprecision(std::numeric_limits<double>::digits10);
    for (size_t i = 0; i < dataset->num_instances(); i++) {
      output << scores[i] << std::endl;
    }
    output.close();
  }

  delete[] scores;

  return EXIT_SUCCESS;
}

