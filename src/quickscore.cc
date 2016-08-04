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
#include <vector>

#include "paramsmap/paramsmap.h"

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

  ParamsMap pmap;

  // Declare the supported options.
  pmap.addMessage({"QuickScore options:"});
  pmap.addOption("help", "h", {"print help message"});
  pmap.addOptionWithArg<std::string>("dataset", "d",
                                     {"Input dataset in SVML format"});
  pmap.addOptionWithArg<int>("rounds", "r", {"Number of test repetitions"}, 10);
  pmap.addOptionWithArg<std::string>("scores", "s",
                                     {"File where scores are saved (Optional)."});

  bool parse_status = pmap.parse(argc, argv);
  if (!parse_status || pmap.isSet("help") || !pmap.isSet("dataset")) {
    std::cout << pmap.help();
    return EXIT_FAILURE;
  }

  // parameters
  std::string dataset_file = pmap.get<std::string>("dataset");
  size_t rounds = pmap.get<int>("rounds");
  std::string scores_file;
  if (pmap.isSet("scores")) scores_file = pmap.get<std::string>("scores");


  // read dataset
  quickrank::io::Svml reader;
  auto dataset = reader.read_horizontal(dataset_file);
  std::cout << *dataset;

  // score dataset
  std::vector<double> scores(dataset->num_instances());
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
    output << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (size_t i = 0; i < dataset->num_instances(); i++) {
      output << scores[i] << std::endl;
    }
    output.close();
  }

  return EXIT_SUCCESS;
}

