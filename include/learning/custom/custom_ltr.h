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

#pragma once

#include <memory>

#include "data/dataset.h"
#include "metric/ir/metric.h"
#include "learning/ltr_algorithm.h"

namespace quickrank {
namespace learning {

/*
 * Command line
 ./bin/quickrank --algo custom \
 --train tests/data/msn1.fold1.train.5k.txt \
 --valid tests/data/msn1.fold1.vali.5k.txt \
 --test tests/data/msn1.fold1.test.5k.txt \
 --model model.xml
 */

class CustomLTR: public LTR_Algorithm {

 public:
  CustomLTR();

  CustomLTR(const pugi::xml_document &model) {
  }

  virtual ~CustomLTR();

  /// Returns the name of the ranker.
  virtual std::string name() const {
    return NAME_;
  }

  static const std::string NAME_;

  /// Executes the learning process.
  ///
  /// \param training_dataset The training dataset.
  /// \param validation_dataset The validation training dataset.
  /// \param metric The metric to be optimized.
  /// \param partial_save Allows to save a partial model every given number of iterations.
  /// \param model_filename The file where the model, and the partial models, are saved.
  virtual void learn(std::shared_ptr<data::Dataset> training_dataset,
                     std::shared_ptr<data::Dataset> validation_dataset,
                     std::shared_ptr<metric::ir::Metric> metric,
                     size_t partial_save,
                     const std::string model_filename);

  /// Returns the score of a given document.
  virtual Score score_document(const Feature *d) const;

  /// Return the xml model representing the current object
  virtual pugi::xml_document *get_xml_model() const;

  /// \todo TODO: add load_model();

  const Score FIXED_SCORE = 666.0;

 private:

  /// The output stream operator.
  friend std::ostream &operator<<(std::ostream &os, const CustomLTR &a) {
    return a.put(os);
  }

  /// Prints the description of Algorithm, including its parameters
  virtual std::ostream &put(std::ostream &os) const;
};

}  // namespace learning
}  // namespace quickrank
