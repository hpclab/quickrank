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

#include "types.h"
#include "learning/forests/lambdamart.h"
#include "learning/tree/rt.h"
#include "learning/tree/ensemble.h"
#include "learning/meta/meta_cleaver.h"

namespace quickrank {
namespace learning {
namespace forests {

class Dart: public LambdaMart {

 public:

  enum class SamplingType {
    UNIFORM, WEIGHTED, WEIGHTED_INV, COUNT2, COUNT3
  };

  enum class NormalizationType {
    TREE, NONE, WEIGHTED, FOREST, TREE_ADAPTIVE
  };

  /// Initializes a new Dart instance with the given learning parameters.
  /// LambdaMart parameters are not reported
  ///
  /// \param sample_type Sampling type for selecting trees to dropout
  /// \param normalize_type Normalization strategy to adopt
  /// \param rate_drop Probability to dropout a single tree
  /// \param skip_drop Probability to skip the dropout phase
  Dart(size_t ntrees, double shrinkage, size_t nthresholds,
       size_t ntreeleaves, size_t minleafsupport,
       size_t valid_iterations,
       SamplingType sample_type, NormalizationType normalize_type,
       double rate_drop, double skip_drop, bool keep_drop)
      : LambdaMart(ntrees, shrinkage, nthresholds, ntreeleaves,
             minleafsupport, valid_iterations),
        sample_type(sample_type),
        normalize_type(normalize_type),
        rate_drop(rate_drop),
        skip_drop(skip_drop),
        keep_drop(keep_drop) {
  }

  /// Generates a LTR_Algorithm instance from a previously saved XML model.
  Dart(const pugi::xml_document &model);

  virtual ~Dart();

  /// Returns the name of the ranker.
  virtual std::string name() const {
    return NAME_;
  }

  static const std::string NAME_;

  /// Start the learning process.
  virtual void learn(std::shared_ptr<data::Dataset> training_dataset,
                     std::shared_ptr<data::Dataset> validation_dataset,
                     std::shared_ptr<metric::ir::Metric> training_metric,
                     size_t partial_save,
                     const std::string output_basename);

  /// Returns the score by the current ranker
  ///
  /// \param d Document to be scored.
  virtual Score score_document(const Feature *d) const {
    return ensemble_model_.score_instance(d, 1);
  }

  static const std::vector<std::string> samplingTypesNames;

  static SamplingType get_sampling_type(std::string name) {
    auto i_item = std::find(samplingTypesNames.cbegin(),
                            samplingTypesNames.cend(),
                            name);
    if (i_item != samplingTypesNames.cend()) {

      return SamplingType(std::distance(samplingTypesNames.cbegin(), i_item));
    }

    // TODO: Fix return value...
    throw std::invalid_argument("sampling type " + name + " is not valid");
//    return NULL;
  }

  static std::string get_sampling_type(SamplingType samplingType) {
    return samplingTypesNames[static_cast<int>(samplingType)];
  }

  static const std::vector<std::string> normalizationTypesNames;

  static NormalizationType  get_normalization_type(std::string name) {
    auto i_item = std::find(normalizationTypesNames.cbegin(),
                            normalizationTypesNames.cend(),
                            name);
    if (i_item != normalizationTypesNames.cend()) {

      return NormalizationType (std::distance(normalizationTypesNames.cbegin(), i_item));
    }

    // TODO: Fix return value...
    throw std::invalid_argument("normalization type " + name + " is not valid");
//    return NULL;
  }

  static std::string get_normalization_type(NormalizationType normalizationType) {
    return normalizationTypesNames[static_cast<int>(normalizationType)];
  }

 protected:

  virtual pugi::xml_document *get_xml_model() const;

  virtual bool import_model_state(LTR_Algorithm &other);

  virtual void update_modelscores(std::shared_ptr<data::Dataset> dataset,
                                  bool add, Score *scores,
                                  std::vector<int>& dropped_trees);
  virtual void update_modelscores(std::shared_ptr<data::VerticalDataset> dataset,
                                  bool add, Score *scores,
                                  std::vector<int>& dropped_trees);

 protected:
  SamplingType sample_type;
  NormalizationType normalize_type;
  double rate_drop;           // dropout rate
  double skip_drop;           // probability of skipping dropout
  bool keep_drop;

 private:

  /// Prints the description of Algorithm, including its parameters.
  virtual std::ostream &put(std::ostream &os) const;

  std::vector<int> select_trees_to_dropout(std::vector<double>& weights,
                                           size_t trees_to_dropout);

  void normalize_trees_restore_drop(std::vector<double> &weights,
                                    std::vector<int> dropped_trees);

  void set_weight_last_tree(std::vector<double> &weights,
                            std::vector<int> dropped_trees);

  int binary_search(std::vector<double>& array, double elem);
};

template <typename T>
std::ostream& operator<< (std::ostream& out, const std::vector<T>& v) {
  if ( !v.empty() ) {
    out << '[';
    std::copy (v.begin(), v.end(), std::ostream_iterator<T>(out, ", "));
    out << "\b\b]" << std::endl;
  }
  return out;
}

}  // namespace forests
}  // namespace learning
}  // namespace quickrank

