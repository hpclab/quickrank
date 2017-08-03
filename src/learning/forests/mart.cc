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
#include "learning/forests/mart.h"

#include <fstream>
#include <iomanip>
#include <chrono>

#include "utils/radix.h"

namespace quickrank {
namespace learning {
namespace forests {

const std::string Mart::NAME_ = "MART";

Mart::Mart(const pugi::xml_document &model) {
  ntrees_ = 0;
  shrinkage_ = 0;
  nthresholds_ = 0;
  nleaves_ = 0;
  minleafsupport_ = 0;
  valid_iterations_ = 0;

  pugi::xml_node model_info = model.child("ranker").child("info");
  pugi::xml_node model_tree = model.child("ranker").child("ensemble");

  // read (training) info
  ntrees_ = model_info.child("trees").text().as_int();
  nleaves_ = model_info.child("leaves").text().as_int();
  minleafsupport_ = model_info.child("leafsupport").text().as_int();
  nthresholds_ = model_info.child("discretization").text().as_int();
  valid_iterations_ = model_info.child("estop").text().as_int();
  shrinkage_ = model_info.child("shrinkage").text().as_double();

  // read ensemble
  ensemble_model_.set_capacity(ntrees_);

  // loop over trees
  for (const auto &tree: model_tree.children()) {
    RTNode *root = NULL;
    float tree_weight = tree.attribute("weight").as_float();

    const auto &root_split = tree.child("split");
    if (root_split)
      root = RTNode::parse_xml(root_split);

    if (root == NULL) {
      std::cerr << "!!! Unable to parse tree from XML model." << std::endl;
      exit(EXIT_FAILURE);
    }

    ensemble_model_.push(root, tree_weight, -1);
  }
}

Mart::~Mart() {
  // TODO: fix the destructor...
}

std::ostream &Mart::put(std::ostream &os) const {
  os << "# Ranker: " << name() << std::endl
     << "# max no. of trees = " << ntrees_ << std::endl
     << "# no. of tree leaves = " << nleaves_ << std::endl
     << "# shrinkage = " << shrinkage_ << std::endl
     << "# min leaf support = " << minleafsupport_ << std::endl;
  if (nthresholds_)
    os << "# no. of thresholds = " << nthresholds_ << std::endl;
  else
    os << "# no. of thresholds = unlimited" << std::endl;
  if (valid_iterations_)
    os << "# no. of no gain rounds before early stop = " << valid_iterations_
       << std::endl;
  return os;
}

void Mart::init(
    std::shared_ptr<quickrank::data::VerticalDataset> training_dataset) {

  const size_t nentries = training_dataset->num_instances();
  scores_on_training_ = new double[nentries]();  //0.0f initialized
  pseudoresponses_ = new double[nentries]();  //0.0f initialized
  const size_t nfeatures = training_dataset->num_features();
  sortedsid_ = new size_t * [nfeatures];
  sortedsize_ = nentries;

#pragma omp parallel for
  for (size_t i = 0; i < nfeatures; ++i)
    sortedsid_[i] = idx_radixsort(training_dataset->at(0, i),
                                  training_dataset->num_instances()).release();

  thresholds_ = new float *[nfeatures];
  thresholds_size_ = new size_t[nfeatures];

#pragma omp parallel for
  for (size_t i = 0; i < nfeatures; ++i) {
    //select feature array related to the current feature index
    float const *features = training_dataset->at(0, i);  // ->get_fvector(i);
    //init with values with the 1st sample
    size_t *idx = sortedsid_[i];
    //get_ sample indexes sorted by the fid-th feature
    size_t uniqs_size = 0;
    float *uniqs = (float *) malloc(
        sizeof(float)
            * (nthresholds_ == 0 ? sortedsize_ + 1 : nthresholds_ + 1));
    //skip samples with the same feature value. early stop for if nthresholds!=size_max
    uniqs[uniqs_size++] = features[idx[0]];
    for (size_t j = 1;
         j < sortedsize_
             && (nthresholds_ == 0 || uniqs_size != nthresholds_ + 1); ++j) {
      const float fval = features[idx[j]];
      if (uniqs[uniqs_size - 1] < fval)
        uniqs[uniqs_size++] = fval;
    }

    //define thresholds
    if (uniqs_size <= nthresholds_ || nthresholds_ == 0) {
      uniqs[uniqs_size++] = FLT_MAX;
      thresholds_size_[i] = uniqs_size, thresholds_[i] =
                                            (float *) realloc(uniqs,
                                                              sizeof(float)
                                                                  * uniqs_size);
    } else {
      free(uniqs);
      thresholds_size_[i] = nthresholds_ + 1;
      thresholds_[i] = (float *) malloc(sizeof(float) * (nthresholds_ + 1));
      float t = features[idx[0]];  //equals fmin
      const float step = fabs(features[idx[sortedsize_ - 1]] - t)
          / nthresholds_;  //(fmax-fmin)/nthresholds
      for (size_t j = 0; j != nthresholds_; t += step)
        thresholds_[i][j++] = t;
      thresholds_[i][nthresholds_] = FLT_MAX;
    }
  }

  // here, pseudo responses is empty !
  hist_ = new RTRootHistogram(training_dataset.get(), sortedsid_, sortedsize_,
                              thresholds_, thresholds_size_);
}

void Mart::clear(size_t num_features) {
  if (scores_on_training_)
    delete[] scores_on_training_;
  if (scores_on_validation_)
    delete[] scores_on_validation_;
  if (pseudoresponses_)
    delete[] pseudoresponses_;
  if (hist_)
    delete hist_;
  if (thresholds_size_)
    delete[] thresholds_size_;
  if (sortedsid_) {
    for (size_t i = 0; i < num_features; ++i) {
      delete[] sortedsid_[i];
      free(thresholds_[i]);
    }
    delete[] sortedsid_;
    delete[] thresholds_;
  }

  // Reset pointers to internal data structures
  scores_on_training_ = NULL;
  scores_on_validation_ = NULL;
  pseudoresponses_ = NULL;
  thresholds_size_ = NULL;
  sortedsid_ = NULL;
  thresholds_ = NULL;
  hist_ = NULL;
}

void Mart::learn(std::shared_ptr<quickrank::data::Dataset> training_dataset,
                 std::shared_ptr<quickrank::data::Dataset> validation_dataset,
                 std::shared_ptr<quickrank::metric::ir::Metric> scorer,
                 size_t partial_save, const std::string output_basename) {
  // ---------- Initialization ----------
  std::cout << "# Initialization";
  std::cout.flush();

  std::chrono::high_resolution_clock::time_point chrono_init_start =
      std::chrono::high_resolution_clock::now();

  // create a copy of the training datasets and put it in vertical format
  std::shared_ptr<quickrank::data::VerticalDataset> vertical_training(
      new quickrank::data::VerticalDataset(training_dataset));

  best_metric_on_validation_ = std::numeric_limits<double>::lowest();
  best_metric_on_training_ = std::numeric_limits<double>::lowest();
  best_model_ = 0;

  ensemble_model_.set_capacity(ntrees_);

  init(vertical_training);

  if (validation_dataset) {
    scores_on_validation_ = new Score[validation_dataset->num_instances()]();
  }

  // if the ensemble size is greater than zero, it means the learn method has
  // to start not from scratch but from a previously saved (intermediate) model
  if (ensemble_model_.is_notempty()) {
    best_model_ = ensemble_model_.get_size() - 1;

    // Update the model's outputs on all training samples
    score_dataset(training_dataset, scores_on_training_);
    // run metric
    best_metric_on_training_ = scorer->evaluate_dataset(
        vertical_training, scores_on_training_);

    if (validation_dataset) {
      // Update the model's outputs on all validation samples
      score_dataset(validation_dataset, scores_on_validation_);
      // run metric
      best_metric_on_validation_ = scorer->evaluate_dataset(
          validation_dataset, scores_on_validation_);
    }
  }

  auto chrono_init_end = std::chrono::high_resolution_clock::now();
  double init_time = std::chrono::duration_cast<std::chrono::duration<double>>(
      chrono_init_end - chrono_init_start).count();
  std::cout << ": " << std::setprecision(2) << init_time << " s." << std::endl;

  // ---------- Training ----------
  std::cout << std::fixed << std::setprecision(4);

  std::cout << "# Training:" << std::endl;
  std::cout << "# -------------------------" << std::endl;
  std::cout << "# iter. training validation" << std::endl;
  std::cout << "# -------------------------" << std::endl;

  // shows the performance of the already trained model..
  if (ensemble_model_.is_notempty()) {
    std::cout << std::setw(7) << ensemble_model_.get_size()
              << std::setw(9) << best_metric_on_training_;

    if (validation_dataset)
      std::cout << std::setw(9) << best_metric_on_validation_;

    std::cout << " *" << std::endl;
  }

  auto chrono_train_start = std::chrono::high_resolution_clock::now();

  // start iterations from 0 or (ensemble_size - 1)
  for (size_t m = ensemble_model_.get_size(); m < ntrees_; ++m) {
    if (validation_dataset
        && (valid_iterations_ && m > best_model_ + valid_iterations_))
      break;

    compute_pseudoresponses(vertical_training, scorer.get());

    // update the histogram with these training_setting labels
    // (the feature histogram will be used to find the best tree rtnode)
    hist_->update(pseudoresponses_, training_dataset->num_instances());

    //Fit a regression tree
    std::unique_ptr<RegressionTree>
        tree = fit_regressor_on_gradient(vertical_training);

    //add this tree to the ensemble (our model)
    ensemble_model_.push(tree->get_proot(), shrinkage_, 0);  // maxlabel);

    //Update the model's outputs on all training samples
    update_modelscores(vertical_training, scores_on_training_, tree.get());
    // run metric
    quickrank::MetricScore metric_on_training = scorer->evaluate_dataset(
        vertical_training, scores_on_training_);

    //show results
    std::cout << std::setw(7) << m + 1 << std::setw(9) << metric_on_training;

    //Evaluate the current model on the validation data (if available)
    if (validation_dataset) {
      // update validation scores
      update_modelscores(validation_dataset, scores_on_validation_, tree.get());

      // run metric
      quickrank::MetricScore metric_on_validation = scorer->evaluate_dataset(
          validation_dataset, scores_on_validation_);
      std::cout << std::setw(9) << metric_on_validation;

      if (metric_on_validation > best_metric_on_validation_) {
        best_metric_on_training_ = metric_on_training;
        best_metric_on_validation_ = metric_on_validation;
        best_model_ = ensemble_model_.get_size() - 1;
        std::cout << " *";
      }
    } else {
      if (metric_on_training > best_metric_on_training_) {
        best_metric_on_training_ = metric_on_training;
        best_model_ = ensemble_model_.get_size() - 1;
        std::cout << " *";
      }
    }
    std::cout << std::endl;

    if (partial_save != 0 and !output_basename.empty()
        and (m + 1) % partial_save == 0) {
      save(output_basename, m + 1);
    }

  }

  //Rollback to the best model observed on the validation data
  if (validation_dataset) {
    while (ensemble_model_.is_notempty()
        && ensemble_model_.get_size() > best_model_ + 1) {
      ensemble_model_.pop();
    }
  }

  auto chrono_train_end = std::chrono::high_resolution_clock::now();
  double train_time = std::chrono::duration_cast<std::chrono::duration<double>>(
      chrono_train_end - chrono_train_start).count();

  //Finishing up
  std::cout << std::endl;
  std::cout << *scorer << " on training data = " << best_metric_on_training_
            << std::endl;

  if (validation_dataset) {
    std::cout << *scorer << " on validation data = "
              << best_metric_on_validation_ << std::endl;
  }

  clear(vertical_training->num_features());

  std::cout << std::endl;
  std::cout << "#\t Training Time: " << std::setprecision(2) << train_time
            << " s." << std::endl;
}

void Mart::compute_pseudoresponses(
    std::shared_ptr<quickrank::data::VerticalDataset> training_dataset,
    quickrank::metric::ir::Metric *scorer) {
  const size_t nentries = training_dataset->num_instances();
  for (size_t i = 0; i < nentries; i++)
    pseudoresponses_[i] = training_dataset->getLabel(i)
        - scores_on_training_[i];
}

std::unique_ptr<RegressionTree> Mart::fit_regressor_on_gradient(
    std::shared_ptr<data::VerticalDataset> training_dataset) {
  //Fit a regression tree
  /// \todo TODO: memory management of regression tree is wrong!!!
  RegressionTree *tree = new RegressionTree(nleaves_, training_dataset.get(),
                                            pseudoresponses_, minleafsupport_);
  tree->fit(hist_);
  //update the outputs of the tree (with gamma computed using the Newton-Raphson pruning_method)
  //float maxlabel =
  tree->update_output(pseudoresponses_);
  return std::unique_ptr<RegressionTree>(tree);
}

void Mart::update_modelscores(std::shared_ptr<data::Dataset> dataset,
                              Score *scores, RegressionTree *tree) {
  const quickrank::Feature *d = dataset->at(0, 0);
  const size_t offset = 1;
  const size_t num_features = dataset->num_features();
  #pragma omp parallel for
  for (size_t i = 0; i < dataset->num_instances(); ++i) {
    scores[i] += shrinkage_ * tree->get_proot()->score_instance(
        d + i * num_features, offset);
  }
}

void Mart::update_modelscores(std::shared_ptr<data::VerticalDataset> dataset,
                              Score *scores, RegressionTree *tree) {

  const quickrank::Feature *d = dataset->at(0, 0);
  const size_t offset = dataset->num_instances();
  #pragma omp parallel for
  for (size_t i = 0; i < dataset->num_instances(); ++i) {
    scores[i] += shrinkage_ * tree->get_proot()->score_instance(
        d + i, offset);
  }
}

pugi::xml_document *Mart::get_xml_model() const {

  pugi::xml_document *doc = new pugi::xml_document();
  pugi::xml_node root = doc->append_child("ranker");
  pugi::xml_node info = root.append_child("info");

  info.append_child("type").text() = name().c_str();
  info.append_child("trees").text() = ntrees_;
  info.append_child("leaves").text() = nleaves_;
  info.append_child("shrinkage").text() = shrinkage_;
  info.append_child("leafsupport").text() = minleafsupport_;
  info.append_child("discretization").text() = nthresholds_;
  info.append_child("estop").text() = valid_iterations_;

  ensemble_model_.append_xml_model(root);

  return doc;
}

bool Mart::import_model_state(LTR_Algorithm &other) {

  // Check the object is derived from Mart
  try
  {
    Mart& otherCast = dynamic_cast<Mart&>(other);

    if (std::abs(shrinkage_ - otherCast.shrinkage_) > 0.000001 ||
        nthresholds_ != otherCast.nthresholds_ ||
        nleaves_ != otherCast.nleaves_ ||
        minleafsupport_ != otherCast.minleafsupport_ ||
        valid_iterations_ != otherCast.valid_iterations_)
      return false;

    // Move assignemnt operator
    // Move the ownership of the ensemble object to the current model
    ensemble_model_ = std::move(otherCast.ensemble_model_);
  }
  catch(std::bad_cast)
  {
    return false;
  }

  return true;
}

bool Mart::update_weights(std::vector<double>& weights) {
  return ensemble_model_.update_ensemble_weights(weights);
}

void Mart::print_additional_stats(void) const {
#ifdef QUICKRANK_PERF_STATS
  std::cout << "#" << std::endl;
  std::cout << "# Internal Nodes Traversed: " << RTNode::internal_nodes_traversed() << std::endl;
#endif
}

}  // namespace forests
}  // namespace learning
}  // namespace quickrank
