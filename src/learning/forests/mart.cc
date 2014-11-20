#include "learning/forests/mart.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cfloat>
#include <cmath>
#include <chrono>

#include "utils/radix.h"
#include "utils/qsort.h"
#include "utils/mergesorter.h"
#include "data/rankedresults.h"

namespace quickrank {
namespace learning {
namespace forests {

std::ostream& Mart::put(std::ostream& os) const {
  os  << "# Ranker: MART" << std::endl
      << "# max no. of trees = " << ntrees_ << std::endl
      << "# no. of tree leaves = " << nleaves_ << std::endl
      << "# shrinkage = " << shrinkage_ << std::endl
      << "# min leaf support = " << minleafsupport_ << std::endl;
  if (nthresholds_)
    os << "# no. of thresholds = " << nthresholds_ << std::endl;
  else
    os << "# no. of thresholds = unlimited" << std::endl;
  if (valid_iterations_)
    os << "# no. of no gain rounds before early stop = " << valid_iterations_ << std::endl;
  return os;
}

void Mart::init(std::shared_ptr<quickrank::data::Dataset> training_dataset,
                std::shared_ptr<quickrank::data::Dataset> validation_dataset) {
  // make sure dataset is vertical
  preprocess_dataset(training_dataset);

  const unsigned int nentries = training_dataset->num_instances();
  scores_on_training_ = new double[nentries]();  //0.0f initialized
  pseudoresponses_ = new double[nentries]();  //0.0f initialized
  const unsigned int nfeatures = training_dataset->num_features();
  sortedsid_ = new unsigned int*[nfeatures];
  sortedsize_ = nentries;
#pragma omp parallel for
  for (unsigned int i = 0; i < nfeatures; ++i)
    sortedsid_[i] = idx_radixsort(training_dataset->at(0, i),
                                 training_dataset->num_instances()).release();
  // for(unsigned int i=0; i<nfeatures; ++i)
  //    training_set->sort_dpbyfeature(i, sortedsid[i], sortedsize);
  //for each featureid, init threshold array by keeping track of the list of "unique values" and their max, min
  thresholds_ = new float*[nfeatures];
  thresholds_size_ = new unsigned int[nfeatures];
#pragma omp parallel for
  for (unsigned int i = 0; i < nfeatures; ++i) {
    //select feature array realted to the current feature index
    float const* features = training_dataset->at(0, i);  // ->get_fvector(i);
    //init with values with the 1st sample
    unsigned int *idx = sortedsid_[i];
    //get_ sample indexes sorted by the fid-th feature
    unsigned int uniqs_size = 0;
    float *uniqs = (float*) malloc(
        sizeof(float) * (nthresholds_ == 0 ? sortedsize_ + 1 : nthresholds_ + 1));
    //skip samples with the same feature value. early stop for if nthresholds!=size_max
    uniqs[uniqs_size++] = features[idx[0]];
    for (unsigned int j = 1;
        j < sortedsize_ && (nthresholds_ == 0 || uniqs_size != nthresholds_ + 1);
        ++j) {
      const float fval = features[idx[j]];
      if (uniqs[uniqs_size - 1] < fval)
        uniqs[uniqs_size++] = fval;
    }
    //define thresholds
    if (uniqs_size <= nthresholds_ || nthresholds_ == 0) {
      uniqs[uniqs_size++] = FLT_MAX;
      thresholds_size_[i] = uniqs_size, thresholds_[i] = (float*) realloc(
          uniqs, sizeof(float) * uniqs_size);
    } else {
      free(uniqs), thresholds_size_[i] = nthresholds_ + 1, thresholds_[i] =
          (float*) malloc(sizeof(float) * (nthresholds_ + 1));
      float t = features[idx[0]];  //equals fmin
      const float step = fabs(features[idx[sortedsize_ - 1]] - t) / nthresholds_;  //(fmax-fmin)/nthresholds
      for (unsigned int j = 0; j != nthresholds_; t += step)
        thresholds_[i][j++] = t;
      thresholds_[i][nthresholds_] = FLT_MAX;
    }
  }
  if (validation_dataset) {
    preprocess_dataset(validation_dataset);
    scores_on_validation_ = new Score[validation_dataset->num_instances()]();
  }
  hist_ = new RTRootHistogram(training_dataset.get(), pseudoresponses_, sortedsid_,
                             sortedsize_, thresholds_, thresholds_size_);
}

void Mart::clear(std::shared_ptr<data::Dataset> training_dataset){
  if (scores_on_training_) delete [] scores_on_training_;
  if (scores_on_validation_) delete [] scores_on_validation_;
  if (pseudoresponses_) delete [] pseudoresponses_;
  if (hist_) delete hist_;
  if (thresholds_size_) delete [] thresholds_size_;
  if (sortedsid_) {
    for (unsigned int i = 0; i < training_dataset->num_features(); ++i) {
      delete[] sortedsid_[i];
      free(thresholds_[i]);
    }
  }
}

void Mart::preprocess_dataset(std::shared_ptr<data::Dataset> dataset) const {
  if (dataset->format() != data::Dataset::VERT)
    dataset->transpose();
}


void Mart::learn(std::shared_ptr<quickrank::data::Dataset> training_dataset,
                 std::shared_ptr<quickrank::data::Dataset> validation_dataset,
                 std::shared_ptr<quickrank::metric::ir::Metric> scorer, unsigned int partial_save,
                 const std::string output_basename) {
  // ---------- Initialization ----------
  std::cout << "# Initialization";
  std::cout.flush();

  std::chrono::high_resolution_clock::time_point chrono_init_start =
      std::chrono::high_resolution_clock::now();

  init(training_dataset, validation_dataset);

  std::chrono::high_resolution_clock::time_point chrono_init_end =
      std::chrono::high_resolution_clock::now();
  double init_time = std::chrono::duration_cast<std::chrono::duration<double>>(
      chrono_init_end - chrono_init_start).count();
  std::cout << ": " << std::setprecision(2) << init_time << " s." << std::endl;


  // ---------- Training ----------
  std::cout << std::fixed << std::setprecision(4);

  std::cout << "# Training:" << std::endl;
  std::cout << "# -------------------------" << std::endl;
  std::cout << "# iter. training validation" << std::endl;
  std::cout << "# -------------------------" << std::endl;

  std::chrono::high_resolution_clock::time_point chrono_train_start =
      std::chrono::high_resolution_clock::now();

  quickrank::MetricScore best_metric_on_validation = 0.0;
  //set max capacity of the ensamble
  ensemble_model_.set_capacity(ntrees_);
  //start iterations
  for (unsigned int m = 0;
      m < ntrees_ && (valid_iterations_ == 0 || m <= validation_bestmodel_ + valid_iterations_); ++m) {
    compute_pseudoresponses(training_dataset, scorer.get());

    //update the histogram with these training_seting labels (the feature histogram will be used to find the best tree rtnode)
    hist_->update(pseudoresponses_, training_dataset->num_instances());

    //Fit a regression tree
    std::unique_ptr<RegressionTree> tree = fit_regressor_on_gradient ( training_dataset );

    //add this tree to the ensemble (our model)
    ensemble_model_.push(tree->get_proot(), shrinkage_, 0 ); // maxlabel);

    //Update the model's outputs on all training samples
    update_modelscores(training_dataset, scores_on_training_, tree.get());
    // run metric
    quickrank::MetricScore metric_on_training = scorer->evaluate_dataset(
        training_dataset, scores_on_training_);

    //show results
    std::cout << std::setw(7) << m+1 << std::setw(9) << metric_on_training;
    //Evaluate the current model on the validation data (if available)
    if (validation_dataset) {
      // update validation scores
      update_modelscores(validation_dataset, scores_on_validation_, tree.get());

      // run metric
      quickrank::MetricScore metric_on_validation = scorer->evaluate_dataset(
          validation_dataset, scores_on_validation_);
      std::cout << std::setw(9) << metric_on_validation;

      if (metric_on_validation > best_metric_on_validation
          || best_metric_on_validation == 0.0f) {
        best_metric_on_validation = metric_on_validation;
        validation_bestmodel_ = ensemble_model_.get_size() - 1;
        std::cout <<" *";
      }
    }
    std::cout << std::endl;

    if (partial_save != 0 and !output_basename.empty()
        and (m + 1) % partial_save == 0) {
      save(output_basename, m + 1);
    }

  }
  //Rollback to the best model observed on the validation data
  if (validation_dataset)
    while (ensemble_model_.is_notempty() && ensemble_model_.get_size() > validation_bestmodel_ + 1)
      ensemble_model_.pop();

  std::chrono::high_resolution_clock::time_point chrono_train_end =
      std::chrono::high_resolution_clock::now();
  double train_time = std::chrono::duration_cast<std::chrono::duration<double>>(
      chrono_train_end - chrono_train_start).count();

  //Finishing up
  score_dataset(training_dataset, scores_on_training_);
  quickrank::MetricScore metric_on_training = scorer->evaluate_dataset(
      training_dataset, scores_on_training_);

  std::cout << std::endl;
  std::cout << *scorer << " on training data = " << metric_on_training
      << std::endl;
  if (validation_dataset) {
    score_dataset(validation_dataset, scores_on_validation_);
    best_metric_on_validation = scorer->evaluate_dataset(validation_dataset,
                                                         scores_on_validation_);
    std::cout << *scorer << " on validation data = "
        << best_metric_on_validation << std::endl;
  }

  clear(training_dataset);

  std::cout << std::endl;
  std::cout << "#\t Training Time: " << std::setprecision(2) << train_time << " s." << std::endl;
}


void Mart::compute_pseudoresponses( std::shared_ptr<quickrank::data::Dataset> training_dataset,
                                    quickrank::metric::ir::Metric* scorer) {
  const unsigned int nentries = training_dataset->num_instances();
  for(unsigned int i=0; i<nentries; i++)
    pseudoresponses_[i] = training_dataset->getLabel(i) - scores_on_training_[i];
}

std::unique_ptr<RegressionTree> Mart::fit_regressor_on_gradient (
    std::shared_ptr<data::Dataset> training_dataset ) {
  //Fit a regression tree
  /// \todo TODO: memory management of regression tree is wrong!!!
  RegressionTree* tree = new RegressionTree ( nleaves_, training_dataset.get(),
                                              pseudoresponses_, minleafsupport_);
  tree->fit(hist_);
  //update the outputs of the tree (with gamma computed using the Newton-Raphson method)
  //float maxlabel =
  tree->update_output(pseudoresponses_);
  return std::unique_ptr<RegressionTree>(tree);

}

void Mart::update_modelscores(std::shared_ptr<data::Dataset> dataset, Score *scores,
                              RegressionTree* tree) {
  quickrank::Score* score_i = scores;
  for (unsigned int q = 0; q < dataset->num_queries(); q++) {
    std::shared_ptr<quickrank::data::QueryResults> results = dataset->getQueryResults(q);
    const unsigned int offset = dataset->num_instances();
    const Feature* d = results->features();
    for (unsigned int i = 0; i < results->num_results(); i++) {
      score_i[i] += shrinkage_ * tree->get_proot()->score_instance(d, offset);
      d++;
    }
    score_i += results->num_results();
  }
}

std::ofstream& Mart::save_model_to_file(std::ofstream& os) const {
  // write ranker description
  os << *this;
  // save xml model
  ensemble_model_.save_model_to_file(os);
  return os;
}

}  // namespace forests
}  // namespace learning
}  // namespace quickrank
