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
#include "learning/meta/meta_cleaver.h"
#include "learning/forests/mart.h"
#include "learning/ltr_algorithm.h"
#include "optimization/post_learning/cleaver/cleaver_factory.h"
#include "driver/driver.h"


#include <memory>
#include <fstream>
#include <iomanip>
#include <chrono>

#include "utils/radix.h"

namespace quickrank {
namespace learning {
namespace meta {

const std::string MetaCleaver::NAME_ = "METACLEAVER";

MetaCleaver::MetaCleaver(const pugi::xml_document& model) {
  ntrees_ = 0;
  ntrees_per_iter_ = 0;
  pruning_rate_per_iter_ = 0;
  opt_last_only_ = false;

  pugi::xml_node info = model.child("ranker").child("info");

  // read (training) info
  ntrees_ = info.child("trees").text().as_int();
  ntrees_per_iter_ = info.child("trees-per-iter").text().as_int();
  pruning_rate_per_iter_ =
      info.child("pruning-rate-per-iter").text().as_double();
  opt_last_only_ = info.child("opt-last-only").text().as_bool();
  valid_iterations_ = info.child("meta-end-after-rounds").text().as_int();

  model.child("ranker").remove_child("info");

  model.child("ranker").child("ltr-model").set_name("info");
  ltr_algo_ = quickrank::learning::LTR_Algorithm::load_model_from_xml(model);
  model.child("ranker").child("info").set_name("ltr-model");

  model.child("ranker").child("cleaver").set_name("info");
  model.child("ranker").set_name("optimizer");
  cleaver_ =
      std::dynamic_pointer_cast<optimization::post_learning::pruning::Cleaver>(
          optimization::post_learning::pruning::create_pruner(model));
  model.child("optimizer").set_name("ranker");
  model.child("ranker").child("info").set_name("cleaver");

  model.child("ranker").append_copy(info);
}

pugi::xml_document* MetaCleaver::get_xml_model() const {

  pugi::xml_document *doc = new pugi::xml_document();
  pugi::xml_node root = doc->append_child("ranker");

  pugi::xml_node info = root.append_child("info");
  info.append_child("type").text() = name().c_str();
  info.append_child("trees-per-iter").text() = ntrees_per_iter_;
  info.append_child("pruning-rate-per-iter").text() = pruning_rate_per_iter_;
  info.append_child("opt-last-only").text() = opt_last_only_;
  info.append_child("meta-end-after-rounds").text() = valid_iterations_;

  pugi::xml_document& ltr_model = *ltr_algo_->get_xml_model();
  pugi::xml_node ltr_info = ltr_model.child("ranker").child("info");
  ltr_info.set_name("ltr-model");
  root.append_copy(ltr_info);

  pugi::xml_document& cleaver_model = *cleaver_->get_xml_model();
  pugi::xml_node opt_info = cleaver_model.child("optimizer").child("info");

  opt_info.set_name("cleaver");
  root.append_copy(opt_info);

  pugi::xml_node ls_info = cleaver_model.child("optimizer").child("line-search");
  root.append_copy(ls_info);

  pugi::xml_node ensemble = ltr_model.child("ranker").child("ensemble");
  root.append_copy(ensemble);

  return doc;
}

std::ostream &MetaCleaver::put(std::ostream &os) const {
  os << "# Meta Ranker: " << name() << std::endl
     << "# max no. of trees = " << ntrees_ << std::endl
     << "# no. of trees per iter = " << ntrees_per_iter_ << std::endl
     << "# pruning rate per iter = " << cleaver_->get_pruning_rate() <<std::endl
     << "# optimize only last trees = " << opt_last_only_ <<std::endl
     << "# no. of no gain rounds before early stop = " << valid_iterations_
     << std::endl << std::endl << *ltr_algo_ << std::endl << *cleaver_;
  return os;
}

void MetaCleaver::learn(std::shared_ptr<quickrank::data::Dataset> training_dataset,
                 std::shared_ptr<quickrank::data::Dataset> validation_dataset,
                 std::shared_ptr<quickrank::metric::ir::Metric> scorer,
                 size_t partial_save, const std::string output_basename) {

  // Do the cast in order to access to the ensemble...
  auto ltr_algo_ensemble =
      std::dynamic_pointer_cast<quickrank::learning::forests::Mart>(ltr_algo_);

  auto chrono_train_start = std::chrono::high_resolution_clock::now();

  if (!verbose_) {
    std::cout << "# Training:" << std::endl;
    std::cout << "# -------------------------------" << std::endl;
    std::cout << "# iter. trees training validation" << std::endl;
    std::cout << "# -------------------------------" << std::endl;
  }

  quickrank::MetricScore best_metric_on_training =
      std::numeric_limits<double>::lowest();
  quickrank::MetricScore best_metric_on_validation =
      std::numeric_limits<double>::lowest();

  size_t best_model = ltr_algo_ensemble->ensemble_model_.get_size();
  std::vector<double> best_weights;
  best_weights.reserve(ntrees_);
  if (best_model > 0)
    best_weights = ltr_algo_ensemble->get_weights();

  // if we optimize the full model at each iteration (not only the last part)
  // we cannot do more than one iteration without improvement...
  if (!opt_last_only_)
    valid_iterations_ = 1;

  // Let's take the control of updating the LtR model
  cleaver_->set_update_model(false);

  size_t last_ensemble_size;
  unsigned int iter = 0;
  unsigned int best_iter = 0;
  size_t last_save = best_model;
  do {

    ++iter;
    if (valid_iterations_ && iter > best_iter + valid_iterations_)
      break;

    // Suppress output from cleaver and line_search. Print only summary of iter
    if (!verbose_)
      std::cout.setstate(std::ios_base::failbit);

      // Record the ensemble size before doing this iteration
    last_ensemble_size = ltr_algo_ensemble->ensemble_model_.get_size();

    // update ensemble based model to train additioanl ntrees_per_iter_ trees
    ltr_algo_ensemble->ntrees_ = last_ensemble_size + ntrees_per_iter_;

    ltr_algo_ensemble->learn(training_dataset,
                             nullptr, //validation_dataset,
                             scorer,
                             0,
                             output_basename);

    print_weights(ltr_algo_ensemble->get_weights(), "LtR Weights post-train");

    size_t new_ensemble_size = ltr_algo_ensemble->ensemble_model_.get_size();
    size_t diff_ensemble_size = new_ensemble_size - last_ensemble_size;

    size_t trees_to_keep;
    if (pruning_rate_per_iter_ < 1)
      trees_to_keep =
          (size_t) round( (1. - pruning_rate_per_iter_) * ntrees_per_iter_);
    else {
      trees_to_keep = ntrees_per_iter_ - pruning_rate_per_iter_;
      if (trees_to_keep < 0) {
        std::cerr << "Incorrect pruning rate value (too high). Quit!"
                  << std::endl;
        exit(EXIT_FAILURE);
      }
    }

    size_t trees_to_prune = diff_ensemble_size - trees_to_keep;
    if (new_ensemble_size - trees_to_prune > ntrees_)
      trees_to_prune = new_ensemble_size - ntrees_;

    // If the LtR training do not learned additional trees, stop now.
    if (!diff_ensemble_size)
      break;

    // extract the partial version of the training and validation datasets
    std::shared_ptr<quickrank::data::Dataset> training_partial_dataset =
      quickrank::driver::Driver::extract_partial_scores(
          ltr_algo_ensemble, training_dataset, true);

    std::shared_ptr<quickrank::data::Dataset>
        validation_partial_dataset = nullptr;
    if (validation_dataset) {
      validation_partial_dataset =
          quickrank::driver::Driver::extract_partial_scores(
              ltr_algo_ensemble, validation_dataset, true);
    }

    // Set the number of trees to prune (when pruning rate is >= 1)
    cleaver_->set_pruning_rate(trees_to_prune);
    // Update the weights used by cleaver, using the weights of the ltr model
    auto ltr_weights = ltr_algo_ensemble->get_weights();
    cleaver_->update_weights(ltr_weights);

    // Set the optimization process to run only on the last trees
    if (opt_last_only_)
      cleaver_->set_last_estimators_to_optimize(diff_ensemble_size);

    if (cleaver_->get_line_search()) {
      // Reset the weights of the line search model in order to force it to
      // not reuse the learned weights from the previous iteration but use the
      // current weights of the cleaver model
      cleaver_->get_line_search()->reset_weights();
    }

    // run the optimization process
    cleaver_->optimize(
        ltr_algo_ensemble,
        training_partial_dataset,
        validation_partial_dataset,
        scorer,
        0,
        output_basename);

    // Check if there is a metric score improvement
    bool improvement = false;
    if (validation_dataset) {
      if (cleaver_->get_metric_on_validation() > best_metric_on_validation) {
        best_metric_on_validation = cleaver_->get_metric_on_validation();
        best_metric_on_training = cleaver_->get_metric_on_training();
        improvement = true;
      }
    } else if (cleaver_->get_metric_on_training() > best_metric_on_training) {
      best_metric_on_training = cleaver_->get_metric_on_training();
      improvement = true;
    }

    if (verbose_) {
      std::cout << std::fixed << std::setprecision(4) << std::endl;
      std::cout << "metric on training: "
                << cleaver_->get_metric_on_training()
                << " ( " << best_metric_on_training << " )" << std::endl;
      std::cout << "metric on validation: "
                << cleaver_->get_metric_on_validation() <<
                " ( " << best_metric_on_validation << " )" << std::endl;
      std::cout << "improvement: " << improvement << std::endl;
    }


    auto cur_weights = cleaver_->get_weigths();
    // Apply changes on LtR algo if there is an impr. OR if we can backtrack
    if (improvement || opt_last_only_) {
      bool res = ltr_algo_ensemble->update_weights(cur_weights);
      if (!res)
        std::exit(EXIT_FAILURE);
    }

    // If there is no improvement and we cannot backtrack, stop it now!
    if (!improvement && !opt_last_only_) {
      break;
    }

    // Save partial infos to do the backtrack
    size_t cur_ens_size = ltr_algo_ensemble->ensemble_model_.get_size();
    if (improvement) {
      best_model = cur_ens_size;
      best_weights = ltr_algo_ensemble->get_weights();
      best_iter = iter;
    }

    print_weights(cleaver_->get_weigths(), "Cleaver Weights post-optimization");
    print_weights(ltr_algo_ensemble->get_weights(), "LtR Weights "
        "post-optimization");

    // check if we have to print only the summary of each iteration
    if (!verbose_) {
      // Reset the stream state to print again
      std::cout.clear();

      std::cout << std::fixed << std::setprecision(4);

      // shows the performance of the already trained model..
      std::cout
          << std::setw(7) << iter
          << std::setw(6) << ltr_algo_ensemble->ensemble_model_.get_size()
          << std::setw(9) << cleaver_->get_metric_on_training();

      if (validation_dataset)
        std::cout << std::setw(11)
                  << cleaver_->get_metric_on_validation();

      std::cout << std::endl;
    } else {

      std::cout << std::endl
                << "# ---------------------------------------------"<< std::endl
                << "# |        Completed Meta Iteration. " << iter
                << "        |" << std::endl
                << "# ---------------------------------------------"
                << std::endl << std::endl;
    }

    if (partial_save != 0 and !output_basename.empty()) {

      if ( floor( 1.0f * last_save / partial_save) <
           floor( 1.0f * cur_ens_size / partial_save) ) {

        save(output_basename, cur_ens_size);
        last_save = cur_ens_size;
      }
    }

  } while (ltr_algo_ensemble->ensemble_model_.get_size() < ntrees_);

  // Check if we need to the the backtrack on the LtR model
  size_t cur_size = ltr_algo_ensemble->ensemble_model_.get_size();
  if (cur_size > best_model) {
    std::vector<double> weight_mask = best_weights;
    // Add as many 0-weight elem as the trees added from the last best model
    for (size_t i=0; i < cur_size - best_model; ++i)
      weight_mask.push_back(0);
    // Reset the model with the old best weight and last trees
    bool res = ltr_algo_ensemble->update_weights(weight_mask);
    if (!res)
      std::exit(EXIT_FAILURE);
  }

  // Reset the stream state to print again
  std::cout.clear();

  auto chrono_train_end = std::chrono::high_resolution_clock::now();
  double train_time = std::chrono::duration_cast<std::chrono::duration<double>>(
      chrono_train_end - chrono_train_start).count();

  //Finishing up
  std::cout <<  std::endl;
  std::cout << "Final ensemble size = "
            << ltr_algo_ensemble->ensemble_model_.get_size() << std::endl;

  std::cout <<  std::fixed << std::setprecision(4);
  std::cout << *scorer << " on training data = "
            << best_metric_on_training << std::endl;

  if (validation_dataset) {
    std::cout << *scorer << " on validation data = "
              << best_metric_on_validation << std::endl;
  }

  std::cout << std::endl;
  std::cout << "#\t Training Time: " << std::setprecision(2) << train_time
            << " s." << std::endl;
}

bool MetaCleaver::import_model_state(LTR_Algorithm &other) {

  // Check the object is derived from MetaCleaver
  try
  {
    MetaCleaver& otherCast = dynamic_cast<MetaCleaver&>(other);

    // TODO: improve this check...
    if (ntrees_per_iter_ != otherCast.ntrees_per_iter_ ||
        pruning_rate_per_iter_ != otherCast.pruning_rate_per_iter_ ||
        opt_last_only_ != otherCast.opt_last_only_)
      return false;

    if (!ltr_algo_->import_model_state(*otherCast.ltr_algo_))
      return false;

  }
  catch(std::bad_cast)
  {
    return false;
  }

  return true;
}

}  // namespace forests
}  // namespace learning
}  // namespace quickrank
