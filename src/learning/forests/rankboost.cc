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
 * Contributors:
 *   HPC. Laboratory - ISTI - CNR - http://hpc.isti.cnr.it/
 *   Tommaso Papini - https://github.com/oddlord
 *   Gabriele Bani - https://github.com/brnibani
 */
#include "learning/forests/rankboost.h"

#include <fstream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <sstream>
#include <string.h>

namespace quickrank {
namespace learning {
namespace forests {

const std::string Rankboost::NAME_ = "RANKBOOST";

Rankboost::Rankboost(size_t max_wr) {
  T = max_wr;
  best_T = 0;
  go_parallel = true;
  omp_schedule = "dynamic";
}

Rankboost::Rankboost(const pugi::xml_document &model) {

  pugi::xml_node model_info = model.child("ranker").child("info");
  pugi::xml_node model_wr = model.child("ranker").child("ensemble");

  //read (training) info
  T = model_info.child("maxweakrankers").text().as_uint();
  best_T = 0;

  // allocate weak rankers and their weights
  weak_rankers = new WeakRanker *[T];
  alphas = new float[T]();

  for (const auto &wr: model_wr.children()) {
    if (strcmp(wr.name(), "weakranker") == 0) {
      unsigned int id = wr.child("id").text().as_uint();
      unsigned int feature_id = wr.child("featureid").text().as_uint();
      Feature theta = wr.child("theta").text().as_float();
      int sign = wr.child("sign").text().as_uint();
      float alpha = wr.child("alpha").text().as_float();

      alphas[id] = alpha;
      weak_rankers[id] = new WeakRanker(feature_id, theta, sign);
      best_T++;
    }
  }
}

Rankboost::~Rankboost() {
  if (weak_rankers) {
    for (unsigned int t = 0; t < best_T; t++) {
      delete weak_rankers[t];
    }
    delete[] weak_rankers;
  }
  if (alphas) {
    delete[] alphas;
  }
}

struct external_sort_op_t {
  const Feature *values_;
  external_sort_op_t(const Feature *values) {
    values_ = values;
  }
  bool operator()(int i, int j) {
    return (values_[i] > values_[j]);
  }
};

std::ostream &Rankboost::put(std::ostream &os) const {
  os << "# Ranker: " << name() << std::endl;
  os << "# Weak Rankers: " << T << std::endl;
  return os;
}

void Rankboost::learn(
    std::shared_ptr<quickrank::data::Dataset> training_dataset,
    std::shared_ptr<quickrank::data::Dataset> validation_dataset,
    std::shared_ptr<quickrank::metric::ir::Metric> scorer,
    size_t partial_save, const std::string output_basename) {

  std::cout << std::endl << "# Rankboost running..." << std::endl;
  auto rank_start = std::chrono::high_resolution_clock::now();
  std::cout << "#" << std::endl;
  const char *on_off[2] = {"OFF", "ON"};
  std::cout << "# Parallel: " << on_off[go_parallel] << std::endl;

  // initialization
  init(training_dataset, validation_dataset);
  best_T = 0;
  MetricScore best_metric_on_training = 0;
  MetricScore best_metric_on_validation = 0;

  std::cout << "#" << std::endl;
  std::cout << "# Training started..." << std::endl;
  std::chrono::high_resolution_clock::time_point
      train_start = std::chrono::high_resolution_clock::now();
  std::cout << "#" << std::endl;
  char const *table_hline =
      "---------------------------------------------------------------------------------------------------------";
  std::cout << table_hline << std::endl;
  std::cout
      << "|  Weak  | Feature |   Threshold   |     R      |    alpha    |           "
      << *scorer << "           |   Time    |" << std::endl;
  std::cout
      << "| Ranker |    ID   |               |            |             | on training | on validation |           |"
      << std::endl;
  std::cout << table_hline << std::endl;

  // main loop (learning)
  for (unsigned int t = 0; t < T; t++) {

    printf("| %-7d", t + 1);
    std::chrono::high_resolution_clock::time_point
        train_t_start = std::chrono::high_resolution_clock::now();

    // compute potential
    compute_pi(training_dataset);

    // compute weak ranker
    WeakRanker *wr = compute_weak_ranker(training_dataset);

    // compute weak ranker's weight
    float alpha = 0;
    if (r_t >= 1) {
      alpha = max_alpha * r_t;
    } else {
      alpha = (float) (log((z_t + r_t) / (z_t - r_t)) / 2.0);
    }

    printf("| %-12.4g", alpha);

    if (alpha > max_alpha and r_t < 1) {
      max_alpha = alpha;
    }

    // store weak ranker and its weight
    weak_rankers[t] = wr;
    alphas[t] = alpha;

    for (unsigned int i = 0; i < training_dataset->num_instances(); i++) {
      training_scores[i] +=
          alpha * wr->score_document(training_dataset->at(i, 0));
    }

    MetricScore metric_on_training =
        scorer->evaluate_dataset(training_dataset, training_scores);
    printf("| %-12.4g", metric_on_training);

    if (validation_dataset) {
      for (unsigned int i = 0; i < validation_dataset->num_instances(); i++)
        validation_scores[i] +=
            alpha * wr->score_document(validation_dataset->at(i, 0));
      MetricScore metric_on_validation =
          scorer->evaluate_dataset(validation_dataset, validation_scores);
      printf("| %-13.4g", metric_on_validation);

      if (metric_on_validation > best_metric_on_validation) {
        best_metric_on_validation = metric_on_validation;
        best_metric_on_training = metric_on_training;
        best_T = t + 1;
        printf("*");
      } else
        printf(" ");
    } else {
      best_metric_on_training = metric_on_training;
      best_T = t + 1;
      printf("| ------        ");
    }

    // update document pair weights
    update_d(training_dataset, wr, alpha);

    auto train_t_end = std::chrono::high_resolution_clock::now();
    double train_t_time =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            train_t_end - train_t_start).count();

    printf("| %-5.3g s.%-2s|\n", train_t_time, "");

  } // main loop (learning)

  std::cout << table_hline << std::endl;

  // destroy temp objects
  clean(training_dataset);

  std::cout << "#" << std::endl;
  auto train_end = std::chrono::high_resolution_clock::now();
  double train_time = std::chrono::duration_cast<std::chrono::duration<double>>(
      train_end - train_start).count();
  std::cout << "# Training completed! (" << std::setprecision(3) << train_time
            << " s.)" << std::endl;

  // print metric on training/validation
  std::cout << "#" << std::endl;
  std::cout << std::setprecision(4) << "# " << *scorer << " on training: "
            << best_metric_on_training << std::endl;
  if (validation_dataset)
    std::cout << std::setprecision(4) << "# " << *scorer << " on validation: "
              << best_metric_on_validation << std::endl;


  std::cout << "#" << std::endl;
  auto rank_end = std::chrono::high_resolution_clock::now();
  double rank_time = std::chrono::duration_cast<std::chrono::duration<double>>(
      rank_end - rank_start).count();
  std::cout << "# Rankboost done! (" << std::setprecision(3) << rank_time
            << " s.)" << std::endl;
}

// Initialization

void Rankboost::init(std::shared_ptr<data::Dataset> training_dataset,
                     std::shared_ptr<data::Dataset> validation_dataset) {

  std::cout << "#" << std::endl;
  std::cout << "# Initializing...";
  auto init_start = std::chrono::high_resolution_clock::now();

  const unsigned int nq = training_dataset->num_queries();
  const unsigned int nf = training_dataset->num_features();
  const unsigned int ni = training_dataset->num_instances();

  training_scores = new Score[training_dataset->num_instances()]();
  if (validation_dataset)
    validation_scores = new Score[validation_dataset->num_instances()]();

  setenv("OMP_SCHEDULE", omp_schedule, 1);

  // allocate weak rankers and their weights
  weak_rankers = new WeakRanker *[T];
  alphas = new float[T]();

  // calculate N (number of correctly ranked document pairs)
  unsigned int N = 0;
#pragma omp parallel for if(go_parallel) schedule(runtime) reduction(+:N)
  for (unsigned int q = 0; q < nq; q++) {
    std::shared_ptr<data::QueryResults> qr =
        training_dataset->getQueryResults(q);
    const quickrank::Label *l = qr->labels();
    for (unsigned int i = 0; i < qr->num_results() - 1; i++)
      for (unsigned int j = i + 1; j < qr->num_results(); j++)
        if (l[j] > l[i])
          N++;
  }

  // allocate matrices D (document pair weights) and PI (potential)
  // D_ij = 1/N if j more relevant than i, 0 otherwise
  D = new float **[nq];
  PI = new float *[nq];
#pragma omp parallel for if(go_parallel) schedule(runtime)
  for (unsigned int q = 0; q < nq; q++) {
    std::shared_ptr<data::QueryResults>
        qr = training_dataset->getQueryResults(q);
    const quickrank::Label *l = qr->labels();
    D[q] = new float *[qr->num_results()];
    PI[q] = new float[qr->num_results()]();
    for (unsigned int i = 0; i < qr->num_results() - 1; i++) {
      D[q][i] = new float[qr->num_results()]();
      for (unsigned int j = i + 1; j < qr->num_results(); j++) {
        if (l[j] > l[i])
          D[q][i][j] = (float) (1.0 / N);
      }
    }
  }

  // allocate matrix THETA (feature thresholds) and calculates n_theta (number of thresholds for each feature)
  THETA = new Feature *[nf];
  n_theta = new unsigned int[nf]();
#pragma omp parallel for if(go_parallel) schedule(runtime)
  for (unsigned int f = 0; f < nf; f++) {
    Feature *tmp_theta = new Feature[ni];
    for (unsigned int k = 0; k < ni; k++)
      tmp_theta[k] = *training_dataset->at(k, f);
    std::sort(tmp_theta, tmp_theta + ni);
    std::reverse(tmp_theta, tmp_theta + ni);
    n_theta[f] = 1;
    for (unsigned int k = 1; k < ni; k++)
      if (tmp_theta[k] != tmp_theta[k - 1])
        tmp_theta[n_theta[f]++] = tmp_theta[k];
    THETA[f] = new Feature[n_theta[f]];
    for (unsigned int k = 0; k < n_theta[f]; k++)
      THETA[f][k] = tmp_theta[k];
    delete[] tmp_theta;
  }

  // Sorted Document Features
  SDF = new unsigned int **[training_dataset->num_features()];
#pragma omp parallel for if(go_parallel) schedule(runtime)
  for (unsigned int f = 0; f < nf; f++) {
    SDF[f] = new unsigned int *[training_dataset->num_queries()]();
    for (unsigned int q = 0; q < nq; q++) {
      std::shared_ptr<data::QueryResults>
          r = training_dataset->getQueryResults(q);
      unsigned int nr = r->num_results();
      SDF[f][q] = new unsigned int[nr]();
      Feature *feature_values = new Feature[nr];
      for (unsigned int i = 0; i < nr; i++) {
        feature_values[i] =
            *training_dataset->at(training_dataset->offset(q) + i, f);
      }
      unsigned int *sorted_indices = new unsigned int[nr];
      external_sort_op_t comp(feature_values);
      for (unsigned int i = 0; i < nr; ++i) {
        sorted_indices[i] = i;
      }
      std::sort(sorted_indices, sorted_indices + nr, comp);
      SDF[f][q] = sorted_indices;
      delete[] feature_values;
    }
  }

  auto init_end = std::chrono::high_resolution_clock::now();
  double init_time = std::chrono::duration_cast<std::chrono::duration<double>>(
      init_end - init_start).count();
  std::cout << " [Done] (" << std::setprecision(3) << init_time << " s.)"
            << std::endl;
} // init

// Compute potential matrix PI

void Rankboost::compute_pi(std::shared_ptr<data::Dataset> dataset) {

  for (unsigned int q = 0; q < dataset->num_queries(); q++) {
    std::shared_ptr<data::QueryResults> qr = dataset->getQueryResults(q);
    for (unsigned int i = 0; i < qr->num_results(); i++) {
      PI[q][i] = 0.0;
      for (unsigned int k = 0; k < i; k++)
        PI[q][i] += D[q][k][i];
      for (unsigned int k = i + 1; k < qr->num_results(); k++)
        PI[q][i] -= D[q][i][k];
    }
  }
}

// Compute a new weak ranker

WeakRanker *Rankboost::compute_weak_ranker(
    std::shared_ptr<data::Dataset> dataset) {

  best_r = 0.0;
  int best_feature_id = -1;
  Feature best_theta = -1;
  int sign = 1;
  const unsigned int nq = dataset->num_queries();

#pragma omp parallel for if(go_parallel) schedule(runtime) shared(best_feature_id, best_theta, sign)
  for (unsigned int f = 0; f < dataset->num_features(); f++) {
    unsigned int *last = new unsigned int[nq];
    for (unsigned int q = 0; q < nq; q++)
      last[q] = -1;
    float r = 0.0;
    for (unsigned int j = 0; j < n_theta[f]; j++) {
      Feature feat = THETA[f][j];
      for (unsigned int q = 0; q < nq; q++) {
        std::shared_ptr<data::QueryResults> qr = dataset->getQueryResults(q);
        for (unsigned int l = last[q] + 1; l < qr->num_results(); l++) {
          unsigned int id_doc = dataset->offset(q) + SDF[f][q][l];
          if (*dataset->at(id_doc, f) > feat) {
            r += PI[q][SDF[f][q][l]];
            last[q] = l;
          } else
            break;
        }
      }
      // sign = 1;
      // if (r < 0) {
      //     sign = -1;
      //     r = -r;
      // }
#pragma omp critical
      {
        if (r > best_r) {
          best_r = r;
          best_theta = feat;
          best_feature_id = f;
        }
      }
    }
    delete[] last;
  }

  r_t = z_t * best_r;

  printf("| %-8d| %-14.8g| %-11.7f", best_feature_id, best_theta, r_t);

  return new WeakRanker(best_feature_id, best_theta, sign);
} // compute_weak_ranker

// Update matrix D (document pair weights)

void Rankboost::update_d(std::shared_ptr<data::Dataset> dataset, WeakRanker *wr,
                         float alpha) {

  z_t = 0.0;
  const unsigned int nq = dataset->num_queries();

  // update matrix D
  for (unsigned int q = 0; q < nq; q++) {
    std::shared_ptr<data::QueryResults> qr = dataset->getQueryResults(q);
    for (unsigned int j = 0; j < qr->num_results() - 1; j++) {
      for (unsigned int k = j + 1; k < qr->num_results(); k++) {
        D[q][j][k] = (float) (D[q][j][k] * exp(alpha * (int) (
            wr->score_document(dataset->at(dataset->offset(q) + j, 0))
                - wr->score_document(dataset->at(dataset->offset(q) + k, 0)))));
        z_t += D[q][j][k];
      }
    }
  }

  // normalize
  for (unsigned int q = 0; q < nq; q++) {
    std::shared_ptr<data::QueryResults> qr = dataset->getQueryResults(q);
    for (unsigned int j = 0; j < qr->num_results() - 1; j++)
      for (unsigned int k = j + 1; k < qr->num_results(); k++)
        D[q][j][k] /= z_t;
  }
} // update_d

// Clean up some space

void Rankboost::clean(std::shared_ptr<data::Dataset> dataset) {

  std::cout << "#" << std::endl;
  std::cout << "# Cleaning...";
  auto clean_start = std::chrono::high_resolution_clock::now();

/*    if (weak_rankers) {
        for (unsigned int t = best_T; t < T; t++)
            delete weak_rankers[t];
    }
*/
  const unsigned int nq = dataset->num_queries();
  const unsigned int nf = dataset->num_features();

  if (D) {
#pragma omp parallel for if(go_parallel) schedule(runtime)
    for (unsigned int q = 0; q < nq; q++) {
      std::shared_ptr<data::QueryResults> qr = dataset->getQueryResults(q);
      for (unsigned int i = 0; i < qr->num_results() - 1; i++)
        delete[] D[q][i];
      delete[] D[q];
    }
    delete[] D;
  }

  if (PI) {
#pragma omp parallel for if(go_parallel) schedule(runtime)
    for (unsigned int q = 0; q < nq; q++)
      delete[] PI[q];
    delete[] PI;
  }

  if (THETA) {
#pragma omp parallel for if(go_parallel) schedule(runtime)
    for (unsigned int f = 0; f < nf; f++)
      delete[] THETA[f];
    delete[] THETA;
  }

  if (SDF) {
#pragma omp parallel for if(go_parallel) schedule(runtime)
    for (unsigned int f = 0; f < nf; f++) {
      for (unsigned int q = 0; q < nq; q++)
        delete[] SDF[f][q];
      delete[] SDF[f];
    }
    delete[] SDF;
  }

  if (n_theta) {
    delete[] n_theta;
  }

  if (training_scores) {
    delete[] training_scores;
  }

  if (validation_scores) {
    delete[] validation_scores;
  }

  std::chrono::high_resolution_clock::time_point clean_end =
      std::chrono::high_resolution_clock::now();
  double clean_time =
      std::chrono::duration_cast<std::chrono::duration<double>>(
          clean_end - clean_start).count();
  std::cout << " [Done] (" << std::setprecision(5) << clean_time << " s.)"
            << std::endl;
}


Score Rankboost::score_document(const quickrank::Feature *d) const {

  Score doc_score = 0.0;
  for (unsigned int t = 0; t < best_T; t++) {
    doc_score += alphas[t] * weak_rankers[t]->score_document(d);
  }
  return doc_score;
}

std::shared_ptr<std::vector<Score>> Rankboost::partial_scores_document(
    const Feature *d, bool ignore_weights) const {
  std::vector<quickrank::Score> scores(best_T);
  for (unsigned int t = 0; t < best_T; t++) {
    scores[t] = weak_rankers[t]->score_document(d);
    if (!ignore_weights)
      scores[t] *= alphas[t];
  }
  return std::make_shared<std::vector<quickrank::Score>>(std::move(scores));
}

pugi::xml_document *Rankboost::get_xml_model() const {

  pugi::xml_document *doc = new pugi::xml_document();
  pugi::xml_node root = doc->append_child("ranker");

  pugi::xml_node info = root.append_child("info");

  info.append_child("type").text() = name().c_str();
  info.append_child("maxweakrankers").text() = T;

  pugi::xml_node ensemble = root.append_child("ensemble");
  for (unsigned int t = 0; t < best_T; t++) {

    pugi::xml_node wr = ensemble.append_child("weakranker");
    wr.append_child("id").text() = t;
    wr.append_child("featureid").text() = weak_rankers[t]->get_feature_id();
    wr.append_child("theta").text() = weak_rankers[t]->get_theta();
    wr.append_child("sign").text() = weak_rankers[t]->get_sign();
    wr.append_child("alpha").text() = alphas[t];
  }

  return doc;
}

bool Rankboost::update_weights(std::vector<double> &weights) {

  if (weights.size() != best_T) {
    std::cerr << "# ## ERROR!! Weak ranker size does not match size of the "
        "weight vector in updating the weights" << std::endl;
    return false;
  }

  for (unsigned int t = 0; t < best_T; t++)
    alphas[t] = weights[t];

  return true;
}

std::vector<double> Rankboost::get_weights() const {
  std::vector<double> weights(best_T);
  for (unsigned int i = 0; i < best_T; ++i)
    weights[i] = weights[i];
  return weights;
}

} // namespace forests
} // namespace learning
} // namespace quickrank
