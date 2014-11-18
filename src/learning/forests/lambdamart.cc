#include "learning/forests/lambdamart.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cfloat>
#include <cmath>

#include "utils/radix.h"
#include "utils/qsort.h"
#include "utils/mergesorter.h"
#include "data/rankedresults.h"

namespace quickrank {
namespace learning {
namespace forests {

std::ostream& LambdaMart::put(std::ostream& os) const {
  os << "# Ranker: Lambda-MART" << std::endl << "# max no. of trees = "
      << ntrees << std::endl << "# no. of tree leaves = " << ntreeleaves
      << std::endl << "# shrinkage = " << shrinkage << std::endl
      << "# min leaf support = " << minleafsupport << std::endl;
  if (nthresholds)
    os << "# no. of thresholds = " << nthresholds << std::endl;
  else
    os << "# no. of thresholds = unlimited" << std::endl;
  if (esr)
    os << "# no. of no gain rounds before early stop = " << esr << std::endl;
  return os;
}

void LambdaMart::preprocess_dataset(std::shared_ptr<data::Dataset> dataset) const {
  if (dataset->format() != data::Dataset::VERT)
    dataset->transpose();
}

void LambdaMart::init(std::shared_ptr<quickrank::data::Dataset> training_dataset,
                      std::shared_ptr<quickrank::data::Dataset> validation_dataset) {
  printf("Initialization:\n");
#ifdef SHOWTIMER
  double timer = omp_get_wtime();
#endif

  // make sure dataset is vertical
  preprocess_dataset(training_dataset);

  const unsigned int nentries = training_dataset->num_instances();
  trainingmodelscores = new double[nentries]();  //0.0f initialized
  pseudoresponses = new double[nentries]();  //0.0f initialized
  cachedweights = new double[nentries]();  //0.0f initialized
  const unsigned int nfeatures = training_dataset->num_features();
  sortedsid = new unsigned int*[nfeatures];
  sortedsize = nentries;
#pragma omp parallel for
  for (unsigned int i = 0; i < nfeatures; ++i)
    sortedsid[i] = idx_radixsort(training_dataset->at(0, i),
                                 training_dataset->num_instances()).release();
  // for(unsigned int i=0; i<nfeatures; ++i)
  //    training_set->sort_dpbyfeature(i, sortedsid[i], sortedsize);
  //for each featureid, init threshold array by keeping track of the list of "unique values" and their max, min
  thresholds = new float*[nfeatures];
  thresholds_size = new unsigned int[nfeatures];
#pragma omp parallel for
  for (unsigned int i = 0; i < nfeatures; ++i) {
    //select feature array realted to the current feature index
    float const* features = training_dataset->at(0, i);  // ->get_fvector(i);
    //init with values with the 1st sample
    unsigned int *idx = sortedsid[i];
    //get_ sample indexes sorted by the fid-th feature
    unsigned int uniqs_size = 0;
    float *uniqs = (float*) malloc(
        sizeof(float) * (nthresholds == 0 ? sortedsize + 1 : nthresholds + 1));
    //skip samples with the same feature value. early stop for if nthresholds!=size_max
    uniqs[uniqs_size++] = features[idx[0]];
    for (unsigned int j = 1;
        j < sortedsize && (nthresholds == 0 || uniqs_size != nthresholds + 1);
        ++j) {
      const float fval = features[idx[j]];
      if (uniqs[uniqs_size - 1] < fval)
        uniqs[uniqs_size++] = fval;
    }
    //define thresholds
    if (uniqs_size <= nthresholds || nthresholds == 0) {
      uniqs[uniqs_size++] = FLT_MAX;
      thresholds_size[i] = uniqs_size, thresholds[i] = (float*) realloc(
          uniqs, sizeof(float) * uniqs_size);
    } else {
      free(uniqs), thresholds_size[i] = nthresholds + 1, thresholds[i] =
          (float*) malloc(sizeof(float) * (nthresholds + 1));
      float t = features[idx[0]];  //equals fmin
      const float step = fabs(features[idx[sortedsize - 1]] - t) / nthresholds;  //(fmax-fmin)/nthresholds
      for (unsigned int j = 0; j != nthresholds; t += step)
        thresholds[i][j++] = t;
      thresholds[i][nthresholds] = FLT_MAX;
    }
  }
  if (validation_dataset) {
    preprocess_dataset(validation_dataset);
    scores_on_validation = new Score[validation_dataset->num_instances()]();
  }
  hist = new RTRootHistogram(training_dataset.get(), pseudoresponses, sortedsid,
                             sortedsize, thresholds, thresholds_size);
#ifdef SHOWTIMER
  printf("\t\033[1melapsed time = %.3f seconds\033[0m\n", omp_get_wtime()-timer);
#endif
  printf("\tdone\n");
}

void LambdaMart::learn(std::shared_ptr<quickrank::data::Dataset> training_dataset,
                       std::shared_ptr<quickrank::data::Dataset> validation_dataset,
                       std::shared_ptr<quickrank::metric::ir::Metric> scorer, unsigned int partial_save,
                       const std::string output_basename) {

  init(training_dataset, validation_dataset);

  // TODO: move this somewhere else?
  // fix output format for ndcg scores
  std::cout << std::fixed << std::setprecision(4);

  quickrank::MetricScore best_metric_on_validation = 0.0;
  printf("Training:\n");
  printf("\t-----------------------------\n");
  printf("\titeration training validation\n");
  printf("\t-----------------------------\n");
  //set max capacity of the ensamble
  ens.set_capacity(ntrees);
#ifdef SHOWTIMER
  double timer = omp_get_wtime();
#endif
  //start iterations
  for (unsigned int m = 0;
      m < ntrees && (esr == 0 || m <= validation_bestmodel + esr); ++m) {
    compute_pseudoresponses(training_dataset, scorer.get());

    //for (int ii=0; ii<20; ii++)
    //  printf("## %d \t %.16f\n", ii, pseudoresponses[ii]);

    //update the histogram with these training_seting labels (the feature histogram will be used to find the best tree rtnode)
    hist->update(pseudoresponses, training_dataset->num_instances());
    //Fit a regression tree
    RegressionTree tree(ntreeleaves, training_dataset.get(), pseudoresponses,
                        minleafsupport);
    tree.fit(hist);
    //update the outputs of the tree (with gamma computed using the Newton-Raphson method)
    float maxlabel = update_tree_prediction(&tree);

    //add this tree to the ensemble (our model)
    ens.push(tree.get_proot(), shrinkage, maxlabel);

    //Update the model's outputs on all training samples
    update_modelscores(training_dataset, trainingmodelscores, &tree);
    // run metric
    quickrank::MetricScore metric_on_training = scorer->evaluate_dataset(
        training_dataset, trainingmodelscores);

    //				for (int ii=0; ii<20; ii++)
    //					printf("## %d \t %.16f\n", ii, trainingmodelscores[ii]);

    //show results
    printf("\t#%-8u %8.4f", m + 1, metric_on_training);
    //Evaluate the current model on the validation data (if available)
    if (validation_dataset) {
      // update validation scores
      update_modelscores(validation_dataset, scores_on_validation, &tree);

      // run metric
      quickrank::MetricScore metric_on_validation = scorer->evaluate_dataset(
          validation_dataset, scores_on_validation);
      printf(" %9.4f", metric_on_validation);

      if (metric_on_validation > best_metric_on_validation
          || best_metric_on_validation == 0.0f)
        best_metric_on_validation = metric_on_validation, validation_bestmodel =
            ens.get_size() - 1, printf("*");
    }
    printf("\n");

    if (partial_save != 0 and !output_basename.empty()
        and (m + 1) % partial_save == 0) {
      save(output_basename, m + 1);
    }

  }
#ifdef SHOWTIMER
  timer = omp_get_wtime()-timer;
#endif
  //Rollback to the best model observed on the validation data
  if (validation_dataset)
    while (ens.is_notempty() && ens.get_size() > validation_bestmodel + 1)
      ens.pop();
  //Finishing up
  //training_score = compute_score(training_set, scorer);
  score_dataset(training_dataset, trainingmodelscores);
  quickrank::MetricScore metric_on_training = scorer->evaluate_dataset(
      training_dataset, trainingmodelscores);

  printf("\t-----------------------------\n");
  std::cout << "\t" << *scorer << " on training data = " << metric_on_training
      << std::endl;
  if (validation_dataset) {
    score_dataset(validation_dataset, scores_on_validation);
    best_metric_on_validation = scorer->evaluate_dataset(validation_dataset,
                                                         scores_on_validation);
    std::cout << "\t" << *scorer << " on validation data = "
        << best_metric_on_validation << std::endl;
  }
#ifdef SHOWTIMER
  printf("\t\033[1melapsed time = %.3f seconds\033[0m\n", timer);
#endif
  printf("\tdone\n");
}


void LambdaMart::update_modelscores(std::shared_ptr<data::Dataset> dataset, Score *scores,
                                    RegressionTree* tree) {
  quickrank::Score* score_i = scores;
  for (unsigned int q = 0; q < dataset->num_queries(); q++) {
    std::shared_ptr<quickrank::data::QueryResults> results = dataset
        ->getQueryResults(q);
    const unsigned int offset = dataset->num_instances();
    const Feature* d = results->features();
    for (unsigned int i = 0; i < results->num_results(); i++) {
      score_i[i] += shrinkage * tree->get_proot()->score_instance(d, offset);
      d++;
    }
    score_i += results->num_results();
  }

}


float LambdaMart::update_tree_prediction( RegressionTree* tree) {
  return tree->update_output(pseudoresponses, cachedweights);
}

void LambdaMart::compute_pseudoresponses( std::shared_ptr<quickrank::data::Dataset> training_dataset,
                                          quickrank::metric::ir::Metric* scorer) {
  const unsigned int cutoff = scorer->cutoff();

  const unsigned int nrankedlists = training_dataset->num_queries();
  #pragma omp parallel for
  for (unsigned int i = 0; i < nrankedlists; ++i) {
    std::shared_ptr<data::QueryResults> qr = training_dataset->getQueryResults(i);

    const unsigned int offset = training_dataset->offset(i);
    double *lambdas = pseudoresponses + offset;
    double *weights = cachedweights + offset;
    for (unsigned int j = 0; j < qr->num_results(); ++j)
      lambdas[j] = weights[j] = 0.0;

    auto ranked = std::shared_ptr<data::RankedResults>( new data::RankedResults(qr, trainingmodelscores + offset) );

    std::unique_ptr<Jacobian> jacobian = scorer->jacobian(ranked);

    // \todo TODO: rank by label once and for all ?
    // \todo TODO: avoid n^2 loop ?
    for (unsigned int j = 0; j < ranked->num_results(); j++) {
      Label jthlabel = ranked->sorted_labels()[j];
      for (unsigned int k = 0; k < ranked->num_results(); k++)
        if (k != j) {
          // skip if we are beyond the top-K results
          if (j >= cutoff && k >= cutoff)
            break;

          Label kthlabel = ranked->sorted_labels()[k];
          if (jthlabel > kthlabel) {
            double deltandcg = fabs(jacobian->at(j, k));

            double rho = 1.0 / (1.0 + exp(
                        trainingmodelscores[offset + ranked->pos_of_rank(j)] -
                        trainingmodelscores[offset + ranked->pos_of_rank(k)] ));
            double lambda = rho * deltandcg;
            double delta = rho * (1.0 - rho) * deltandcg;
            lambdas[ranked->pos_of_rank(j)] += lambda;
            lambdas[ranked->pos_of_rank(k)] -= lambda;
            weights[ranked->pos_of_rank(j)] += delta;
            weights[ranked->pos_of_rank(k)] += delta;
          }
        }
    }
  }
}


// Changes by Cla:
// - added processing of ranked list in ranked order
// - added cut-off in measure changes matrix
/*
void LambdaMart::compute_pseudoresponses(std::shared_ptr<quickrank::data::Dataset> training_dataset, quickrank::metric::ir::Metric* scorer) {
  const unsigned int cutoff = scorer->cutoff();

  const unsigned int nrankedlists = training_dataset->num_queries();
  //const unsigned int *rloffsets = training_set->get_rloffsets();
  #pragma omp parallel for
  for (unsigned int i = 0; i < nrankedlists; ++i) {
    std::shared_ptr<data::QueryResults> qr = training_dataset->getQueryResults(i);

    const unsigned int offset = training_dataset->offset(i);
    double *lambdas = pseudoresponses + offset;
    double *weights = cachedweights + offset;
    for (unsigned int j = 0; j < qr->num_results(); ++j)
      lambdas[j] = weights[j] = 0.0;

    // CLA: line below uses the old sort and not mergesort as in ranklib
    // unsigned int *idx = idxdouble_qsort(trainingmodelscores+offset, ql.size);
    unsigned int *idx = idxdouble_mergesort<Score>(
        trainingmodelscores + offset, qr->num_results());

    Label* sortedlabels = new Label[qr->num_results()];
    for (unsigned int i = 0; i < qr->num_results(); ++i)
      sortedlabels[i] = qr->labels()[idx[i]];

    std::shared_ptr<data::QueryResults> ranked_list = std::shared_ptr<data::QueryResults>(
        new data::QueryResults(qr->num_results(), sortedlabels, NULL) );

    std::unique_ptr<Jacobian> changes = scorer->get_jacobian(ranked_list);

    // \todo TODO: rank by label one and for all ?
    // \todo TODO: look at the top score or at the top labelled ?
    for (unsigned int j = 0; j < ranked_list->num_results(); ++j) {
      float jthlabel = ranked_list->labels()[j];
      for (unsigned int k = 0; k < ranked_list->num_results(); ++k)
        if (k != j) {
          // skip if we are beyond the top-K results
          if (j >= cutoff && k >= cutoff)
            break;

          float kthlabel = ranked_list->labels()[k];
          if (jthlabel > kthlabel) {
            int i_max = j >= k ? j : k;
            int i_min = j >= k ? k : j;
            double deltandcg = fabs(changes->at(i_min, i_max));

            double rho = 1.0 / (1.0 + exp(
                        trainingmodelscores[offset + idx[j]] -
                        trainingmodelscores[offset + idx[k]] ));
            double lambda = rho * deltandcg;
            double delta = rho * (1.0 - rho) * deltandcg;
            lambdas[idx[j]] += lambda;
            lambdas[idx[k]] -= lambda;
            weights[idx[j]] += delta;
            weights[idx[k]] += delta;
          }
        }
    }

    delete[] idx;
    delete[] sortedlabels;

  }
}
*/

std::ofstream& LambdaMart::save_model_to_file(std::ofstream& os) const {
  // write ranker description
  os << *this;
  // save xml model
  ens.save_model_to_file(os);
  return os;
}

}  // namespace forests
}  // namespace learning
}  // namespace quickrank
