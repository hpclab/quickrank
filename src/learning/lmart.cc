#include "learning/lmart.h"

#include <iostream>
#include <iomanip>
#include <cfloat>
#include <cmath>

#include "utils/qsort.h"
#include "utils/mergesorter.h"

namespace quickrank {
namespace learning {
namespace forests {

void LambdaMart::init()  {
  printf("Initialization:\n");
#ifdef SHOWTIMER
  double timer = omp_get_wtime();
#endif
  const unsigned int nentries = training_set->get_ndatapoints();
  trainingmodelscores = new double[nentries]();  //0.0f initialized
  pseudoresponses = new double[nentries](); //0.0f initialized
  cachedweights = new double[nentries](); //0.0f initialized
  const unsigned int nfeatures = training_set->get_nfeatures();
  sortedsid = new unsigned int*[nfeatures],
      sortedsize = training_set->get_ndatapoints();
#pragma omp parallel for
  for(unsigned int i=0; i<nfeatures; ++i)
    training_set->sort_dpbyfeature(i, sortedsid[i], sortedsize);
  //for each featureid, init threshold array by keeping track of the list of "unique values" and their max, min
  thresholds = new float*[nfeatures],
      thresholds_size = new unsigned int[nfeatures];
#pragma omp parallel for
  for(unsigned int i=0; i<nfeatures; ++i) {
    //select feature array realted to the current feature index
    float const* features = training_set->get_fvector(i);
    //init with values with the 1st sample
    unsigned int *idx = sortedsid[i];
    //get_ sample indexes sorted by the fid-th feature
    unsigned int uniqs_size = 0;
    float *uniqs = (float*)malloc(sizeof(float)*(nthresholds==0?sortedsize+1:nthresholds+1));
    //skip samples with the same feature value. early stop for if nthresholds!=size_max
    uniqs[uniqs_size++] = features[idx[0]];
    for(unsigned int j=1; j<sortedsize && (nthresholds==0 || uniqs_size!=nthresholds+1); ++j) {
      const float fval = features[idx[j]];
      if(uniqs[uniqs_size-1]<fval) uniqs[uniqs_size++] = fval;
    }
    //define thresholds
    if(uniqs_size<=nthresholds || nthresholds==0) {
      uniqs[uniqs_size++] = FLT_MAX;
      thresholds_size[i] = uniqs_size,
          thresholds[i] = (float*)realloc(uniqs, sizeof(float)*uniqs_size);
    } else {
      free(uniqs),
          thresholds_size[i] = nthresholds+1,
          thresholds[i] = (float*)malloc(sizeof(float)*(nthresholds+1));
      float t = features[idx[0]]; //equals fmin
      const float step = fabs(features[idx[sortedsize-1]]-t)/nthresholds; //(fmax-fmin)/nthresholds
      for(unsigned int j=0; j!=nthresholds; t+=step)
        thresholds[i][j++] = t;
      thresholds[i][nthresholds] = FLT_MAX;
    }
  }
  if(validation_set) {
    unsigned int ndatapoints = validation_set->get_ndatapoints();
    validationmodelscores = new double[ndatapoints]();

    if (validation_dataset->format()!=quickrank::data::Dataset::VERT)
      validation_dataset->transpose();
    scores_on_validation = new qr::Score[validation_dataset->num_instances()] ();
  }
  hist = new RTRootHistogram(training_set, pseudoresponses, sortedsid, sortedsize, thresholds, thresholds_size);
#ifdef SHOWTIMER
  printf("\t\033[1melapsed time = %.3f seconds\033[0m\n", omp_get_wtime()-timer);
#endif
  printf("\tdone\n");
}

void LambdaMart::learn() {
  // TODO: move this somewhere else?
  // fix output format for ndcg scores
  std::cout << std::fixed << std::setprecision(4);

  training_score = 0.0f,
      validation_bestscore = 0.0f;
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
  for(unsigned int m=0; m<ntrees && (esr==0 || m<=validation_bestmodel+esr); ++m) {
    compute_pseudoresponses();

    //				for (int ii=0; ii<20; ii++)
    //					printf("## %d \t %.16f\n", ii, pseudoresponses[ii]);

    //update the histogram with these training_seting labels (the feature histogram will be used to find the best tree rtnode)
    hist->update(pseudoresponses, training_set->get_ndatapoints());
    //Fit a regression tree
    RegressionTree tree(ntreeleaves, training_set, pseudoresponses, minleafsupport);
    tree.fit(hist);
    //update the outputs of the tree (with gamma computed using the Newton-Raphson method)
    float maxlabel = tree.update_output(pseudoresponses, cachedweights);

    //add this tree to the ensemble (our model)
    ens.push(tree.get_proot(), shrinkage, maxlabel);
    //Update the model's outputs on all training samples
    training_score = compute_modelscores(training_set, trainingmodelscores, tree);

    //				for (int ii=0; ii<20; ii++)
    //					printf("## %d \t %.16f\n", ii, trainingmodelscores[ii]);

    //show results
    printf("\t#%-8u %8.4f", m+1, training_score);
    //Evaluate the current model on the validation data (if available)
    if(validation_set) {
      const float validation_score = compute_modelscores(validation_set, validationmodelscores, tree);
      printf(" %9.4f", validation_score);

      // update validation scores
      qr::Score* score_i = scores_on_validation;
      for (unsigned int q=0; q<validation_dataset->num_queries(); q++) {
        std::shared_ptr<quickrank::data::QueryResults> results = validation_dataset->getQueryResults(q);
        const unsigned int offset = results->num_results();
        const qr::Feature* d = results->features();
        for (unsigned int i=0; i<results->num_results(); i++) {
          score_i[i] += shrinkage*tree.get_proot()->score_instance(d,offset);
          d++;
        }
        score_i += results->num_results();
      }

      // run metric
      qr::MetricScore metric_on_validation = scorer->evaluate_dataset(*validation_dataset, scores_on_validation);
      printf(" %9.4f", metric_on_validation);

      if(validation_score>validation_bestscore || validation_bestscore==0.0f)
        validation_bestscore = validation_score,
        validation_bestmodel = ens.get_size()-1,
        printf("*");

      /*
      for (unsigned int i=0; i<validation_dataset->num_instances(); i++)
        if ( validationmodelscores[i] != scores_on_validation[i] ) {
          printf("\t %d : %.16f : %.16f", i, validationmodelscores[i], scores_on_validation[i]);
          break;
        }*/
    }
    printf("\n");
    if(partialsave_niterations!=0 and output_basename and (m+1)%partialsave_niterations==0) {
      char filename[256];
      sprintf(filename, "%s.%u.xml", output_basename, m+1);
      write_outputtofile(filename);
    }
  }
#ifdef SHOWTIMER
  timer = omp_get_wtime()-timer;
#endif
  //Rollback to the best model observed on the validation data
  if(validation_set)
    while(ens.is_notempty() && ens.get_size()>validation_bestmodel+1)
      ens.pop();
  //Finishing up
  training_score = compute_score(training_set, scorer);
  printf("\t-----------------------------\n");
  std::cout << "\t" << *scorer
            << " on training data = " << training_score << std::endl;
  if(validation_set) {
    validation_bestscore = compute_score(validation_set, scorer);
    std::cout << "\t" << *scorer
              << " on validation data = " << validation_bestscore << std::endl;
  }
#ifdef SHOWTIMER
  printf("\t\033[1melapsed time = %.3f seconds\033[0m\n", timer);
#endif
  printf("\tdone\n");
}

float LambdaMart::compute_modelscores(LTR_VerticalDataset const *samples, double *mscores, RegressionTree const &tree) {
  const unsigned int ndatapoints = samples->get_ndatapoints();
  float **featurematrix = samples->get_fmatrix();
#pragma omp parallel for
  for(unsigned int i=0; i<ndatapoints; ++i)
    mscores[i] += shrinkage*tree.eval(featurematrix, i);
  const unsigned int nrankedlists = samples->get_nrankedlists();
//  const unsigned int *offsets = samples->get_rloffsets()();
  float score = 0.0f;
  if(nrankedlists) {
#pragma omp parallel for reduction(+:score)
    for(unsigned int i=0; i<nrankedlists; ++i) {
      ResultList orig = samples->get_qlist(i);
      //double *sortedlabels = copyextdouble_qsort(orig.labels, mscores+samples->get_rloffsets(i), orig.size);
      std::unique_ptr<double[]> sortedlabels = copyextdouble_mergesort<double,double>(orig.labels, mscores+samples->get_rloffsets(i), orig.size);
      score += scorer->evaluate_result_list(ResultList(orig.size, sortedlabels.get(), orig.qid));
    }
    score /= nrankedlists;
  }
  return score;
}

std::unique_ptr<qr::Jacobian> LambdaMart::compute_mchange(const ResultList &orig, const unsigned int offset) {
  //build a ql made up of label values picked up from orig order by indexes of trainingmodelscores reversely sorted
  unsigned int *idx = idxdouble_mergesort(trainingmodelscores+offset, orig.size);
  //unsigned int *idx = idxdouble_qsort(trainingmodelscores+offset, orig.size);
  double* sortedlabels = new double [orig.size]; // float sortedlabels[orig.size];
  for(unsigned int i=0; i<orig.size; ++i)
    sortedlabels[i] = orig.labels[idx[i]];
  ResultList tmprl(orig.size, sortedlabels, orig.qid);
  //alloc mem
  std::unique_ptr<qr::Jacobian> reschanges =
      std::unique_ptr<qr::Jacobian>(new qr::Jacobian(orig.size));
  qr::Jacobian* reschanges_p = reschanges.get();
  //compute temp swap changes on ql
  std::unique_ptr<qr::Jacobian> tmpchanges = scorer->get_jacobian(tmprl);
  qr::Jacobian* tmpchanges_p = tmpchanges.get();
#pragma omp parallel for
  for(unsigned int i=0; i<orig.size; ++i)
    for(unsigned int j=i; j<orig.size; ++j)
      reschanges_p->at(idx[i],idx[j]) = tmpchanges_p->at(i,j);
  // delete tmpchanges,
  delete [] idx;
  delete [] sortedlabels;
  return reschanges;
}

// Changes by Cla:
// - added processing of ranked list in ranked order
// - added cut-off in measure changes matrix
void LambdaMart::compute_pseudoresponses() {
  const unsigned int cutoff = scorer->cutoff();

  const unsigned int nrankedlists = training_set->get_nrankedlists();
  //const unsigned int *rloffsets = training_set->get_rloffsets();
#pragma omp parallel for
  for(unsigned int i=0; i<nrankedlists; ++i) {
    const unsigned int offset = training_set->get_rloffsets(i);
    ResultList ql = training_set->get_qlist(i);

    // CLA: line below uses the old sort and not mergesort as in ranklib
    // unsigned int *idx = idxdouble_qsort(trainingmodelscores+offset, ql.size);
    unsigned int *idx = idxdouble_mergesort(trainingmodelscores+offset, ql.size);

    double* sortedlabels = new double [ql.size];
    for(unsigned int i=0; i<ql.size; ++i)
      sortedlabels[i] = ql.labels[idx[i]];
    ResultList ranked_list(ql.size, sortedlabels, ql.qid);
    //compute temp swap changes on ql
    std::unique_ptr<qr::Jacobian> changes = scorer->get_jacobian(ranked_list);
    qr::Jacobian* changes_p = changes.get();

    double *lambdas = pseudoresponses+offset;
    double *weights = cachedweights+offset;
    for(unsigned int j=0; j<ranked_list.size; ++j)
      lambdas[j] = 0.0,
      weights[j] = 0.0;
    for(unsigned int j=0; j<ranked_list.size; ++j) {
      float jthlabel = ranked_list.labels[j];
      for(unsigned int k=0; k<ranked_list.size; ++k) if(k!=j) {
        // skip if we are beyond the top-K results
        if (j>=cutoff && k>=cutoff) break;

        float kthlabel = ranked_list.labels[k];
        if(jthlabel>kthlabel) {
          int i_max = j>=k ? j : k;
          int i_min = j>=k ? k : j;
          double deltandcg = fabs(changes_p->at(i_min,i_max));

          double rho = 1.0/(1.0+exp(trainingmodelscores[offset+idx[j]]-trainingmodelscores[offset+idx[k]]));
          double lambda = rho*deltandcg;
          double delta = rho*(1.0-rho)*deltandcg;
          lambdas[ idx[j] ] += lambda,
              lambdas[ idx[k] ] -= lambda,
              weights[ idx[j] ] += delta,
              weights[ idx[k] ] += delta;
        }
      }
    }

    delete [] idx;
    delete [] sortedlabels;
  }
}

} // namespace forests
} // namespace learning
} // namespace quickrank


