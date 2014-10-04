#ifndef QUICKRANK_LEARNING_LMART_H_
#define QUICKRANK_LEARNING_LMART_H_

#include "types.h"
#include "learning/ranker.h"
#include "learning/tree/rt.h"
#include "learning/tree/ensemble.h"

class LambdaMart : public LTR_Algorithm {
 public:
  const unsigned int ntrees; //>0
  const double shrinkage; //>0.0f
  const unsigned int nthresholds; //if nthresholds==0 then no. of thresholds is not limited
  const unsigned int ntreeleaves; //>0
  const unsigned int minleafsupport; //>0
  const unsigned int esr; //If no performance gain on validation data is observed in 'esr' rounds, stop the training process right away (if esr==0 feature is disabled).
 protected:
  float **thresholds = NULL;
  unsigned int *thresholds_size = NULL;
  double *trainingmodelscores = NULL; //[0..nentries-1]
  double *validationmodelscores = NULL; //[0..nentries-1]
  unsigned int validation_bestmodel = 0;
  double *pseudoresponses = NULL;  //[0..nentries-1]
  double *cachedweights = NULL; //corresponds to datapoint.cache
  unsigned int **sortedsid = NULL;
  unsigned int sortedsize = 0;
  RTRootHistogram *hist = NULL;
  Ensemble ens;
 public:
  LambdaMart(unsigned int ntrees, float shrinkage, unsigned int nthresholds,
        unsigned int ntreeleaves, unsigned int minleafsupport, unsigned int esr) :
          ntrees(ntrees), shrinkage(shrinkage), nthresholds(nthresholds),
          ntreeleaves(ntreeleaves), minleafsupport(minleafsupport), esr(esr) {}
  ~LambdaMart() {
    const unsigned int nfeatures = training_set ? training_set->get_nfeatures() : 0;
    for(unsigned int i=0; i<nfeatures; ++i)
      delete [] sortedsid[i],
      free(thresholds[i]);
    delete [] thresholds,
    delete [] thresholds_size,
    delete [] trainingmodelscores,
    delete [] validationmodelscores,
    delete [] pseudoresponses,
    delete [] sortedsid,
    delete [] cachedweights;
    delete hist;
  }

  const char *whoami() const { return "LAMBDA MART"; }

  void showme() {
    printf("\tranker type = %s\n", whoami());
    printf("\tno. of trees = %u\n", ntrees);
    printf("\tshrinkage = %.3f\n", shrinkage);
    if(nthresholds) printf("\tno. of thresholds = %u\n", nthresholds); else printf("\tno. of thresholds = unlimited\n");
    if(esr) printf("\tno. of no gain rounds before early stop = %u\n", esr);
    printf("\tmin leaf support = %u\n", minleafsupport);
    printf("\tno. of tree leaves = %u\n", ntreeleaves);
  }

  void init();

  void learn();

  float eval_dp(float *const *const features, unsigned int idx) const {
    return ens.eval(features, idx);
  }

  void write_outputtofile() {
    if(output_basename) {
      char filename[256];
      sprintf(filename, "%s.best.xml", output_basename);
      write_outputtofile(filename);
      printf("\tmodel filename = %s\n", filename);
    }
  }

 protected:
  float compute_modelscores(DataPointDataset const *samples, double *mscores, RegressionTree const &tree);

  qr::Jacobian *compute_mchange(const qlist &orig, const unsigned int offset);

  // Changes by Cla:
  // - added processing of ranked list in ranked order
  // - added cut-off in measure changes matrix
  void compute_pseudoresponses();

  void write_outputtofile(char *filename) {
    FILE *f = fopen(filename, "w");
    if(f) {
      fprintf(f, "## LambdaMART\n## No. of trees = %u\n## No. of leaves = %u\n## No. of threshold candidates = %d\n## Learning rate = %f\n## Stop early = %u\n\n", ntrees, ntreeleaves, nthresholds==0?-1:(int)nthresholds, shrinkage, esr);
      ens.write_outputtofile(f);
      fclose(f);
    }
  }

};

#endif
