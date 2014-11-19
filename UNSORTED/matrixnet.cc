#include "learning/matrixnet.h"

#include <iostream>
#include <iomanip>
#include <cfloat>
#include <cmath>

#include "learning/ltr_algorithm.h"
#include "learning/tree/ot.h"
#include "learning/tree/ensemble.h"
#include "utils/qsort.h"

namespace quickrank {
namespace learning {
namespace forests {

void MatrixNet::learn() {
  std::cout << std::fixed << std::setprecision(4);

  training_score = 0.0f, validation_bestscore = 0.0f;
  printf("Training:\n");
  printf("\t-----------------------------\n");
  printf("\titeration training validation\n");
  printf("\t-----------------------------\n");
  //set max capacity of the ensamble
  ens.set_capacity(ntrees);
#ifdef SHOWTIMER
  double timer = 0.0;
#endif
  //start iterations
  for (unsigned int m = 0;
      m < ntrees && (esr == 0 || m <= validation_bestmodel + esr); ++m) {
#ifdef SHOWTIMER
    timer -= omp_get_wtime();
#endif
    compute_pseudoresponses();
    //update the histogram with these training_seting labels (the feature histogram will be used to find the best tree rtnode)
    hist->update(pseudoresponses, training_set->get_ndatapoints());
    //Fit a oblivious tree
    ObliviousRT tree(ntreeleaves, training_set, pseudoresponses, minleafsupport,
                     treedepth);
    tree.fit(hist);
    //update the outputs of the tree (with gamma computed using the Newton-Raphson method)
    float maxlabel = tree.update_output(pseudoresponses, instance_weights_);
    //add this tree to the ensemble (our model)
    ens.push(tree.get_proot(), shrinkage, maxlabel);
    //Update the model's outputs on all training samples
    training_score = compute_modelscores(training_set, trainingmodelscores,
                                         tree);
    //show results
    printf("\t#%-8u %8.4f", m + 1, training_score);
    //Evaluate the current model on the validation data (if available)
    if (validation_set) {
      float validation_score = compute_modelscores(validation_set,
                                                   validationmodelscores, tree);
      printf(" %9.4f", validation_score);
      if (validation_score > validation_bestscore
          || validation_bestscore == 0.0f)
        validation_bestscore = validation_score, validation_bestmodel = ens
            .get_size() - 1, printf("*");
    }
    printf("\n");
#ifdef SHOWTIMER
    timer += omp_get_wtime();
#endif
    if (partialsave_niterations != 0 and output_basename
        and (m + 1) % partialsave_niterations == 0) {
      char filename[256];
      sprintf(filename, "%s.%u.xml", output_basename, m + 1);
      write_outputtofile(filename);
    }
  }
  //Rollback to the best model observed on the validation data
  if (validation_set)
    while (ens.is_notempty() && ens.get_size() > validation_bestmodel + 1)
      ens.pop();
  //Finishing up
  training_score = compute_score(training_set, scorer);
  printf("\t-----------------------------\n");
  std::cout << "\t" << *scorer << " on training data = " << training_score
      << std::endl;
  if (validation_set) {
    validation_bestscore = compute_score(validation_set, scorer);
    std::cout << "\t" << *scorer << " on validation data = "
        << validation_bestscore << std::endl;
  }
#ifdef SHOWTIMER
  printf("\t\033[1melapsed time = %.3f seconds\033[0m\n", timer);
#endif
  printf("\tdone\n");
}

void MatrixNet::write_outputtofile(const char *filename) {
  FILE *f = fopen(filename, "w");
  if (f) {
    fprintf(
        f,
        "## MatrixNet\n## No. of trees = %u\n## No. of leaves = %u\n## No. of threshold candidates = %d\n## Learning rate = %f\n## Stop early = %u\n\n",
        ntrees, ntreeleaves, nthresholds == 0 ? -1 : (int) nthresholds,
        shrinkage, esr);
    ens.write_outputtofile(f);
    fclose(f);
  }
}

}  // namespace forests
}  // namespace learning
}  // namespace quickrank
