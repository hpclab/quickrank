#ifndef __MATRIXNET_HPP__
#define __MATRIXNET_HPP__

#include <cfloat>
#include <cmath>

#include "learning/ranker.hpp"
#include "learning/tree/ot.hpp"
#include "learning/tree/ensemble.hpp"
#include "utils/qsort.hpp"

class matrixnet : public lmart {
	public:
		const unsigned int treedepth; //>0
	public:
		matrixnet(unsigned int ntrees, float shrinkage, unsigned int nthresholds, unsigned int treedepth, unsigned int minleafsupport, unsigned int esr) : lmart(ntrees, shrinkage, nthresholds, 1<<treedepth, minleafsupport, esr), treedepth(treedepth) {}
		const char *whoami() const {
			return "MATRIX NET";
		}
		void showme() {
			lmart::showme();
			printf("\ttree depth = %u\n", treedepth);
		}
		void learn() {
			training_score = 0.0f,
			validation_bestscore = 0.0f;
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
			for(unsigned int m=0; m<ntrees && (esr==0 || m<=validation_bestmodel+esr); ++m) {
				#ifdef SHOWTIMER
				timer -= omp_get_wtime();
				#endif
				compute_pseudoresponses();
				//update the histogram with these training_seting labels (the feature histogram will be used to find the best tree rtnode)
				hist->update(pseudoresponses, training_set->get_ndatapoints());
				//Fit a oblivious tree
				ot tree(ntreeleaves, training_set, pseudoresponses, minleafsupport, treedepth);
				tree.fit(hist);
				//update the outputs of the tree (with gamma computed using the Newton-Raphson method)
				float maxlabel = tree.update_output(pseudoresponses, cachedweights);
				//add this tree to the ensemble (our model)
				ens.push(tree.get_proot(), shrinkage, maxlabel);
				//Update the model's outputs on all training samples
				training_score = compute_modelscores(training_set, trainingmodelscores, tree);
				//show results
				printf("\t#%-8u %8.4f", m+1, training_score);
				//Evaluate the current model on the validation data (if available)
				if(validation_set) {
					float validation_score = compute_modelscores(validation_set, validationmodelscores, tree);
					printf(" %9.4f", validation_score);
					if(validation_score>validation_bestscore || validation_bestscore==0.0f)
						validation_bestscore = validation_score,
						validation_bestmodel = ens.get_size()-1,
						printf("*");
				}
				printf("\n");
				#ifdef SHOWTIMER
				timer += omp_get_wtime();
				#endif
				if(partialsave_niterations!=0 and output_basename and (m+1)%partialsave_niterations==0) {
					char filename[256];
					sprintf(filename, "%s.%u.xml", output_basename, m+1);
					write_outputtofile(filename);
				}
			}
			//Rollback to the best model observed on the validation data
			if(validation_set)
				while(ens.is_notempty() && ens.get_size()>validation_bestmodel+1)
					ens.pop();
			//Finishing up
			training_score = compute_score(training_set, scorer);
			printf("\t-----------------------------\n");
			printf("\t%s@%u on training data = %.4f\n", scorer->whoami(), scorer->get_k(), training_score);
			if(validation_set) {
				validation_bestscore = compute_score(validation_set, scorer);
				printf("\t%s@%u on validation data = %.4f\n", scorer->whoami(), scorer->get_k(), validation_bestscore);
			}
			#ifdef SHOWTIMER
			printf("\t\033[1melapsed time = %.3f seconds\033[0m\n", timer);
			#endif
			printf("\tdone\n");
		}
	protected:
		void write_outputtofile(const char *filename) {
			FILE *f = fopen(filename, "w");
			if(f) {
				fprintf(f, "## MatrixNet\n## No. of trees = %u\n## No. of leaves = %u\n## No. of threshold candidates = %d\n## Learning rate = %f\n## Stop early = %u\n\n", ntrees, ntreeleaves, nthresholds==0?-1:(int)nthresholds, shrinkage, esr);
				ens.write_outputtofile(f);
				fclose(f);
			}
		}
};

#endif
