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
		matrixnet(unsigned int ntrees, float shrinkage, unsigned int nthresholds, unsigned int treedepth, unsigned int minleafsupport, unsigned int esr, const bool verbose=true) : lmart(ntrees, shrinkage, nthresholds, 1<<treedepth, minleafsupport, esr, false), treedepth(treedepth) {
			if(verbose) printf("\tranker type = '%s'\n\tno. of trees = %u\n\tshrinkage = %f\n\tno. of thresholds = %u\n\tno. of treee leaves = %u\n\toblivious tree depth = %u\n\tmin leaf support = %u\n\tno. of no gain rounds before early stop = %u\n", whoami(), ntrees, shrinkage, nthresholds, ntreeleaves, treedepth, minleafsupport, esr);
		}
		const char *whoami() const {
			return "MATRIX NET";
		}
		void learn() {
			training_score = 0.0f,
			validation_bestscore = 0.0f;
			printf("Training:\n");
			printf("\t---------------------------------------------\n");
			printf("\titeration training validation validation-gain\n");
			printf("\t---------------------------------------------\n");
			#ifdef SHOWTIMER
			#define NTIMERS 2
			double timervalues[NTIMERS];
			unsigned int timercounter = 0;
			for(int i=0; i<NTIMERS; ++i) timervalues[i] = 0.0;
			const char *timerlabels[NTIMERS] = {"regression tree computation", "model evaluation"};
			#endif
			//set max capacity of the ensamble
			ens.set_capacity(ntrees);
			//start iterations
			for(unsigned int m=0; m<ntrees && (esr==0 || m<=validation_bestmodel+esr); ++m) {
				compute_pseudoresponses();
				//update the histogram with these training_seting labels (the feature histogram will be used to find the best tree rtnode)
				hist->update(pseudoresponses, training_set->get_ndatapoints());
				//Fit a regression tree
				ot tree(ntreeleaves, training_set, pseudoresponses, minleafsupport, treedepth);
				#ifdef SHOWTIMER
				++timercounter,
				timervalues[0] -= omp_get_wtime();
				#endif
				tree.fit(hist);
				#ifdef SHOWTIMER
				timervalues[0] += omp_get_wtime();
				#endif
				//update the outputs of the tree (with gamma computed using the Newton-Raphson method)
				float maxlabel = tree.update_output(pseudoresponses, cachedweights);
				//add this tree to the ensemble (our model)
				ens.push(tree.get_proot(), shrinkage, maxlabel);
				//Update the model's outputs on all training samples
				unsigned int ndatapoints = training_set->get_ndatapoints();
				float **featurematrix = training_set->get_fmatrix();
				#pragma omp parallel for
				for(unsigned int i=0; i<ndatapoints; ++i)
					modelscores[i] += shrinkage*tree.eval(featurematrix, i);
				#ifdef SHOWTIMER
				timervalues[1] -= omp_get_wtime();
				#endif
				//Evaluate the current model
				training_score = compute_trainingscore();
				#ifdef SHOWTIMER
				timervalues[1] += omp_get_wtime();
				#endif
				printf("\t#%-8u %-8.4f", m+1, training_score);
				//Evaluate the current model on the validation data (if available)
				if(validation_set) {
					unsigned int ndatapoints = validation_set->get_ndatapoints();
					float **featurematrix = validation_set->get_fmatrix();
					#pragma omp parallel for
					for(unsigned int i=0; i<ndatapoints; ++i)
						validationmodelscores[i] += shrinkage*tree.eval(featurematrix, i);
					float validation_score = compute_validationmodelscores();
					printf(" %-8.4f", validation_score);
					if(validation_score>validation_bestscore || validation_bestscore==0.0f)
						validation_bestscore = validation_score,
						validation_bestmodel = ens.get_size()-1,
						printf("   *");
				}
				printf("\n");
				if(partialsave_niterations!=0 and output_basename and (m+1)%partialsave_niterations==0) {
					int ndigits = 1+(int)log10(ntrees);
					char filename[1000];
					sprintf(filename, "%s.%0*u.xml", output_basename, ndigits, m+1);
					write_outputtofile(filename);
				}
			}
			//Rollback to the best model observed on the validation data
			while(ens.is_notempty() && ens.get_size()>validation_bestmodel+1)
				ens.pop();
			//Finishing up
			training_score = compute_score(training_set, scorer);
			printf("\t---------------------------------------------\n");
			printf("\t%s@%u on training data = %.4f\n", scorer->whoami(), scorer->get_k(), training_score);
			if(validation_set) {
				validation_bestscore = compute_score(validation_set, scorer);
				printf("\t%s@%u on validation data = %.4f\n", scorer->whoami(), scorer->get_k(), validation_bestscore);
			}
			#ifdef SHOWTIMER
			for(int i=0; i<NTIMERS; ++i)
				printf("\tavg '%s' elapsed time = %.3f seconds\n", timerlabels[i], timervalues[i]/timercounter);
			#undef NTIMERS
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
