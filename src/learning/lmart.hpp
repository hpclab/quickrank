#ifndef __LMART_HPP__
#define __LMART_HPP__

#include <cfloat>
#include <cmath>

#include "learning/ranker.hpp"
#include "learning/tree/rt.hpp"
#include "learning/tree/ensemble.hpp"
#include "utils/qsort.hpp"
#include "utils/mergesorter.hpp"

class lmart : public ranker {
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
		roothistogram *hist = NULL;
		ensemble ens;
	public:
		lmart(unsigned int ntrees, float shrinkage, unsigned int nthresholds, unsigned int ntreeleaves, unsigned int minleafsupport, unsigned int esr) : ntrees(ntrees), shrinkage(shrinkage), nthresholds(nthresholds), ntreeleaves(ntreeleaves), minleafsupport(minleafsupport), esr(esr) {}
		~lmart() {
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
		const char *whoami() const {
			return "LAMBDA MART";
		}
		void showme() {
			printf("\tranker type = %s\n", whoami());
			printf("\tno. of trees = %u\n", ntrees);
			printf("\tshrinkage = %.3f\n", shrinkage);
			if(nthresholds) printf("\tno. of thresholds = %u\n", nthresholds); else printf("\tno. of thresholds = unlimited\n");
			if(esr) printf("\tno. of no gain rounds before early stop = %u\n", esr);
			printf("\tmin leaf support = %u\n", minleafsupport);
			printf("\tno. of tree leaves = %u\n", ntreeleaves);
		}
		void init()  {
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
			}
			hist = new roothistogram(training_set, pseudoresponses, sortedsid, sortedsize, thresholds, thresholds_size);
			#ifdef SHOWTIMER
			printf("\t\033[1melapsed time = %.3f seconds\033[0m\n", omp_get_wtime()-timer);
			#endif
			printf("\tdone\n");
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
				rt tree(ntreeleaves, training_set, pseudoresponses, minleafsupport);
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
					if(validation_score>validation_bestscore || validation_bestscore==0.0f)
						validation_bestscore = validation_score,
						validation_bestmodel = ens.get_size()-1,
						printf("*");
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
		float compute_modelscores(dpset const *samples, double *mscores, rt const &tree) {
			const unsigned int ndatapoints = samples->get_ndatapoints();
			float **featurematrix = samples->get_fmatrix();
			#pragma omp parallel for
			for(unsigned int i=0; i<ndatapoints; ++i)
				mscores[i] += shrinkage*tree.eval(featurematrix, i);
			const unsigned int nrankedlists = samples->get_nrankedlists();
			const unsigned int *offsets = samples->get_rloffsets();
			float score = 0.0f;
			if(nrankedlists) {
				#pragma omp parallel for reduction(+:score)
				for(unsigned int i=0; i<nrankedlists; ++i) {
					qlist orig = samples->get_qlist(i);
					// double *sortedlabels = copyextdouble_qsort(orig.labels, mscores+offsets[i], orig.size);
					double *sortedlabels = copyextdouble_mergesort(orig.labels, mscores+offsets[i], orig.size);
					score += scorer->compute_score(qlist(orig.size, sortedlabels, orig.qid));
					delete[] sortedlabels;
				}
				score /= nrankedlists;
			}
			return score;
		}

		fsymmatrix *compute_mchange(const qlist &orig, const unsigned int offset) {
			//build a ql made up of label values picked up from orig order by indexes of trainingmodelscores reversely sorted
			unsigned int *idx = idxdouble_qsort(trainingmodelscores+offset, orig.size);
			double* sortedlabels = new double [orig.size]; // float sortedlabels[orig.size];
			for(unsigned int i=0; i<orig.size; ++i)
				sortedlabels[i] = orig.labels[idx[i]];
			qlist tmprl(orig.size, sortedlabels, orig.qid);
			//alloc mem
			fsymmatrix *reschanges = new fsymmatrix(orig.size);
			//compute temp swap changes on ql
			fsymmatrix *tmpchanges = scorer->swap_change(tmprl);
			#pragma omp parallel for
			for(unsigned int i=0; i<orig.size; ++i)
				for(unsigned int j=i; j<orig.size; ++j)
					reschanges->at(idx[i],idx[j]) = tmpchanges->at(i,j);
			delete tmpchanges,
			delete [] idx;
			delete [] sortedlabels;
			return reschanges;
		}

		// Changes by Cla:
		// - added processing of ranked list in ranked order
		// - added cut-off in measure changes matrix
		void compute_pseudoresponses() {
			const unsigned int cutoff = scorer->get_k();

			const unsigned int nrankedlists = training_set->get_nrankedlists();
			const unsigned int *rloffsets = training_set->get_rloffsets();
			#pragma omp parallel for
			for(unsigned int i=0; i<nrankedlists; ++i) {
				const unsigned int offset = rloffsets[i];
				qlist ql = training_set->get_qlist(i);

				// CLA: line below uses the old sort and not mergesort as in ranklib
				// unsigned int *idx = idxdouble_qsort(trainingmodelscores+offset, ql.size);
				unsigned int *idx = idxdouble_mergesort(trainingmodelscores+offset, ql.size);

				double* sortedlabels = new double [ql.size];
				for(unsigned int i=0; i<ql.size; ++i)
					sortedlabels[i] = ql.labels[idx[i]];
				qlist ranked_list(ql.size, sortedlabels, ql.qid);
				//compute temp swap changes on ql
				fsymmatrix *changes = scorer->swap_change(ranked_list);

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
							double deltandcg = fabs(changes->at(i_min,i_max));

							double rho = 1.0/(1.0+exp(trainingmodelscores[offset+idx[j]]-trainingmodelscores[offset+idx[k]]));
							double lambda = rho*deltandcg;
							double delta = rho*(1.0-rho)*deltandcg;
							lambdas[ idx[j] ] += lambda,
							lambdas[ idx[k] ] -= lambda,
							weights[ idx[j] ] += delta,
							weights[ idx[k] ] += delta;

//							if (i==0 && (idx[j]==0 || idx[k]==0)) {
//								// printf("## lambda[0]------------------\n");
//								printf("## %d\t%d", idx[j],idx[k]);
//								printf("\t%d\t%d", j,k);
//								printf("\t%.6f\t%.6f", trainingmodelscores[offset+idx[j]], trainingmodelscores[offset+idx[k]]);
//								printf("\t%.6f", rho);
//								printf("\t%.6f", deltandcg);
//								printf("\t%.6f\n", pseudoresponses[0]);
//							}
						} /* else {
							if (i==0 && (idx[j]==0 || idx[k]==0)) {
								// printf("## lambda[0]------------------\n");
								printf("** %d\t%d", idx[j],idx[k]);
								printf("\t%d\t%d", j,k);
								printf("\t%.6f\t%.6f", trainingmodelscores[offset+idx[j]], trainingmodelscores[offset+idx[k]]);
								printf("\t%.6f", jthlabel);
								printf("\t%.6f", kthlabel);
								printf("\t%.6f\n", pseudoresponses[0]);
							}
						}*/
					}
				}

//				if (i==0) {
//					printf("## lambda[0]------------------\n");
//					printf("## cur scoce  = %.15f\n", trainingmodelscores[0]);
//					printf("## pseudoresp = %.15f\n", pseudoresponses[0]);
//					printf("## weights    = %.15f\n", cachedweights[0]);
//				}

				delete [] idx;
				delete [] sortedlabels;
				delete changes;
			}
		}


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
