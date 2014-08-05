#ifndef __LMART_HPP__
#define __LMART_HPP__

#include <cfloat>
#include <cmath>

#include "learning/ranker.hpp"
#include "learning/tree/rt.hpp"
#include "learning/tree/ensemble.hpp"
#include "utils/qsort.hpp"

class lmart : public ranker {
	public:
		const unsigned int ntrees; //>0
		const float shrinkage; //>0.0f
		const unsigned int nthresholds; //if nthresholds==0 no. of thresholds is not limited
		const unsigned int ntreeleaves;
		const unsigned int minleafsupport; //>0
		const unsigned int esr; //If no performance gain on validation data is observed in eerounds, stop the training process right away (if esr==0 feature is disabled).
	protected:
		float **thresholds = NULL;
		unsigned int *thresholds_size = NULL;
		float *modelscores = NULL; //[0..nentries-1]
		float *validationmodelscores = NULL; //[0..nentries-1]
		unsigned int validation_bestmodel = 0;
		float *pseudoresponses = NULL;  //[0..nentries-1]
		float *cachedweights = NULL; //corresponds to datapoint.cache
		unsigned int **sortedsid = NULL;
		unsigned int sortedsize = 0;
		basehistogram *hist = NULL;
		ensemble ens;
	public:
		lmart(unsigned int ntrees, float shrinkage, unsigned int nthresholds, unsigned int ntreeleaves, unsigned int minleafsupport, unsigned int esr, const bool verbose=true) : ntrees(ntrees), shrinkage(shrinkage), nthresholds(nthresholds), ntreeleaves(ntreeleaves), minleafsupport(minleafsupport), esr(esr) {
			if(verbose) printf("\tranker type = '%s'\n\tno. of trees = %u\n\tshrinkage = %f\n\tno. of thresholds = %u (0 means unlimited)\n\tno. of tree leaves = %u\n\tmin leaf support = %u\n\tno. of no gain rounds before early stop = %u (0 means unlimited)\n", whoami(), ntrees, shrinkage, nthresholds, ntreeleaves, minleafsupport, esr);
		};
		~lmart() {
			const unsigned int nfeatures = training_set ? training_set->get_nfeatures() : 0;
			for(unsigned int i=0; i<nfeatures; ++i)
				delete [] sortedsid[i],
				free(thresholds[i]);
			delete [] thresholds,
			delete [] thresholds_size,
			delete [] modelscores,
			delete [] validationmodelscores,
			delete [] pseudoresponses,
			delete [] sortedsid,
			delete [] cachedweights;
			delete hist;
		}
		const char *whoami() const {
			return "LAMBDA MART";
		}
		void init()  {
			printf("Initialization:\n");
			#ifdef SHOWTIMER
			double timer = omp_get_wtime();
			#endif
			const unsigned int nentries = training_set->get_ndatapoints();
			modelscores = new float[nentries]();  //0.0f initialized
			pseudoresponses = new float[nentries](); //0.0f initialized
			cachedweights = new float[nentries](); //0.0f initialized
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
				validationmodelscores = new float[ndatapoints]();
			}
			#ifdef LOGFILE
			fprintf(flog, "sortedsid\n");
			for(unsigned int i=0; i<nfeatures; ++i) {
				for(unsigned int j=0; j<sortedsize; ++j)
					fprintf(flog, "%u ", sortedsid[i][j]);
				fprintf(flog, "\n");
			}
			fprintf(flog, "thresholds\n");
			for(unsigned int i=0; i<nfeatures; ++i) {
				for(unsigned int j=0; j<thresholds_size[i]-1; ++j)
					fprintf(flog, "%.4f ", thresholds[i][j]);
				fprintf(flog, "\n");
			}
			#endif
			hist = new basehistogram(training_set, pseudoresponses, sortedsid, sortedsize, thresholds, thresholds_size);
			#ifdef SHOWTIMER
			printf("\telapsed time = %.3f seconds\n", omp_get_wtime()-timer);
			#endif
			printf("\tdone\n");
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
				rt tree(ntreeleaves, training_set, pseudoresponses, minleafsupport);
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
		float eval_dp(float *const *const features, unsigned int idx) const {
			return ens.eval(features, idx);
		}
		void write_outputtofile() {
			if(output_basename) {
				char filename[1000];
				sprintf(filename, "%s.xml", output_basename);
				write_outputtofile(filename);
				printf("\tmodel filename = '%s'\n", filename);
			}

		}
	protected:
		float compute_validationmodelscores() {
			float score = 0.0f;
			unsigned int nrankedlists = validation_set->get_nrankedlists();
			unsigned int *offsets = validation_set->get_rloffsets();
			if(nrankedlists) {
				#pragma omp parallel for reduction(+:score)
				for(unsigned int i=0; i<nrankedlists; ++i) {
					qlist orig = validation_set->get_ranklist(i);
					float *sortedlabels = copyextfloat_qsort(orig.labels, validationmodelscores+offsets[i], orig.size);
					score += scorer->compute_score(qlist(orig.size, sortedlabels, orig.qid));
					delete[] sortedlabels;
				}
				score /= nrankedlists;
			}
			return score;
		}
		void compute_pseudoresponses() {
			const unsigned int nrankedlists = training_set->get_nrankedlists();
			const unsigned int *rloffsets = training_set->get_rloffsets();
			#pragma omp parallel for
			for(unsigned int i=0; i<nrankedlists; ++i) {
				const unsigned int offset = rloffsets[i];
				qlist ql = training_set->get_ranklist(i);
				fsymmatrix *changes = compute_mchange(ql, offset);
				float *lambdas = pseudoresponses+offset;
				float *weights = cachedweights+offset;
				for(unsigned int j=0; j<ql.size; ++j)
					lambdas[j] = 0.0f,
					weights[j] = 0.0f;
				for(unsigned int j=0; j<ql.size; ++j) {
					float jthlabel = ql.labels[j];
					for(unsigned int k=0; k<ql.size; ++k) if(k!=j) {
						float kthlabel = ql.labels[k];
						float deltandcg = fabs(changes->at(j,k));
						if(jthlabel>kthlabel) {
							float rho = 1.0/(1.0+exp(modelscores[offset+j]-modelscores[offset+k]));
							float lambda = rho*deltandcg;
							float delta = rho*(1.0-rho)*deltandcg;
							lambdas[j] += lambda,
							lambdas[k] -= lambda,
							weights[j] += delta,
							weights[k] += delta;
						}
					}
				}
				delete changes;
			}
			#ifdef LOGFILE
			fprintf(flog,"pseudoresponses\n");
			const unsigned int nentries = training_set->get_ndatapoints();
			for(unsigned int i=0; i<nentries; ++i)
				fprintf(flog,"%.4f ", pseudoresponses[i]);
			fprintf(flog,"\n");
			#endif
		}
		float compute_trainingscore() {
			unsigned int nrankedlists = training_set->get_nrankedlists();
			unsigned int *offsets = training_set->get_rloffsets();
			float avg = 0.0f;
			if(nrankedlists) {
				#pragma omp parallel for reduction(+:avg)
				for(unsigned int i=0; i<nrankedlists; ++i) {
					qlist orig = training_set->get_ranklist(i);
					float *sortedlabels = copyextfloat_qsort(orig.labels, modelscores+offsets[i], orig.size);
					avg += scorer->compute_score(qlist(orig.size, sortedlabels, orig.qid));
					delete [] sortedlabels;
				}
				avg /= nrankedlists;
			}
			return avg;
		}

		// TODO: (by cla) this doesn't need to be full if k<qlist.size
		fsymmatrix *compute_mchange(const qlist &orig, const unsigned int offset) {
			//build a ql made up of label values picked up from orig order by indexes of modelscores reversely sorted
			unsigned int *idx = idxfloat_qsort(modelscores+offset, orig.size);
			float* sortedlabels = new float [orig.size]; // float sortedlabels[orig.size];
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
			#ifdef LOGFILE
			unsigned int changes_size = tmpchanges->get_size();
			fprintf(flog, "changes %u :", changes_size);
			for(unsigned int ii=0; ii<tmprl.size; ++ii)
				fprintf(flog, " %u(%.3f)", idx[ii], modelscores[ii+offset]);
			fprintf(flog, "\n");
			for(unsigned int ii=0; ii<changes_size; ++ii) {
				for(unsigned int jj=0; jj<changes_size; ++jj)
					fprintf(flog, "%.3f ", tmpchanges->at(ii,jj));
				fprintf(flog, "\n");
			}
			#endif
			delete tmpchanges,
			delete [] idx;
			delete [] sortedlabels;
			return reschanges;
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
