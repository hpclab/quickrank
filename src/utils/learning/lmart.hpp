#ifndef __LMART_HPP__
#define __LMART_HPP__

#include <cfloat>
#include <cmath>

#include "learning/ranker.hpp"
#include "learning/tree/rt.hpp"
#include "learning/tree/ensemble.hpp"
#include "utils/cpuinfo.hpp" // info from /proc/cpuinfo

class lmartranker : public ranker {
	private:
		unsigned int ntrees = 10; //the number of trees
		float learningrate = 0.1f; //or shrinkage
		unsigned int nthreshold = 0xFFFFFFFF;
		unsigned int ntreeleaves = 10;
		unsigned int minleafsupport = 1;
		float **thresholds = NULL;
		unsigned int *thresholds_size = NULL;
		float *modelscores = NULL; //[0..nentries-1]
		float *validationmodelscores = NULL; //[0..nentries-1]
		unsigned int validation_bestmodel = 0xFFFFFFFF;
		float *pseudoresponses = NULL;  //[0..nentries-1]
		unsigned int **sortedsid = NULL;
		unsigned int *sortedsize = NULL;
		permhistogram *hist = NULL;
		float *cachedweights = NULL; //corresponds to datapoint.cache
		unsigned int eenrounds = 0; //If no performance gain on validation data is observed in eerounds, stop the training process right away (if eenrounds==0 feature is disabled).
		ensemble ens;
	public:
		lmartranker() {};
		lmartranker(dpset *training_set) : ranker(training_set) {}
		~lmartranker() {
			const unsigned int nfeatures = training_set->get_nfeatures();
			for(unsigned int i=0; i<nfeatures; ++i)
				delete [] sortedsid[i],
				free(thresholds[i]);
			delete [] thresholds,
			delete [] thresholds_size,
			delete [] modelscores,
			delete [] validationmodelscores,
			delete [] pseudoresponses,
			delete [] sortedsid,
			delete [] sortedsize,
			delete [] cachedweights;
			delete hist;
		}
		const char *whoami() const {
			return "LAMBDA MART";
		}
		void init()  {
			if(nthreshold==0xFFFFFFFF) printf(">>> INIT:\n\tno. of thresholds has no limits\n");
			else printf(">>> INIT:\n\tno. of threshold candidates = %u\n", nthreshold);
			#ifdef SHOWTIMER
			double timer = omp_get_wtime();
			#endif
			const unsigned int nentries = training_set->get_ndatapoints();
			modelscores = new float[nentries]();  //0.0f initialized
			pseudoresponses = new float[nentries](); //0.0f initialized
			cachedweights = new float[nentries](); //0.0f initialized
			const unsigned int nfeatures = training_set->get_nfeatures();
			sortedsid = new unsigned int*[nfeatures],
			sortedsize = new unsigned int[nfeatures];
			#pragma omp parallel for
			for(unsigned int i=0; i<nfeatures; ++i)
				training_set->sort_dpbyfeature(i, sortedsid[i], sortedsize[i]);
			//for each featureid, init threshold array by keeping track of the list of "unique values" and their max, min
			thresholds = new float*[nfeatures],
			thresholds_size = new unsigned int[nfeatures];
			#pragma omp parallel for
			for(unsigned int i=0; i<nfeatures; ++i) {
				//select feature array realted to the current feature index
				float const* features = training_set->get_fvector(i);
				//init with values with the 1st sample
				unsigned int *idx = sortedsid[i];
				unsigned int idxsize = sortedsize[i];
				//get_ sample indexes sorted by the fid-th feature
				unsigned int uniqs_size = 0;
				float *uniqs = (float*)malloc(sizeof(float)*(nthreshold==0xFFFFFFFF?idxsize+1:nthreshold+1));
				//skip samples with the same feature value. early exit for if nthreshold!=size_max
				uniqs[uniqs_size++] = features[idx[0]];
				for(unsigned int j=1; j<idxsize && (nthreshold==0xFFFFFFFF || uniqs_size!=nthreshold+1); ++j) {
					const float fval = features[idx[j]];
					if(uniqs[uniqs_size-1]<fval) uniqs[uniqs_size++] = fval;
				}
				//define thresholds
				if(uniqs_size<=nthreshold || nthreshold==0xFFFFFFFF) {
					uniqs[uniqs_size++] = FLT_MAX;
					thresholds_size[i] = uniqs_size,
					thresholds[i] = (float*)realloc(uniqs, sizeof(float)*uniqs_size);
				} else {
					free(uniqs),
					thresholds_size[i] = nthreshold+1,
					thresholds[i] = (float*)malloc(sizeof(float)*(nthreshold+1));
					float t = features[idx[0]]; //equals fmin
					const float step = fabs(features[idx[idxsize-1]]-t)/nthreshold; //(fmax-fmin)/nthreshold
					for(unsigned int j=0; j!=nthreshold; t+=step)
						thresholds[i][j++] = t;
					thresholds[i][nthreshold] = FLT_MAX;
				}
			}
			if(validation_set) {
				unsigned int ndatapoints = validation_set->get_ndatapoints();
				validationmodelscores = new float[ndatapoints]();
			}
			#ifdef LOGFILE
			for(unsigned int i=0; i<nfeatures; ++i) {
				for(unsigned int j=0; j<sortedsize[i]; ++j)
					fprintf(flog, "%u ", sortedsid[i][j]);
				fprintf(flog, "\n");
			}
			for(unsigned int i=0; i<nfeatures; ++i) {
				for(unsigned int j=0; j<thresholds_size[i]-1; ++j)
					fprintf(flog, "%f ", thresholds[i][j]);
				fprintf(flog, "\n");
			}
			#endif
			hist = new permhistogram(training_set, pseudoresponses, sortedsid, sortedsize, thresholds, thresholds_size);
			#ifdef SHOWTIMER
			printf("\telapsed time for init = %.3f seconds\n", omp_get_wtime()-timer);
			#endif
		}
		void learn() {
			training_score = 0.0f,
			validation_bestscore = 0.0f;
			printf(">>> TRAINING:\n\tno. of tree(s) = %u\n\tlearning rate = %f\n\tno. of tree leaves = %u\n\tmin no. of leaves = %u\n", ntrees, learningrate, ntreeleaves, minleafsupport);
			if(eenrounds>0) printf("\tstop training if no gain is obtained for %u consecutive round(s)\n", eenrounds);
			printf("\n\titeration training-score validation-score validation-gain\n\t---------------------------------------------------------\n");
			#ifdef SHOWTIMER
			#define NTIMERS 2
			double timervalues[NTIMERS];
			unsigned int timercounter = 0;
			for(int i=0; i<NTIMERS; ++i) timervalues[i] = 0.0;
			const char *timerlabels[NTIMERS] = {"regression tree computation", "model evaluation"};
			#endif
			for(unsigned int m=0; m<ntrees && (eenrounds==0 || m<=validation_bestmodel+eenrounds); ++m) {
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
				float maxlabel = update_treeoutput(tree.leaves, tree.nleaves);
				//add this tree to the ensemble (our model)
				ens.push(tree.root, learningrate, maxlabel);
				//Update the model's outputs on all training samples
				unsigned int ndatapoints = training_set->get_ndatapoints();
				float **featurematrix = training_set->get_fmatrix();
				#pragma omp parallel for
				for(unsigned int i=0; i<ndatapoints; ++i)
					modelscores[i] += learningrate*tree.root->eval(featurematrix, i);
				#ifdef SHOWTIMER
				timervalues[1] -= omp_get_wtime();
				#endif
				//Evaluate the current model
				training_score = compute_trainingscore();
				#ifdef SHOWTIMER
				timervalues[1] += omp_get_wtime();
				#endif
				printf("\t%-9u %-14f", m+1, training_score);
				//Evaluate the current model on the validation data (if available)
				if(validation_set) {
					unsigned int ndatapoints = validation_set->get_ndatapoints();
					float **featurematrix = validation_set->get_fmatrix();
					#pragma omp parallel for
					for(unsigned int i=0; i<ndatapoints; ++i)
						validationmodelscores[i] += learningrate*tree.root->eval(featurematrix, i);
					float validation_score = compute_validationmodelscores();
					printf(" %-14f", validation_score);
					float delta = validation_score-validation_bestscore;
					if(delta>FLT_EPSILON || validation_bestmodel==0xFFFFFFFF)
						validation_bestscore = validation_score,
						validation_bestmodel = ens.get_size()-1,
						printf("   %+-14f", delta);
				}
				printf("\n");
			}
			//Rollback to the best model observed on the validation data
			while(ens.is_notempty() && ens.get_size()>validation_bestmodel+1)
				ens.pop();
			//Finishing up
			training_score = compute_score(training_set, scorer); //scorer.score(rank(samples));
			printf("\t---------------------------------------------------------\n\t%s@%u on training data = %f\n", scorer->whoami(), scorer->get_k(), training_score);
			if(validation_set) {
				validation_bestscore = compute_score(validation_set, scorer); //scorer.score(rank(validationSamples));
				printf("\t%s@%u on validation data = %f\n", scorer->whoami(), scorer->get_k(), validation_bestscore);
			}
			#ifdef SHOWTIMER
			printf("\n");
			for(int i=0; i<NTIMERS; ++i)
				printf("\tavg '%s' elapsed time = %.3f seconds\n", timerlabels[i], timervalues[i]/timercounter);
			#undef NTIMERS
			#endif
		}
		float eval_dp(float *const *const features, unsigned int idx) const {
			return ens.eval(features, idx);
		}
		void write_outputtofile(const char *filename) {
			FILE *f = fopen(filename, "w");
			if(f) {
				fprintf(f, "## no. of tree(s) = %u\n## learning rate = %f\n## no. of tree leaves = %u\n## min no. of leaves = %u\n", ntrees, learningrate, ntreeleaves, minleafsupport);
				if(eenrounds>0)
					fprintf(f, "## stop training if no gain is obtained for %u consecutive round(s)\n", eenrounds);
				ens.write_outputtofile(f);
				fclose(f);
			}
		}
	protected:
		float compute_validationmodelscores() {
			float score = 0.0f;
			unsigned int nrankedlists = validation_set->get_nrankedlists();
			unsigned int *rloffsets = validation_set->get_rloffsets();
			#pragma omp parallel for reduction(+:score)
			for(unsigned int i=0; i<nrankedlists; ++i) {
				rnklst orig = validation_set->get_ranklist(i);
				float *sortedlabels = copyextfloat_radixsort<descending>(orig.labels, validationmodelscores+rloffsets[i], orig.size);
				score += scorer->compute_score(rnklst(orig.size, sortedlabels, orig.id));
				delete[] sortedlabels;
			}
			return nrankedlists ? score/nrankedlists : 0.0f;
		}
		void compute_pseudoresponses() {
			const unsigned int nrankedlists = training_set->get_nrankedlists();
			const unsigned int *rloffsets = training_set->get_rloffsets();
			#pragma omp parallel for
			for(unsigned int i=0; i<nrankedlists; ++i) {
				const unsigned int offset = rloffsets[i];
				rnklst rl = training_set->get_ranklist(i);
				fsymmatrix *changes = compute_mchange(rl, offset);
				float *lambdas = pseudoresponses+offset;
				float *weights = cachedweights+offset;
				for(unsigned int j=0; j<rl.size; ++j)
					lambdas[j] = 0.0f,
					weights[j] = 0.0f;
				for(unsigned int j=0; j<rl.size; ++j) {
					float jthlabel = rl.labels[j];
					for(unsigned int k=0; k<rl.size; ++k) if(k!=j) {
						float kthlabel = rl.labels[k];
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
			const unsigned int nentries = training_set->get_ndatapoints();
			for(unsigned int i=0; i<nentries; ++i)
				fprintf(flog,"%f ", pseudoresponses[i]);
			fprintf(flog,"\n");
			#endif
		}
		float update_treeoutput(rtnode **leaves, const unsigned int nleaves) {
			float maxlabel = NAN;
			#pragma omp parallel for reduction(max:maxlabel)
			for(unsigned int i=0; i<nleaves; ++i) {
				float s1 = 0.0f;
				float s2 = 0.0f;
				const unsigned int *sampleids = leaves[i]->sampleids;
				const unsigned int nsampleids = leaves[i]->nsampleids;
				for(unsigned int j=0; j<nsampleids; ++j) {
					unsigned int k = sampleids[j];
					s1 += pseudoresponses[k],
					s2 += cachedweights[k];
				}
				float s = s1/s2;
				leaves[i]->avglabel = s;
				if(s>maxlabel)
					maxlabel = s;
			}
			#ifdef LOGFILE
			fprintf(flog, "\nleaves:\n");
			for(unsigned int i=0; i<nleaves; ++i)
				fprintf(flog, "%f ", leaves[i]->avglabel);
			fprintf(flog, "(%u)\n", nleaves);
			#endif
			return maxlabel;
		}
		float compute_trainingscore() {
			unsigned int nrankedlists = training_set->get_nrankedlists();
			unsigned int *offsets = training_set->get_rloffsets();
			float avg = 0.0f;
			if(nrankedlists) {
				#pragma omp parallel for reduction(+:avg)
				for(unsigned int i=0; i<nrankedlists; ++i) {
					rnklst orig = training_set->get_ranklist(i);
					float *sortedlabels = copyextfloat_radixsort<descending>(orig.labels, modelscores+offsets[i], orig.size);
					avg += scorer->compute_score(rnklst(orig.size, sortedlabels, orig.id));
					delete [] sortedlabels;
				}
				avg /= nrankedlists;
			}
			return avg;
		}
		fsymmatrix *compute_mchange(const rnklst &orig, const unsigned int offset) {
			//build a rl made up of label-values picked up from orig order by indexes of modelscores reversely sorted
			unsigned int *idx = idxfloat_radixsort<descending>(modelscores+offset, orig.size);
			float sortedlabels[orig.size];
			for(unsigned int i=0; i<orig.size; ++i)
				sortedlabels[i] = orig.labels[idx[i]];
			rnklst tmprl(orig.size, sortedlabels, orig.id);
			//alloc mem; NOTE re-implement symmetric matrix
			fsymmatrix *changes = new fsymmatrix(orig.size);
			//compute temp swap changes on rl
			fsymmatrix *tmpchanges = scorer->swap_change(tmprl);
			#pragma omp parallel for
			for(unsigned int i=0; i<orig.size; ++i)
				for(unsigned int j=0; j<orig.size; ++j)
					changes->at(idx[i],idx[j]) = tmpchanges->at(i,j);
			delete tmpchanges,
			delete [] idx;
			return changes;
		}
};

#endif
