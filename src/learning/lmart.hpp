#ifndef __LMART_HPP__
#define __LMART_HPP__

#include <cfloat>
#include <cmath>

#include "learning/ranker.hpp"
#include "learning/tree/rt.hpp"
#include "learning/tree/ensemble.hpp"
#include "utils/qsort.hpp"

class lmartrt : public rt {
	public:
		lmartrt(unsigned int nodes, dpset *dps, float *labels, unsigned int minls) : rt(nodes, dps, labels, minls) {}
		void fit(histogram *hist) {
			deviance_maxheap heap;
			unsigned int taken = 0;
			unsigned int nsampleids = training_set->get_ndatapoints();
			unsigned int *sampleids = new unsigned int[nsampleids];
			for(unsigned int i=0; i<nsampleids; ++i)
				sampleids[i] = i;
			root = new rtnode(sampleids, nsampleids, FLT_MAX, 0.0f, hist);
			if(split(root))
				heap.push_chidrenof(root);
			while(heap.is_notempty() && (nodes==0xFFFFFFFF or taken+heap.get_size()<nodes)) {
				//get node with max deviance from heap
				rtnode *node = heap.top();
				//try split
				if(split(node)) heap.push_chidrenof(node);
				else ++taken; //unsplitable (i.e. null variance, or after split variance is higher than before, or #samples<minlsd)
				//remove node from heap
				heap.pop();
			}
			leaves = root->get_leaves(nleaves);
		}
	private:
		//if true the node is splitable if new variance is lt the current node deviance (require_devianceltparent=false in RankLib)
		bool split(rtnode *node, bool require_devianceltparent=false) {
			#ifdef LOGFILE
			static int icounter = 0;
			#endif
			if(node->deviance>0.0f) {
				const float initvar = require_devianceltparent?node->deviance:FLT_MAX;
				//get current nod hidtogram pointer
				histogram *h = node->hist;
				//featureidxs to be used for tree splitnodeting
				unsigned int featuresamples[h->nfeatures];
				unsigned int nfeaturesamples = 0;
				for(unsigned int i=0; i<h->nfeatures; ++i)
					featuresamples[nfeaturesamples++] = i;
				if(h->samplingrate<1.0f) {
					//need to make a sub-sampling
					unsigned int reduced_nfeaturesamples = (unsigned int)floor(h->samplingrate*nfeaturesamples);
					while(nfeaturesamples>reduced_nfeaturesamples && nfeaturesamples>1) {
						unsigned int selectedtoremove = rand()%nfeaturesamples;
						featuresamples[selectedtoremove] = featuresamples[--nfeaturesamples];
					}
				}
				//find best split
				unsigned int best_featureid = 0xFFFFFFFF;
				unsigned int best_thresholdid = 0xFFFFFFFF;
				float best_lvar = 0.0f;
				float best_rvar = 0.0f;
				float minvar = initvar;
				for(unsigned int i=0; i<nfeaturesamples; ++i) {
					const unsigned int f = featuresamples[i];
					//define pointer shortcuts
					float *sumlabels = h->sumlbl[f];
					float *sqsumlabels = h->sqsumlbl[f];
					unsigned int *samplecount = h->count[f];
					//get last elements
					unsigned int threshold_size = h->thresholds_size[f];
					float s = sumlabels[threshold_size-1];
					float sq = sqsumlabels[threshold_size-1];
					unsigned int c = samplecount[threshold_size-1];
					//looking for the feature that minimizes sum of lvar+rvar
					for(unsigned int t=0; t<threshold_size; ++t) {
						unsigned int lcount = samplecount[t];
						unsigned int rcount = c-lcount;
						if(lcount>=minls && rcount>=minls)  {
							float lsum = sumlabels[t];
							float lsqsum = sqsumlabels[t];
							float lvar = fabs(lsqsum-lsum*lsum/lcount);
							float rsum = s-lsum;
							float rsqsum = sq-lsqsum;
							float rvar = fabs(rsqsum-rsum*rsum/rcount);
							float sumvar = lvar+rvar;
							if(FLT_EPSILON+sumvar<minvar) //is required an improvement gt FLT_EPSILON wrt current minvar
								minvar = sumvar,
								best_lvar = lvar,
								best_rvar = rvar,
								best_featureid = f,
								best_thresholdid = t;
						}
					}
				}
				//if minvar is the same of initvalue then the node is unsplitable
				if(minvar==initvar)
					return false;
				//set some result values related to minvar
				const unsigned int last_thresholdidx = h->thresholds_size[best_featureid]-1;
				const float best_threshold = h->thresholds[best_featureid][best_thresholdid];
				const float lsum = h->sumlbl[best_featureid][best_thresholdid];
				const float rsum = h->sumlbl[best_featureid][last_thresholdidx]-lsum;
				const unsigned int lcount = h->count[best_featureid][best_thresholdid];
				const unsigned int rcount = h->count[best_featureid][last_thresholdidx]-lcount;
				//split samples between left and right child
				unsigned int *lsamples = new unsigned int[lcount], lsize = 0;
				unsigned int *rsamples = new unsigned int[rcount], rsize = 0;
				float const* features = training_set->get_fvector(best_featureid);
				for(unsigned int i=0, nsampleids=node->nsampleids; i<nsampleids; ++i) {
					unsigned int k = node->sampleids[i];
					if(features[k]<=best_threshold) lsamples[lsize++] = k; else rsamples[rsize++] = k;
				}
				//create histograms for children
				temphistogram *lhist = new temphistogram(node->hist, lsamples, lsize, training_labels);
				temphistogram *rhist = new temphistogram(node->hist, lhist);
				//update current node
				node->featureid = best_featureid,
				node->threshold = best_threshold,
				node->deviance = minvar,
				//create children
				node->left = new rtnode(lsamples, lsize, best_lvar, lsum, lhist),
				node->right = new rtnode(rsamples, rsize, best_rvar, rsum, rhist);
				#ifdef LOGFILE
				fprintf(flog,"SPLIT #%d (%.12f)\n", ++icounter, node->deviance);
				for(unsigned int i=0; i<lsize; ++i)	fprintf(flog,"%u ", lsamples[i]);
				fprintf(flog,"| ");
				for(unsigned int i=0; i<rsize; ++i)	fprintf(flog,"%u ", rsamples[i]);
				fprintf(flog,"\nVl=%f %f | Vr=%f %f\n%f %u\n", best_lvar, lsum, best_rvar, rsum, best_threshold, best_featureid);
				#endif
				return true;
			}
			#ifdef LOGFILE
			fprintf(flog,"SPLIT #%d: UNSPLITABLE\n", ++icounter);
			#endif
			return false;
		}
};

class lmartranker : public ranker {
	private:
		unsigned int ntrees = 30; //the number of trees
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
		unsigned int sortedsize = 0;
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
				float *uniqs = (float*)malloc(sizeof(float)*(nthreshold==0xFFFFFFFF?sortedsize+1:nthreshold+1));
				//skip samples with the same feature value. early exit for if nthreshold!=size_max
				uniqs[uniqs_size++] = features[idx[0]];
				for(unsigned int j=1; j<sortedsize && (nthreshold==0xFFFFFFFF || uniqs_size!=nthreshold+1); ++j) {
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
					const float step = fabs(features[idx[sortedsize-1]]-t)/nthreshold; //(fmax-fmin)/nthreshold
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
				lmartrt tree(ntreeleaves, training_set, pseudoresponses, minleafsupport);
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
			unsigned int *offsets = validation_set->get_rloffsets();
			#pragma omp parallel for reduction(+:score)
			for(unsigned int i=0; i<nrankedlists; ++i) {
				rnklst orig = validation_set->get_ranklist(i);
				float *sortedlabels = copyextfloat_qsort(orig.labels, validationmodelscores+offsets[i], orig.size);
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
					float *sortedlabels = copyextfloat_qsort(orig.labels, modelscores+offsets[i], orig.size);
					avg += scorer->compute_score(rnklst(orig.size, sortedlabels, orig.id));
					delete [] sortedlabels;
				}
				avg /= nrankedlists;
			}
			return avg;
		}
		fsymmatrix *compute_mchange(const rnklst &orig, const unsigned int offset) {
			//build a rl made up of label-values picked up from orig order by indexes of modelscores reversely sorted
			unsigned int *idx = idxfloat_qsort(modelscores+offset, orig.size);
			float sortedlabels[orig.size];
			for(unsigned int i=0; i<orig.size; ++i)
				sortedlabels[i] = orig.labels[idx[i]];
			rnklst tmprl(orig.size, sortedlabels, orig.id);
			//alloc mem
			fsymmatrix *reschanges = new fsymmatrix(orig.size);
			//compute temp swap changes on rl
			fsymmatrix *tmpchanges = scorer->swap_change(tmprl);
			#pragma omp parallel for
			for(unsigned int i=0; i<orig.size; ++i)
				for(unsigned int j=i; j<orig.size; ++j)
					reschanges->at(idx[i],idx[j]) = tmpchanges->at(i,j);
			delete tmpchanges,
			delete [] idx;
			return reschanges;
		}
};

#endif
