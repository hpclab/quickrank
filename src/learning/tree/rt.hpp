#ifndef __RT_HPP__
#define __RT_HPP__

#include <cfloat>
#include <cmath>
#include <cstring>
#include <omp.h>

#include "utils/maxheap.hpp"
#include "learning/dpset.hpp"
#include "learning/tree/rtnode.hpp"
#include "learning/tree/histogram.hpp"

typedef maxheap<rtnode*> rt_maxheap;

class deviance_maxheap : public rt_maxheap {
	public:
		deviance_maxheap(unsigned int initsize) : rt_maxheap(initsize) {}
		void push_chidrenof(rtnode *parent) {
			push(parent->left->deviance, parent->left);
			push(parent->right->deviance, parent->right);
		}
		void pop() {
			rtnode *node = top();
			delete [] node->sampleids,
			delete node->hist;
			node->sampleids = NULL,
			node->nsampleids = 0,
			node->hist = NULL;
			rt_maxheap::pop();
		}
};

class rt {
	protected:
		const unsigned int nrequiredleaves; //0 for unlimited number of nodes (the size of the tree will then be controlled only by minls)
		const unsigned int minls; //minls>0
		dpset *training_set = NULL;
		float *training_labels = NULL;
		rtnode **leaves = NULL;
		unsigned int nleaves = 0;
		rtnode *root = NULL;
	public:
		rt(unsigned int nrequiredleaves, dpset *dps, float *labels, unsigned int minls) :
			nrequiredleaves(nrequiredleaves),
			minls(minls),
			training_set(dps),
			training_labels(labels) {
		}
		~rt() {
			if(root) {
				delete [] root->sampleids;
				root->sampleids = NULL,
				root->nsampleids = 0;
			}
			//if leaves[0] is the root, hist cannot be deallocated and sampleids has been already deallocated
			for(unsigned int i=0; i<nleaves; ++i)
				if(leaves[i]!=root) {
					delete [] leaves[i]->sampleids,
					delete leaves[i]->hist;
					leaves[i]->hist = NULL,
					leaves[i]->sampleids = NULL,
					leaves[i]->nsampleids = 0;
				}
			free(leaves);
		}
		void fit(histogram *hist) {
			deviance_maxheap heap(nrequiredleaves);
			unsigned int taken = 0;
			unsigned int nsampleids = training_set->get_ndatapoints();
			unsigned int *sampleids = new unsigned int[nsampleids];
			#pragma omp parallel for
			for(unsigned int i=0; i<nsampleids; ++i)
				sampleids[i] = i;
			root = new rtnode(sampleids, nsampleids, DBL_MAX, 0.0, hist);
			if(split(root))
				heap.push_chidrenof(root);
			while(heap.is_notempty() && (nrequiredleaves==0 or taken+heap.get_size()<nrequiredleaves)) {
				//get node with highest deviance from heap
				rtnode *node = heap.top();
				//try split current node
				if(split(node)) heap.push_chidrenof(node); else ++taken; //unsplitable (i.e. null variance, or after split variance is higher than before, or #samples<minlsd)
				//remove node from heap
				heap.pop();
			}
			//visit tree and save leaves in a leaves[] array
			unsigned int capacity = nrequiredleaves;
			leaves = capacity ? (rtnode**)malloc(sizeof(rtnode*)*capacity) : NULL,
			nleaves = 0;
			root->save_leaves(leaves, nleaves, capacity);
		}
		float update_output(float const *pseudoresponses, float const *cachedweights) {
			float maxlabel = -FLT_MAX;
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
		float eval(float const* const* featurematrix, const unsigned int idx) const {
			return root->eval(featurematrix, idx);
		}
		rtnode *get_proot() const {
			return root;
		}
	private:
		//if require_devianceltparent is true the node is split if minvar is lt the current node deviance (require_devianceltparent=false in RankLib)
		bool split(rtnode *node, bool require_devianceltparent=false) {
			if(node->deviance>0.0f) {
				const double initvar = require_devianceltparent?node->deviance:DBL_MAX;
				//get current nod hidtogram pointer
				histogram *h = node->hist;
				//featureidxs to be used for tree splitnodeting
				unsigned int featuresamples[h->nfeatures];
				unsigned int nfeaturesamples = h->nfeatures;
				for(unsigned int i=0; i<nfeaturesamples; ++i)
					featuresamples[i] = i;
				if(h->samplingrate<1.0f) {
					//need to make a sub-sampling
					unsigned int reduced_nfeaturesamples = (unsigned int)floor(h->samplingrate*nfeaturesamples);
					while(nfeaturesamples>reduced_nfeaturesamples && nfeaturesamples>1) {
						unsigned int selectedtoremove = rand()%nfeaturesamples;
						featuresamples[selectedtoremove] = featuresamples[--nfeaturesamples];
					}
				}
				//find best split
				const int nth = omp_get_num_procs();
				double thread_minvar[nth];
				double thread_best_lvar[nth];
				double thread_best_rvar[nth];
				unsigned int thread_best_featureid[nth];
				unsigned int thread_best_thresholdid[nth];
				for(int i=0; i<nth; ++i)
					thread_minvar[i] = initvar,
					thread_best_lvar[i] = 0.0,
					thread_best_rvar[i] = 0.0,
					thread_best_featureid[i] = uint_max,
					thread_best_thresholdid[i] = uint_max;
				#pragma omp parallel for
				for(unsigned int i=0; i<nfeaturesamples; ++i) {
					//get thread identification number
					const int ith = omp_get_thread_num();
					//get feature id
					const unsigned int f = featuresamples[i];
					//define pointer shortcuts
					double *sumlabels = h->sumlbl[f];
					double *sqsumlabels = h->sqsumlbl[f];
					unsigned int *samplecount = h->count[f];
					//get last elements
					unsigned int threshold_size = h->thresholds_size[f];
					double s = sumlabels[threshold_size-1];
					double sq = sqsumlabels[threshold_size-1];
					unsigned int c = samplecount[threshold_size-1];
					//looking for the feature that minimizes sumvar
					for(unsigned int t=0; t<threshold_size; ++t) {
						unsigned int lcount = samplecount[t];
						unsigned int rcount = c-lcount;
						if(lcount>=minls && rcount>=minls)  {
							double lsum = sumlabels[t];
							double lsqsum = sqsumlabels[t];
							double lvar = fabs(lsqsum-lsum*lsum/lcount);
							double rsum = s-lsum;
							double rsqsum = sq-lsqsum;
							double rvar = fabs(rsqsum-rsum*rsum/rcount);
							double sumvar = lvar+rvar;
							if(sumvar<thread_minvar[ith])
								thread_minvar[ith] = sumvar,
								thread_best_lvar[ith] = lvar,
								thread_best_rvar[ith] = rvar,
								thread_best_featureid[ith] = f,
								thread_best_thresholdid[ith] = t;
						}
					}
				}
				//get best minvar among thread partial results
				double minvar = thread_minvar[0];
				double best_lvar = thread_best_lvar[0];
				double best_rvar = thread_best_rvar[0];
				unsigned int best_featureid = thread_best_featureid[0];
				unsigned int best_thresholdid = thread_best_thresholdid[0];
				for(int i=1; i<nth; ++i)
					if(thread_minvar[i]<minvar)
						minvar = thread_minvar[i],
						best_lvar = thread_best_lvar[i],
						best_rvar = thread_best_rvar[i],
						best_featureid = thread_best_featureid[i],
						best_thresholdid = thread_best_thresholdid[i];
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
				histogram *lhist = new histogram(node->hist, lsamples, lsize, training_labels);
				histogram *rhist = new histogram(node->hist, lhist);
				//update current node
				node->featureid = best_featureid,
				node->threshold = best_threshold,
				node->deviance = minvar,
				//create children
				node->left = new rtnode(lsamples, lsize, best_lvar, lsum, lhist),
				node->right = new rtnode(rsamples, rsize, best_rvar, rsum, rhist);
				#ifdef LOGFILE
				fprintf(flog,"SPLIT (dev=%.4f) minvar=%.4f th*=%.4f fid*=%u\n\t", node->deviance, minvar, best_threshold, best_featureid);
				for(unsigned int i=0; i<lsize; ++i)	fprintf(flog,"%u ", lsamples[i]);
				fprintf(flog,"| ");
				for(unsigned int i=0; i<rsize; ++i)	fprintf(flog,"%u ", rsamples[i]);
				fprintf(flog,"\n\tvl=%.4f %.4f | vr=%.4f %.4f\n", best_lvar, lsum, best_rvar, rsum);
				#endif
				return true;
			}
			return false;
		}
};

#endif
