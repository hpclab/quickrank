#ifndef __RT_HPP__
#define __RT_HPP__

#include <cfloat>
#include <cmath>
#include <cstring>

#include "utils/maxheap.hpp"
#include "learning/dpset.hpp"
#include "learning/tree/rtnode.hpp"
#include "learning/tree/histogram.hpp"

class deviance_maxheap : public maxheap<rtnode*> {
	public:
		void push_chidrenof(rtnode *parent) {
			push(parent->left->deviance, parent->left);
			push(parent->right->deviance, parent->right);
		}
		void pop() {
			delete top()->hist;
			top()->hist = NULL;
			maxheap<rtnode*>::pop();
		}
		~deviance_maxheap() {
			for(size_t i=1; i<=arrsize; ++i)
				delete arr[i].val->hist;
		}
};

class rt {
	protected:
		unsigned int nodes = 10; //0xFFFFFFFF for unlimited number of nodes (the size of the tree will then be controlled only by minls)
		unsigned int minls = 1; //minls>0
		dpset *training_set = NULL;
		float *training_labels = NULL;
	public:
		rtnode *root = NULL;
		rtnode **leaves = NULL;
		unsigned int nleaves = 0;
	public:
		rt(unsigned int nodes, dpset *dps, float *labels, unsigned int minls) :
			nodes(nodes),
			minls(minls),
			training_set(dps),
			training_labels(labels) {
		}
		~rt() {
			for(unsigned int i=0; i<nleaves; ++i) {
				delete [] leaves[i]->sampleids;
				leaves[i]->sampleids = NULL,
				leaves[i]->nsampleids = 0;
			}
			free(leaves);
		}
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
							#ifdef LOGFILE
							fprintf(flog,"%.4f f=%u t=%u\n", sumvar, f, t);
							#endif
							if(sumvar<minvar)
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
				fprintf(flog,"SPLIT #%d (dev=%.4f) minvar=%.4f th*=%.4f fid*=%u\n\t", ++icounter, node->deviance, minvar, best_threshold, best_featureid);
				for(unsigned int i=0; i<lsize; ++i)	fprintf(flog,"%u ", lsamples[i]);
				fprintf(flog,"| ");
				for(unsigned int i=0; i<rsize; ++i)	fprintf(flog,"%u ", rsamples[i]);
				fprintf(flog,"\n\tvl=%.4f %.4f | vr=%.4f %.4f\n", best_lvar, lsum, best_rvar, rsum);
				#endif
				return true;
			}
			#ifdef LOGFILE
			fprintf(flog,"SPLIT #%d: UNSPLITABLE\n", ++icounter);
			#endif
			return false;
		}
};

#endif
