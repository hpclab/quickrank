#ifndef __RT_HPP__
#define __RT_HPP__

#include <cfloat>
#include <cmath>
#include <cstring>

#include "learning/dpset.hpp"
#include "learning/tree/rtnode.hpp"
#include "learning/tree/histogram.hpp"
#include "utils/maxheap.hpp"

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
	private:
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
		//improve_parentdeviance is set to false in RankLib
		bool split(rtnode *node, bool improve_parentdeviance=false) {
			#ifdef LOGFILE
			static int icounter = 0;
			#endif
			if(node->deviance>0.0f) {
				splitparams *bs = node->hist->find_bestsplit(minls, improve_parentdeviance?node->deviance:FLT_MAX);
				if(bs) {
					//split samples between left and right child
					unsigned int *leftsamples = new unsigned int[bs->lcount], lsize = 0;
					unsigned int *rightsamples = new unsigned int[bs->rcount], rsize = 0;
					const float best_threshold = bs->best_threshold;
					float const* features = training_set->get_fvector(bs->best_featureid);
					for(unsigned int i=0, nsampleids=node->nsampleids; i<nsampleids; ++i) {
						unsigned int k = node->sampleids[i];
						if(features[k]<=best_threshold) leftsamples[lsize++] = k; else rightsamples[rsize++] = k;
					}
					//create histograms for node's children
					temphistogram *lhist = new temphistogram(node->hist, leftsamples, lsize, training_labels);
					temphistogram *rhist = new temphistogram(node->hist, lhist);
					//update current node
					node->featureid = bs->best_featureid,
					node->threshold = best_threshold,
					node->deviance = bs->deviance,
					node->left = new rtnode(leftsamples, lsize, bs->lvar, bs->lsum, lhist),
					node->right = new rtnode(rightsamples, rsize, bs->rvar, bs->rsum, rhist);
					#ifdef LOGFILE
					fprintf(flog,"SPLIT #%d (%.12f)\n", ++icounter, node->deviance);
					for(unsigned int i=0; i<lsize; ++i)	fprintf(flog,"%u ", leftsamples[i]);
					fprintf(flog,"| ");
					for(unsigned int i=0; i<rsize; ++i)	fprintf(flog,"%u ", rightsamples[i]);
					fprintf(flog,"\nVl=%f %f | Vr=%f %f\n%f %u\n", bs->lvar, bs->lsum, bs->rvar, bs->rsum, best_threshold, bs->best_featureid);
					#endif
					//free memory
					delete bs;
					return true;
				}
			}
			#ifdef LOGFILE
			fprintf(flog,"SPLIT #%d: UNSPLITABLE\n", ++icounter);
			#endif
			return false;
		}
};

#endif
