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
		virtual void fit(histogram *hist) = 0;
};

#endif
