#ifndef __RTNODE_HPP__
#define __RTNODE_HPP__

#include "learning/dpset.hpp"
#include "learning/tree/histogram.hpp"

static const unsigned int uint_max = (unsigned int) -1;

class rtnode {
	public:
		unsigned int *sampleids = NULL;
		unsigned int nsampleids = 0;
		unsigned int featureid = uint_max;
		float threshold = 0.0f;
		double deviance = 0.0;
		double avglabel = 0.0;
		rtnode *left = NULL;
		rtnode *right = NULL;
		histogram *hist = NULL;
	public:
		rtnode(unsigned int *sampleids, unsigned int nsampleids, double deviance, double sumlabel, histogram* hist) :
			sampleids(sampleids), nsampleids(nsampleids), deviance(deviance), hist(hist) {
			avglabel = nsampleids ? sumlabel/nsampleids : 0.0;
		}
		~rtnode() {
			delete left,
			delete right;
		}
		void save_leaves(rtnode **&leaves, unsigned int &nleaves, unsigned int &capacity) {
			if(featureid==uint_max) {
				if(nleaves==capacity) {
					capacity = 2*capacity+1;
					leaves = (rtnode**)realloc(leaves, sizeof(rtnode*)*capacity);
				}
				leaves[nleaves++] = this;
			} else {
				left->save_leaves(leaves, nleaves, capacity);
				right->save_leaves(leaves, nleaves, capacity);
			}
		}
		bool is_leaf() const {
			return featureid==uint_max;
		}
		float eval(float const* const* featurematrix, const unsigned int idx) const {
			return featureid==uint_max ? avglabel : (featurematrix[featureid][idx]<=threshold ? left->eval(featurematrix, idx) : right->eval(featurematrix, idx));
		}
		void write_outputtofile(FILE *f, const int indentsize) {
			char indent[indentsize+1];
			for(int i=0; i<indentsize; indent[i++]='\t');
			indent[indentsize] = '\0';
			if(featureid==uint_max)
				fprintf(f, "%s\t<output> %.3f </output>\n", indent, avglabel);
			else {
				fprintf(f, "%s\t<feature> %u </feature>\n", indent, featureid);
				fprintf(f, "%s\t<threshold> %.3f </threshold>\n", indent, threshold);
				fprintf(f, "%s\t<split pos=\"left\">\n", indent);
				left->write_outputtofile(f, indentsize+1);
				fprintf(f, "%s\t</split>\n", indent);
				fprintf(f, "%s\t<split pos=\"right\">\n", indent);
				right->write_outputtofile(f, indentsize+1);
				fprintf(f, "%s\t</split>\n", indent);
			}
		}
};

#endif
