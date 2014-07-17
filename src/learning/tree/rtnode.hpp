#ifndef __RTNODE_HPP__
#define __RTNODE_HPP__

#include "learning/dpset.hpp"
#include "learning/tree/histogram.hpp"

class rtnode {
	public:
		unsigned int *sampleids = NULL;
		unsigned int nsampleids = 0;
		unsigned int featureid = 0xFFFFFFFF;
		float threshold = 0.0f;
		double deviance = 0.0f;
		double avglabel = 0.0f;
		rtnode *left = NULL;
		rtnode *right = NULL;
		histogram *hist = NULL;
	private:
		void enum_leaves(rtnode **&arr, unsigned int &size, unsigned int &maxsize) {
			if(featureid==0xFFFFFFFF) {
				if(size==maxsize) {
					maxsize = 2*maxsize+1;
					arr = (rtnode**)realloc(arr, sizeof(rtnode*)*maxsize);
				}
				arr[size++] = this;
			} else {
				left->enum_leaves(arr, size, maxsize);
				right->enum_leaves(arr, size, maxsize);
			}
		}
	public:
		rtnode(unsigned int *sampleids, unsigned int nsampleids, double deviance, double sumlabel, histogram* hist) :
			sampleids(sampleids), nsampleids(nsampleids), deviance(deviance), hist(hist) {
			avglabel = sumlabel/nsampleids;
		}
		~rtnode() {
			delete [] sampleids,
			delete left,
			delete right;
		}
		rtnode** get_leaves(unsigned int &nleaves, const unsigned int initsize=0) {
			rtnode** leaves = initsize>0 ? (rtnode**)malloc(sizeof(rtnode*)*initsize) : NULL;
			unsigned int maxsize = initsize;
			nleaves = 0;
			enum_leaves(leaves, nleaves, maxsize);
			return (rtnode**)realloc(leaves, sizeof(rtnode*)*nleaves);
		}
		float eval(float const* const* featurematrix, const unsigned int idx) const {//prediction
			if(featureid==0xFFFFFFFF) return avglabel;
			return featurematrix[featureid][idx]<=threshold ? left->eval(featurematrix, idx) : right->eval(featurematrix, idx);
		}
		void write_outputtofile(FILE *f, const int indentsize) {
			char indent[indentsize+1];
			for(int i=0; i<indentsize; indent[i++]='\t');
			indent[indentsize] = '\0';
			if(featureid==0xFFFFFFFF)
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
