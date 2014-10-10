#ifndef QUICKRANK_LEARNING_TREE_RTNODE_H_
#define QUICKRANK_LEARNING_TREE_RTNODE_H_

#include "data/ltrdata.h"
#include "learning/tree/rtnode_histogram.h"

static const unsigned int uint_max = (unsigned int) -1;

class RTNode {
	public:
		unsigned int *sampleids = NULL;
		unsigned int nsampleids = 0;
		float threshold = 0.0f;
		double deviance = 0.0;
		double avglabel = 0.0;
		RTNode *left = NULL;
		RTNode *right = NULL;
		RTNodeHistogram *hist = NULL;
	private:
		unsigned int featureidx = uint_max; //refer the index in the feature matrix
		unsigned int featureid = uint_max; //refer to the id occurring in the dataset file
	public:
		RTNode(unsigned int *sampleids, unsigned int nsampleids, double deviance, double sumlabel, RTNodeHistogram* hist) :
			sampleids(sampleids), nsampleids(nsampleids), deviance(deviance), hist(hist) {
			avglabel = nsampleids ? sumlabel/nsampleids : 0.0;
		}
		~RTNode() {
			delete left,
			delete right;
		}
		void set_feature(unsigned int fidx, unsigned int fid) {
			//if(fidx==uint_max or fid==uint_max) exit(7);
			featureidx = fidx,
			featureid = fid;
		}
		unsigned int get_feature_id() {
			return featureid;
		}
		unsigned int get_feature_idx() {
			return featureidx;
		}

		void save_leaves(RTNode **&leaves, unsigned int &nleaves, unsigned int &capacity);

		bool is_leaf() const {
			return featureidx==uint_max;
		}
		double eval(float const* const* featurematrix, const unsigned int idx) const {
			return featureidx==uint_max ? avglabel : (featurematrix[featureidx][idx]<=threshold ? left->eval(featurematrix, idx) : right->eval(featurematrix, idx));
		}

		void write_outputtofile(FILE *f, const int indentsize);
};

#endif
