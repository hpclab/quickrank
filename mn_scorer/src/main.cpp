
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <omp.h>

#include "xmlparser.hpp"

class tree;

const unsigned int uint_max = (unsigned int) -1;

bool strisnan(const char *str) {
	return strcmp(str,"-nan")==0 or strcmp(str,"nan")==0 or strcmp(str,"-inf")==0 or strcmp(str,"inf")==0;
}

class tnode {
	public:
		tnode(xmlnode &split) {
			if(split.get_nchilds("output")!=0)
				output = strisnan(split.get_child("output").get_text()) ? 0.0f: atof(split.get_child("output").get_text());
			else
				featureid = atoi(split.get_child("feature").get_text()),
				threshold = atof(split.get_child("threshold").get_text());
			if(split.get_nchilds("split")==2) {
				xmlnode l = split.get_child("split", 0);
				left = new tnode(l);
				xmlnode r = split.get_child("split", 1);
				right = new tnode(r);
			}
		}
		~tnode() {
			delete left,
			delete right;
		}
		void show(const unsigned int n=1) {
			for(unsigned int i=0; i!=n; ++i) printf("\t");
			if(featureid!=uint_max)
				printf("featureid=%u threshold=%f\n", featureid, threshold);
			else
				printf("output=%f\n", output);
			if(left) left->show(n+1);
			if(right) right->show(n+1);
		}
	private:
		float output = 0.0f;
		float threshold = 0.0f;
		unsigned int featureid = uint_max;
		tnode *left = NULL;
		tnode *right = NULL;
	friend class tree;
};

class tree {
	public:
		tree(xmlnode &t) {
			id = atoi(t.get_attr("id")),
			weight = atof(t.get_attr("weight"));
			xmlnode split = t.get_child("split");
			root = new tnode(split);
		}
		~tree() {
			delete root;
		}
		void show() {
			printf("tree id=%u weight=%f\n", id, weight);
			if(root) root->show();
		}
		float get_weight() const {
			return weight;
		}
		float *get_output(int &size) {
			size = 0;
			int capacity = 0;
			float *output = NULL;
			get_nodeoutput(root, output, size, capacity);
			return output;
		}
		unsigned int *get_feautureids(int &size) {
			size = 0;
			int capacity = 0;
			unsigned int *feautureids = NULL;
			get_nodefeautureids(root, feautureids, size, capacity);
			return feautureids;
		}
		float *get_thresholds(int &size) {
			size = 0;
			int capacity = 0;
			float *output = NULL;
			get_nodethreshold(root, output, size, capacity);
			return output;
		}
		int get_depth() {
			int depth = 0;
			tnode *node = root;
			while(node)
				++depth,
				node = node->left;
			return depth;
		}
	private:
		unsigned int id = uint_max;
		float weight = 0.0f;
		tnode *root = NULL;
		void get_nodeoutput(tnode const *node, float *&arr, int &size, int &capacity) {
			if(node->featureid==uint_max) {
				if(size==capacity) {
					capacity = 2*size+1;
					arr = (float*)realloc(arr, sizeof(float)*capacity);
				}
				arr[size++] = node->output;
			} else {
				get_nodeoutput(node->left, arr, size, capacity);
				get_nodeoutput(node->right, arr, size, capacity);
			}
		}
		void get_nodefeautureids(tnode const *node, unsigned int *&arr, int &size, int &capacity) {
			if(node->featureid!=uint_max) {
				if(size==capacity) {
					capacity = 2*size+1;
					arr = (unsigned int*)realloc(arr, sizeof(unsigned int)*capacity);
				}
				arr[size++] = node->featureid;
				get_nodefeautureids(node->left, arr, size, capacity);
				//get_nodefeautureids(node->right, arr, size, capacity);
			}
		}
		void get_nodethreshold(tnode const *node, float *&arr, int &size, int &capacity) {
			if(node->featureid!=uint_max) {
				if(size==capacity) {
					capacity = 2*size+1;
					arr = (float*)realloc(arr, sizeof(float)*capacity);
				}
				arr[size++] = node->threshold;
				get_nodethreshold(node->left, arr, size, capacity);
				//get_nodethreshold(node->right, arr, size, capacity);
			}
		}
};

void eval_split(xmlnode &split, unsigned int level=0) {
	for(unsigned int i=0; i!=level; ++i) printf("\t");
	if(split.get_nchilds("output")!=0)
		printf("split pos=%s : output=%s\n", split.get_attr("pos"), split.get_child("output").get_text());
	else
		printf("split pos=%s : feature=%s threshold=%s\n", split.get_attr("pos"), split.get_child("feature").get_text(), split.get_child("threshold").get_text());
	if(split.get_nchilds("split")!=0) {
		xmlnode child;
		child = split.get_child("split", 0);
		eval_split(child, level+1);
		child = split.get_child("split", 1);
		eval_split(child, level+1);
	}
}

unsigned int evaldp(float *dpfeatures, unsigned int *featureidxs, float *thresholds, int depth) {
	unsigned int leafidx = 0;
	for(int i=0; i<depth; ++i) {
		const float t = thresholds[i];
		const unsigned int f = featureidxs[i];
		if(dpfeatures[f]<=t) leafidx |= 1<<i;
	}
	return leafidx;
}

float eval(float *fvector, unsigned int **fs, float **ts, float **os, int n, tree **trees, int ntrees) {
	float score = 0.0f;
	for(int i=0; i<ntrees; ++i)
		score += trees[i]->get_weight() * os[i][evaldp(fvector, fs[i], ts[i], n)];
	return score;
}

int main(int argc, char *argv[]) {

	if(argc!=3) {
		printf("\nUsage: %s model_filename output_filename\n\n", argv[0]);
		return 1;
	}

	//xml parsing
	xmlnode f = xmlnode::open_xmlfile(argv[1]);
	xmlnode ensemble = f.get_child("ensemble");
	const int ntrees = ensemble.get_nchilds("tree");
	tree *trees[ntrees];
	for(int i=0, it=0; i<ntrees; ++i) {
		xmlnode t = ensemble.get_child("tree", &it);
		trees[i] = new tree(t);
	}

	FILE *outf = fopen(argv[2], "w");

	if(outf && ntrees>0) {
		//compute sum and max treee depths
		int sumdepth = trees[0]->get_depth();
		int maxdepth = trees[0]->get_depth();
		for(int i=1; i<ntrees; ++i) {
			if(maxdepth<trees[i]->get_depth())
				maxdepth = trees[i]->get_depth();
			sumdepth += trees[i]->get_depth();
		}
		//init file, print no of trees and their max depth
		fprintf(outf, "#ifndef __MODEL_HPP__\n#define __MODEL_HPP__\n\n#define N %d //no. of trees\n#define M %d //max tree depth\n\n", ntrees, maxdepth-1);
		//print average tree depths
		fprintf(outf, "const float avgdepth = %.3ff; //average tree depths\n\n", (float)(sumdepth-ntrees)/ntrees);
		//print tree weights
		fprintf(outf, "float ws[N] = {");
		for(int i=0; i<ntrees; ++i)
			fprintf(outf, " %.8f%c", trees[i]->get_weight(), i<ntrees-1?',':' ');
		fprintf(outf, "};\n\n");
		//print tree depths
		fprintf(outf, "unsigned int ds[N] = {");
		for(int i=0; i<ntrees; ++i)
			fprintf(outf, " %d%c", trees[i]->get_depth()-1, i<ntrees-1?',':' ');
		fprintf(outf, "};\n\n");
		//print tree leaves
		fprintf(outf, "float os[N][1<<M] = {\n");
		for(int i=0; i<ntrees; ++i) {
			fprintf(outf, "\t{");
			int n = 0;
			float *os = trees[i]->get_output(n);
			for(int j=0; j<n; ++j)
				fprintf(outf, " %.8f%c", os[j], j<n-1?',':' ');
			fprintf(outf, "}%c\n", i<ntrees-1?',':' ');
			free(os);
		}
		fprintf(outf, "};\n\n");
		//print node featureids
		fprintf(outf, "unsigned int fs[N][M] = {\n");
		for(int i=0; i<ntrees; ++i) {
			fprintf(outf, "\t{");
			int n = 0;
			unsigned int *fs = trees[i]->get_feautureids(n);
			for(int j=0; j<n; ++j)
				fprintf(outf, " %u%c", fs[j], j<n-1?',':' ');
			fprintf(outf, "}%c\n", i<ntrees-1?',':' ');
			free(fs);
		}
		fprintf(outf, "};\n\n");
		//print node thresholds
		fprintf(outf, "float ts[N][M] = {\n");
		for(int i=0; i<ntrees; ++i) {
			fprintf(outf, "\t{");
			int n = 0;
			float *ts = trees[i]->get_thresholds(n);
			for(int j=0; j<n; ++j)
				fprintf(outf, " %.8f%c", ts[j], j<n-1?',':' ');
			fprintf(outf, "}%c\n", i<ntrees-1?',':' ');
			free(ts);
		}
		fprintf(outf, "};\n\n#endif\n");
		//close file
		fclose(outf);
	}

	//free mem
	for(int i=0; i<ntrees; ++i)
		delete trees[i];

	return 0;
}
