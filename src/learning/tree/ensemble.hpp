#ifndef __ENSEMBLE_HPP__
#define __ENSEMBLE_HPP__

#include "learning/dpset.hpp"
#include "learning/tree/rt.hpp"

class ensemble {
	private:
		struct wt {
			wt(rtnode *tree, float weight, float maxlabel) : tree(tree), weight(weight), maxlabel(maxlabel) {}
			rtnode *tree = NULL;
			float weight = 0.0f;
			float maxlabel = 0.0f;
		};
		unsigned int size, maxsize;
		wt *arr;
	public:
		ensemble(unsigned int initsize=0) : size(0), maxsize(initsize), arr(NULL) {
			if(initsize>0) arr = (wt*)malloc(sizeof(wt)*initsize);
		}
		~ensemble() {
			for(unsigned int i=0; i<size; ++i)
				delete arr[i].tree;
			free(arr);
		}
		void push(rtnode *tree, const float weight, const float maxlabel)	{
			if(size==maxsize) {
				maxsize = 2*maxsize+1;
				arr = (wt*)realloc(arr, sizeof(wt)*maxsize);
			}
			arr[size++] = wt(tree, weight, maxlabel);
		}
		void pop() {
			delete arr[--size].tree;
		}
		unsigned int get_size() const {
			return size;
		}
		bool is_notempty() const {
			return size>0;
		}
		float eval(float *const *const features, unsigned int idx) const {
			float sum = 0.0f;
			#pragma omp parallel for reduction(+:sum)
			for(unsigned int i=0; i<size; ++i)
				sum += arr[i].tree->eval(features, idx) * arr[i].weight;
			return sum; //prediction value
		}
		void write_outputtofile(FILE *f) {
			fprintf(f, "\n<ensemble>\n");
			for(unsigned int i=0; i<size; ++i) {
				//fprintf(f, "\t<tree id=\"%u\" weight=\"%.3f\" maxlabel=\"%.3f\">\n", i+1, arr[i].weight, arr[i].maxlabel);
				fprintf(f, "\t<tree id=\"%u\" weight=\"%.3f\">\n", i+1, arr[i].weight);
				if(arr[i].tree) {
					fprintf(f, "\t\t<split>\n");
					arr[i].tree->write_outputtofile(f, 2);
					fprintf(f, "\t\t</split>\n");
				}
				fprintf(f, "\t</tree>\n");
			}
			fprintf(f, "</ensemble>\n");
		}
};

#endif
