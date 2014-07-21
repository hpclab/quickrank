#ifndef __ENSEMBLE_HPP__
#define __ENSEMBLE_HPP__

#include "learning/dpset.hpp"
#include "learning/tree/rt.hpp"

class ensemble {
	private:
		struct wt {
			wt(rtnode *root, float weight, float maxlabel) : root(root), weight(weight), maxlabel(maxlabel) {}
			rtnode *root = NULL;
			float weight = 0.0f;
			float maxlabel = 0.0f;
		};
		unsigned int size = 0;
		wt *arr = NULL;
	public:
		~ensemble() {
			for(unsigned int i=0; i<size; ++i)
				delete arr[i].root;
			free(arr);
		}
		void set_capacity(const unsigned int n) {
			if(arr) {
				for(unsigned int i=0; i<size; ++i)
					delete arr[i].root;
				free(arr);
			}
			arr = (wt*)malloc(sizeof(wt)*n),
			size = 0;
		}
		void push(rtnode *root, const float weight, const float maxlabel)	{
			arr[size++] = wt(root, weight, maxlabel);
		}
		void pop() {
			delete arr[--size].root;
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
				sum += arr[i].root->eval(features, idx)*arr[i].weight;
			return sum;
		}
		void write_outputtofile(FILE *f) {
			fprintf(f, "\n<ensemble>\n");
			for(unsigned int i=0; i<size; ++i) {
				fprintf(f, "\t<tree id=\"%u\" weight=\"%.3f\">\n", i+1, arr[i].weight);
				if(arr[i].root) {
					fprintf(f, "\t\t<split>\n");
					arr[i].root->write_outputtofile(f, 2);
					fprintf(f, "\t\t</split>\n");
				}
				fprintf(f, "\t</tree>\n");
			}
			fprintf(f, "</ensemble>\n");
		}
};

#endif
