/*
void compute_pseudoresponses() {
			//NOTE this method is from MART definition, LambdaMART one is not invoked
			const unsigned int nentries = training_set->get_nentries();
			for(unsigned int i=0; i<nentries; ++i)
				pseudoresponses[i] = training_set->get_label(i) - model_scores[i];
			#ifdef VERBOSE
			//pseudoresponses' values are permuted wrt the order with which filelines are read
			printf("*** compute_pseudoresponses()\npseudoresponses:\n");
			for(unsigned int i=0; i<nentries; ++i)
				printf("%.1f ", pseudoresponses[i]);
			printf("(%u)\n", nentries);
			getchar();
			#endif
		}
		void update_treeoutput(rgtree *rt) {
			splitnode **leaves = rt->get_leaves();
			unsigned int nleaves = rt->get_nleaves();
			#pragma omp parallel for
			for(unsigned int i=0; i<nleaves; ++i) {
				float psum = 0.0f;
				const unsigned int *samples = leaves[i]->samples;
				const unsigned int nsamples = leaves[i]->nsamples;
				for(unsigned int j=0; j<nsamples; ++j)
					psum += pseudoresponses[samples[j]];
				leaves[i]->avglabel = psum/nsamples;
			}
			#ifdef VERBOSE
			printf("*** update_treeoutput()\nleaves:\n");
			for(unsigned int i=0; i<nleaves; ++i)
				printf("%.1f ", leaves[i]->avglabel);
			printf("(%u)\n", nleaves);
			getchar();
			#endif
		}
*/
