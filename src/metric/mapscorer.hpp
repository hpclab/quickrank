#ifndef __MAP_SCORER_HPP__
#define __MAP_SCORER_HPP__

#include <cstdio>

#include "metric/metricscorer.hpp"
#include "utils/trie.hpp" // trie data structure
#include "utils/strutils.hpp" // rtnode string

class mapscorer : public metricscorer {
	private:
		class counter {
			public:
				counter(const char *junk) : noccurences(0) {};
				void incr() { ++noccurences; }
				unsigned int get_noccurences() const { return noccurences; }
			private:
				unsigned int noccurences;
		};
		trie<counter> relevantdocs;
	public:
		mapscorer(const unsigned int kval=0) { k=kval; }
		const char *whoami() const { return "MAP"; }
		void load_judgments(const char *filename) {
			FILE *f = fopen(filename, "r");
			if(f) {
				//copy from dpset.hpp
				fclose(f);
			}
		}
		float compute_score(const rnklst &rl) {
			float ap = 0.0f;
			unsigned int count = 0;
			//compute score for relevant labels
			for(unsigned int i=0; i<rl.size; ++i)
				if(rl.labels[i]>0.0f)
					ap += (++count)/(i+1.0f);
			counter *rc = relevantdocs.lookup(rl.id);
			if(rc and rc->get_noccurences()>count)
				count = rc->get_noccurences();
			return count==0 ? 0.0f : ap/count;
		}
		fsymmatrix *swap_change(const rnklst &rl) {
			int labels[rl.size];
			int relcount[rl.size];
			int count = 0;
			for(unsigned int i=0; i<rl.size; ++i) {
				if(rl.labels[i]>0.0f) //relevant
					labels[i] = 1,
					++count;
				else
					labels[i] = 0;
				relcount[i] = count;
			}
			counter *rc = relevantdocs.lookup(rl.id);
			if(rc and rc->get_noccurences()>(unsigned int)count)
				count = rc->get_noccurences();
			fsymmatrix *changes = new fsymmatrix(rl.size);
			if(count==0)
				return changes; //all zeros
			#pragma omp parallel for
			for(unsigned int i=0; i<rl.size-1; ++i)
				for(unsigned int j=i+1; j<rl.size; ++j)
					if(labels[i]!=labels[j]) {
						const int diff = labels[j]-labels[i];
						float change = ((relcount[i]+diff)*labels[j]-relcount[i]*labels[i])/(i+1.0f);
						for(unsigned int k=i+1; k<j; ++k)
							if(labels[k]>0) change += (relcount[k]+diff)/(k+1.0f);
						change += (-relcount[j]*diff)/(j+1.0f);
						changes->at(i,j) = change/count;
					}
			return changes;
		}

};

#endif
