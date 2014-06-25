#ifndef __METRICSCORER_HPP__
#define __METRICSCORER_HPP__

#include "learning/dpset.hpp"
#include "utils/symmatrix.hpp" // symetric matrix

typedef symmatrix<float> fsymmatrix;

class metricscorer {
	protected:
		unsigned int k = 10;
	public:
		virtual ~metricscorer() {}
		void set_k(const unsigned int _k) { k = _k; }
		unsigned int get_k() const { return k; }
		virtual const char *whoami() const = 0;
		virtual void load_judgments(const char *filename) = 0;
		virtual float compute_score(const rnklst &rl) = 0;
		virtual fsymmatrix *swap_change(const rnklst &rl) = 0;
};

#endif
