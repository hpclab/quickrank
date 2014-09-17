#ifndef __METRICSCORER_HPP__
#define __METRICSCORER_HPP__

#include "learning/dpset.hpp"
#include "utils/symmatrix.hpp" // symetric matrix

typedef symmatrix<double> fsymmatrix;

class metricscorer {
	protected:
		unsigned int k;
	public:
		virtual ~metricscorer() {}
		void set_k(const unsigned int _k) { k = _k; }
		unsigned int get_k() const { return k; }
		virtual const char *whoami() const = 0;
		virtual double compute_score(const qlist &ql) = 0;
		virtual fsymmatrix *swap_change(const qlist &ql) = 0;
		virtual void showme() { printf("\tscorer type = %s@%u\n", whoami(), k); };
};

#endif
