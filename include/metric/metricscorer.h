#ifndef QUICKRANK_METRIC_METRICSCORER_H_
#define QUICKRANK_METRIC_METRICSCORER_H_

#include "learning/dpset.h"
#include "utils/symmatrix.h" // symetric matrix

#include "types.h"

#include <iostream>

typedef SymMatrix<double> fsymmatrix;

class Metric {
 protected:
  unsigned int k;

 public:
  virtual ~Metric() {}
  void set_k(const unsigned int _k) { k = _k; }
  unsigned int get_k() const { return k; }

  virtual const char *whoami() const = 0;

  virtual double compute_score(const qlist &ql) = 0;

  virtual fsymmatrix *swap_change(const qlist &ql) = 0;

  virtual void showme() { printf("\tscorer type = %s@%u\n", whoami(), k); };

  // Metric

};

/*
namespace qr {
namespace metric {
namespace ir {

class Metric
{
 public:
  static const unsigned int DEFAULT_CUTOFF = 10;

  explicit Metric(int k = DEFAULT_CUTOFF) { cutoff_ = (k > 0 ? k : DEFAULT_CUTOFF); }

  virtual ~Metric() {};

  unsigned int cutoff() const { return cutoff_; }
  void set_cutoff(unsigned int k) { cutoff_ = (k > 0 ? k : cutoff_); }

  MetricScore compute_score(const qlist&) const = 0;

 private:
  Metric(const Metric&);
  Metric& operator=(const Metric&);

  unsigned int cutoff_;

  friend std::ostream& operator<<(std::ostream&, const Metric&);
};

} // namespace ir
} // namespace metric
} // namespace qr
*/

#endif
