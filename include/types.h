#ifndef QUICKRANK_TYPES_H_
#define QUICKRANK_TYPES_H_

#include <vector>
#include "utils/symmatrix.h"

namespace quickrank {

typedef float Label;  /// data type for instance truth label
typedef double Score;  /// data type for instance predicted label
typedef float Feature;  /// data type for instance feature
typedef unsigned int QueryID;  /// data type for QueryID in L-t-R datasets
typedef double MetricScore;  /// data type for evaluation metric final outcome

typedef SymMatrix<double> Jacobian;  /// data type for a Metric's Jacobian Matrix

}  // namespace quickrank

#endif
