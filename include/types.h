#ifndef QUICKRANK_TYPES_H_
#define QUICKRANK_TYPES_H_

#include <vector>
#include "utils/symmatrix.h"

namespace qr {

typedef float Label; // data type for instance truth label
typedef float Score; // data type for instance predicted label
typedef float Feature; // data type for instance feature
typedef double MetricScore; // data type for evaluation metric final outcome

typedef std::vector<Feature> FeatureVector; // data type for vector of instance features

typedef SymMatrix<double> Jacobian;  // data type for a Metric's Jacobian Matrix

}


#endif
