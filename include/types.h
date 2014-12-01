/*
 * QuickRank - A C++ suite of Learning to Rank algorithms
 * Webpage: http://quickrank.isti.cnr.it/
 * Contact: quickrank@isti.cnr.it
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Contributor:
 *   HPC. Laboratory - ISTI - CNR - http://hpc.isti.cnr.it/
 */
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
