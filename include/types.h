/*
 * QuickRank - A C++ suite of Learning to Rank algorithms
 * Webpage: http://quickrank.isti.cnr.it/
 * Contact: quickrank@isti.cnr.it
 *
 * Unless explicitly acquired and licensed from Licensor under another
 * license, the contents of this file are subject to the Reciprocal Public
 * License ("RPL") Version 1.5, or subsequent versions as allowed by the RPL,
 * and You may not copy or use this file in either source code or executable
 * form, except in compliance with the terms and conditions of the RPL.
 *
 * All software distributed under the RPL is provided strictly on an "AS
 * IS" basis, WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESS OR IMPLIED, AND
 * LICENSOR HEREBY DISCLAIMS ALL SUCH WARRANTIES, INCLUDING WITHOUT
 * LIMITATION, ANY WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
 * PURPOSE, QUIET ENJOYMENT, OR NON-INFRINGEMENT. See the RPL for specific
 * language governing rights and limitations under the RPL.
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
