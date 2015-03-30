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
#ifndef QUICKRANK_METRIC_IR_METRIC_FACTORY_H_
#define QUICKRANK_METRIC_IR_METRIC_FACTORY_H_

#include <iostream>
#include <climits>
#include <memory>
#include <boost/noncopyable.hpp>

#include "types.h"

#include <boost/algorithm/string/case_conv.hpp>

#include "metric/ir/tndcg.h"
#include "metric/ir/ndcg.h"
#include "metric/ir/dcg.h"
#include "metric/ir/map.h"


namespace quickrank {
namespace metric {
namespace ir {

std::shared_ptr<Metric> ir_metric_factory(
    std::string metric, unsigned int cutoff) {
  boost::to_upper(metric);
  if (metric == Dcg::NAME_)
    return std::shared_ptr<Metric>( new Dcg(cutoff) );
  else if (metric == Ndcg::NAME_)
    return std::shared_ptr<Metric>( new Ndcg(cutoff) );
  else if (metric == Tndcg::NAME_)
    return std::shared_ptr<Metric>( new Tndcg(cutoff) );
  else if (metric == Map::NAME_)
    return std::shared_ptr<Metric>( new Map(cutoff) );
  else
    return std::shared_ptr<Metric>();
}

}  // namespace ir
}  // namespace metric
}  // namespace quickrank

#endif
