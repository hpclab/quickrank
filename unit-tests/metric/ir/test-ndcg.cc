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
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <cmath>
#include <iomanip>

#include "metric/ir/ndcg.h"
#include "data/dataset.h"
#include "data/rankedresults.h"

BOOST_AUTO_TEST_CASE( ndcg_test ) {
  quickrank::Label labels[] = { 3, 2, 1, 0, 0 };
  quickrank::Score scores[] = { 5, 4, 3, 2, 1 };
  auto results = std::shared_ptr<quickrank::data::QueryResults>(
      new quickrank::data::QueryResults(5, &labels[0], NULL) );

  quickrank::metric::ir::Ndcg ndcg_metric(5);
  quickrank::MetricScore idcg;

  // NDCG@k computation with K > num results
  idcg = (pow(2, labels[0]) - 1) + (pow(2, labels[1]) - 1) / log2(3)
      + (pow(2, labels[2]) - 1) / 2;
  BOOST_CHECK_EQUAL(
      ndcg_metric.evaluate_result_list(results.get(), scores),
      ((pow(2, labels[0]) - 1) + (pow(2, labels[1]) - 1) / log2(3)
          + (pow(2, labels[2]) - 1) / 2) / idcg);

  // NDCG@k computation with K = 0
  ndcg_metric.set_cutoff(0);
  BOOST_CHECK_EQUAL(
      ndcg_metric.evaluate_result_list(results.get(), scores),
      ((pow(2, labels[0]) - 1) + (pow(2, labels[1]) - 1) / log2(3)
          + (pow(2, labels[2]) - 1) / 2) / idcg);

  // NDCG@k computation with K = No cutoff
  ndcg_metric.set_cutoff(ndcg_metric.NO_CUTOFF);
  BOOST_CHECK_EQUAL(
      ndcg_metric.evaluate_result_list(results.get(), scores),
      ((pow(2, labels[0]) - 1) + (pow(2, labels[1]) - 1) / log2(3)
          + (pow(2, labels[2]) - 1) / 2) / idcg);

  // NDCG@k computation with K < num results
  ndcg_metric.set_cutoff(2);
  idcg = (pow(2, labels[0]) - 1) + (pow(2, labels[1]) - 1) / log2(3);
  BOOST_CHECK_EQUAL(
      ndcg_metric.evaluate_result_list(results.get(), scores),
      ((pow(2, labels[0]) - 1) + (pow(2, labels[1]) - 1) / log2(3)) / idcg);


  /*
  ndcg_metric.set_cutoff(ndcg_metric.NO_CUTOFF);

  double delta_ndcg = -ndcg_metric.evaluate_result_list(results.get(), scores);
  std::swap(scores[0],scores[2]);
  delta_ndcg += ndcg_metric.evaluate_result_list(results.get(), scores);
  std::swap(scores[0],scores[2]);
  std::cout << std::setprecision(18);

  auto ranked_list = std::shared_ptr<quickrank::data::RankedResults>(new quickrank::data::RankedResults(results, scores));
  double delta2_ndcg = ndcg_metric.jacobian(ranked_list)->at(0,2);

  double delta3_ndcg = ndcg_metric.get_jacobian(results)->at(0,2);
  std::cout << "true delta ndcg: " << delta_ndcg << std::endl;
  std::cout << "delta ndcg by specialized method: " << delta2_ndcg << std::endl;
  std::cout << "old ndcg by specialized method: " << delta3_ndcg << std::endl;

  BOOST_CHECK_EQUAL(delta_ndcg, delta2_ndcg);
*/
}
