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
 *   Claudio Lucchese 2016 - claudio.lucchese@isti.cnr.it
 */

#include "catch/include/catch.hpp"

#include <cmath>
#include <iomanip>

#include "metric/ir/ndcg.h"
#include "data/dataset.h"
#include "data/rankedresults.h"

TEST_CASE( "Testing NDCG", "[metric][ndcg]" ) {

  quickrank::Label labels[] = { 3, 2, 1, 0, 0 };
  quickrank::Score scores[] = { 5, 4, 3, 2, 1 };
  auto results = std::shared_ptr<quickrank::data::QueryResults>(
      new quickrank::data::QueryResults(5, &labels[0], NULL) );

  quickrank::metric::ir::Ndcg ndcg_metric(5);
  quickrank::MetricScore idcg;

  // NDCG@k computation with K > num results
  idcg = (pow(2, labels[0]) - 1) + (pow(2, labels[1]) - 1) / log2(3)
      + (pow(2, labels[2]) - 1) / 2;
  REQUIRE( Approx( ndcg_metric.evaluate_result_list(results.get(), scores) ) ==
      ((pow(2, labels[0]) - 1) + (pow(2, labels[1]) - 1) / log2(3)
          + (pow(2, labels[2]) - 1) / 2) / idcg);

  // NDCG@k computation with K = 0
  ndcg_metric.set_cutoff(0);
  REQUIRE( Approx( ndcg_metric.evaluate_result_list(results.get(), scores)) ==
      ((pow(2, labels[0]) - 1) + (pow(2, labels[1]) - 1) / log2(3)
          + (pow(2, labels[2]) - 1) / 2) / idcg);

  // NDCG@k computation with K = No cutoff
  ndcg_metric.set_cutoff(ndcg_metric.NO_CUTOFF);
  REQUIRE( Approx( ndcg_metric.evaluate_result_list(results.get(), scores)) ==
      ((pow(2, labels[0]) - 1) + (pow(2, labels[1]) - 1) / log2(3)
          + (pow(2, labels[2]) - 1) / 2) / idcg);

  // NDCG@k computation with K < num results
  ndcg_metric.set_cutoff(2);
  idcg = (pow(2, labels[0]) - 1) + (pow(2, labels[1]) - 1) / log2(3);
  REQUIRE( Approx( ndcg_metric.evaluate_result_list(results.get(), scores)) ==
      ((pow(2, labels[0]) - 1) + (pow(2, labels[1]) - 1) / log2(3)) / idcg);

  // Jacobian, witouth cutoff
  double true_delta_ndcg, delta_ndcg;

  ndcg_metric.set_cutoff(ndcg_metric.NO_CUTOFF);

  true_delta_ndcg = -ndcg_metric.evaluate_result_list(results.get(), scores);
  std::swap(scores[0],scores[2]);
  true_delta_ndcg += ndcg_metric.evaluate_result_list(results.get(), scores);
  std::swap(scores[0],scores[2]);

  auto ranked_list = std::shared_ptr<quickrank::data::RankedResults>(new quickrank::data::RankedResults(results, scores));
  delta_ndcg = ndcg_metric.jacobian(ranked_list)->at(0,2);

  // std::cout << std::setprecision(18);
  // std::cout << "true delta ndcg: " << true_delta_ndcg << std::endl;
  // std::cout << "delta ndcg by specialized method: " << delta_ndcg << std::endl;

  REQUIRE( Approx(delta_ndcg) == true_delta_ndcg );

  // Jacobian, witouth cutoff in the middle of the swap
  ndcg_metric.set_cutoff(2);

  true_delta_ndcg = -ndcg_metric.evaluate_result_list(results.get(), scores);
  std::swap(scores[0],scores[2]);
  true_delta_ndcg += ndcg_metric.evaluate_result_list(results.get(), scores);
  std::swap(scores[0],scores[2]);

  ranked_list = std::shared_ptr<quickrank::data::RankedResults>(new quickrank::data::RankedResults(results, scores));
  delta_ndcg = ndcg_metric.jacobian(ranked_list)->at(0,2);

  // std::cout << std::setprecision(18);
  // std::cout << "true delta ndcg: " << true_delta_ndcg << std::endl;
  // std::cout << "delta ndcg by specialized method: " << delta_ndcg << std::endl;

  REQUIRE( Approx(delta_ndcg) == true_delta_ndcg );

  // discount = 1/log(2)
  //  std::cout << ((pow(2, labels[2]) - 1) - (pow(2, labels[0]) - 1))/idcg << std::endl;
  REQUIRE( Approx(delta_ndcg) == ( pow(2, labels[2])  - pow(2, labels[0]) )/idcg );

}
