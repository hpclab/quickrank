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


#include <iomanip>

#include "metric/ir/tndcg.h"
#include "data/dataset.h"
#include <cmath>

TEST_CASE( "Testing TNDCG", "[metric][tndcg]" ) {
  quickrank::Label labels[] = { 3, 2, 1, 0, 0 };
  quickrank::Score scores[] = { 5, 4, 3, 2, 1 };
  auto results = std::shared_ptr<quickrank::data::QueryResults>(
      new quickrank::data::QueryResults(5, &labels[0], NULL) );

  quickrank::metric::ir::Tndcg tndcg_metric(5);
  quickrank::MetricScore idcg;

  // NDCG@k computation with K > num results
  idcg = (pow(2, labels[0]) - 1) + (pow(2, labels[1]) - 1) / log2(3)
          + (pow(2, labels[2]) - 1) / 2;

  REQUIRE( Approx( tndcg_metric.evaluate_result_list(results.get(), scores) ) == 1.0);

  scores[0] = 4;
  REQUIRE( Approx( tndcg_metric.evaluate_result_list(results.get(), scores) ) ==
      (
          (  (pow(2, labels[0]) - 1) + (pow(2, labels[1]) - 1)  ) / 2 +
          (  (pow(2, labels[0]) - 1) + (pow(2, labels[1]) - 1)  ) / 2 / log2(3) +
          (pow(2, labels[2]) - 1) / 2
      ) / idcg );

  scores[1] = 3;
  REQUIRE( Approx( tndcg_metric.evaluate_result_list(results.get(), scores) ) ==
      (
          (pow(2, labels[0]) - 1) +
          (  (pow(2, labels[1]) - 1) + (pow(2, labels[2]) - 1)  ) / 2 / log2(3) +
          (  (pow(2, labels[1]) - 1) + (pow(2, labels[2]) - 1)  ) / 2 / 2
      ) / idcg );

  // Jacobian, witouth cutoff
  double true_delta_tndcg, delta_tndcg;
  scores[0] = 5;
  scores[1] = 4;

  tndcg_metric.set_cutoff(tndcg_metric.NO_CUTOFF);

  true_delta_tndcg = -tndcg_metric.evaluate_result_list(results.get(), scores);
  std::swap(scores[0],scores[2]);
  true_delta_tndcg += tndcg_metric.evaluate_result_list(results.get(), scores);
  std::swap(scores[0],scores[2]);

  auto ranked_list = std::shared_ptr<quickrank::data::RankedResults>(new quickrank::data::RankedResults(results, scores));
  delta_tndcg = tndcg_metric.jacobian(ranked_list)->at(0,2);

//  std::cout << std::setprecision(18);
//  std::cout << "true delta tndcg: " << true_delta_tndcg << std::endl;
//  std::cout << "delta tndcg by specialized method: " << delta_tndcg << std::endl;

  REQUIRE( Approx(delta_tndcg) == true_delta_tndcg );

  // Jacobian, witouth cutoff in the middle of the swap
  tndcg_metric.set_cutoff(2);
  idcg = (pow(2, labels[0]) - 1) + (pow(2, labels[1]) - 1) / log2(3);

  true_delta_tndcg = -tndcg_metric.evaluate_result_list(results.get(), scores);
  std::swap(scores[0],scores[2]);
  true_delta_tndcg += tndcg_metric.evaluate_result_list(results.get(), scores);
  std::swap(scores[0],scores[2]);

  ranked_list = std::shared_ptr<quickrank::data::RankedResults>(new quickrank::data::RankedResults(results, scores));
  delta_tndcg = tndcg_metric.jacobian(ranked_list)->at(0,2);

//  std::cout << std::setprecision(18);
//  std::cout << "true delta tndcg: " << true_delta_tndcg << std::endl;
//  std::cout << "delta tndcg by specialized method: " << delta_tndcg << std::endl;

  REQUIRE( Approx(delta_tndcg) == true_delta_tndcg );


  // discount = 1/log(2)
  //  std::cout << ((pow(2, labels[2]) - 1) - (pow(2, labels[0]) - 1))/idcg << std::endl;
  REQUIRE( Approx(delta_tndcg) == ( pow(2, labels[2])  - pow(2, labels[0]) )/idcg );

}
