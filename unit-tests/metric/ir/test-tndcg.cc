#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <iomanip>

#include "metric/ir/tndcg.h"
#include "data/dataset.h"
#include <cmath>

BOOST_AUTO_TEST_CASE( tndcg_test ) {
  quickrank::Label labels[] = { 3, 2, 1, 0, 0 };
  quickrank::Score scores[] = { 5, 4, 3, 2, 1 };
  auto results = std::shared_ptr<quickrank::data::QueryResults>(
      new quickrank::data::QueryResults(5, &labels[0], NULL) );

  quickrank::metric::ir::Tndcg tndcg_metric(5);
  quickrank::MetricScore idcg;

  // NDCG@k computation with K > num results
  idcg = (pow(2, labels[0]) - 1) + (pow(2, labels[1]) - 1) / log2(3)
          + (pow(2, labels[2]) - 1) / 2;

  BOOST_CHECK_EQUAL(
      tndcg_metric.evaluate_result_list(results.get(), scores),
      1.0);

  scores[0] = 4;
  BOOST_CHECK_EQUAL(
      tndcg_metric.evaluate_result_list(results.get(), scores),
      (
          (  (pow(2, labels[0]) - 1) + (pow(2, labels[1]) - 1)  ) / 2 +
          (  (pow(2, labels[0]) - 1) + (pow(2, labels[1]) - 1)  ) / 2 / log2(3) +
          (pow(2, labels[2]) - 1) / 2
      ) / idcg );

  scores[1] = 3;
  BOOST_CHECK_EQUAL(
      tndcg_metric.evaluate_result_list(results.get(), scores),
      (
          (pow(2, labels[0]) - 1) +
          (  (pow(2, labels[1]) - 1) + (pow(2, labels[2]) - 1)  ) / 2 / log2(3) +
          (  (pow(2, labels[1]) - 1) + (pow(2, labels[2]) - 1)  ) / 2 / 2
      ) / idcg );

  /*
  scores[0] = 5;
  scores[1] = 4;
  tndcg_metric.set_cutoff(tndcg_metric.NO_CUTOFF);

  double delta_tndcg = -tndcg_metric.evaluate_result_list(results.get(), scores);
  std::swap(scores[0],scores[2]);
  delta_tndcg += tndcg_metric.evaluate_result_list(results.get(), scores);
  std::swap(scores[0],scores[2]);
  std::cout << std::setprecision(18);

  auto ranked_list = std::shared_ptr<quickrank::data::RankedResults>(new quickrank::data::RankedResults(results, scores));
  double delta2_tndcg = tndcg_metric.jacobian(ranked_list)->at(0,2);
  //double delta3_tndcg = tndcg_metric.get_jacobian(results)->at(0,2);

  std::cout << "true delta ndcg: " << delta_tndcg << std::endl;
  std::cout << "delta ndcg by specialized method: " << delta2_tndcg << std::endl;
  //std::cout << "old ndcg by specialized method: " << delta3_tndcg << std::endl;
  */

}
