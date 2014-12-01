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
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include "metric/ir/dcg.h"
#include "data/dataset.h"
#include <cmath>

BOOST_AUTO_TEST_CASE( dcg_test ) {
  quickrank::Label labels[] = { 3, 2, 1, 0, 0 };
  quickrank::Score scores[] = { 5, 4, 3, 2, 1 };
  quickrank::data::QueryResults results(5, &labels[0], NULL);

  quickrank::metric::ir::Dcg dcg_metric(5);

  // DCG@k computation with K > num results
  BOOST_CHECK_EQUAL(
      dcg_metric.evaluate_result_list(&results, scores),
      ((pow(2, labels[0]) - 1) + (pow(2, labels[1]) - 1) / log2(3)
          + (pow(2, labels[2]) - 1) / 2));

  // DCG@k computation with K < num results
  dcg_metric.set_cutoff(2);
  BOOST_CHECK_EQUAL(
      dcg_metric.evaluate_result_list(&results, scores),
      ((pow(2, labels[0]) - 1) + (pow(2, labels[1]) - 1) / log2(3)));

  // DCG@k computation with K = 0
  dcg_metric.set_cutoff(0);
  BOOST_CHECK_EQUAL(
      dcg_metric.evaluate_result_list(&results, scores),
      ((pow(2, labels[0]) - 1) + (pow(2, labels[1]) - 1) / log2(3)
          + (pow(2, labels[2]) - 1) / 2));

  // DCG@k computation with K = No cutoff
  dcg_metric.set_cutoff(dcg_metric.NO_CUTOFF);
  BOOST_CHECK_EQUAL(
      dcg_metric.evaluate_result_list(&results, scores),
      ((pow(2, labels[0]) - 1) + (pow(2, labels[1]) - 1) / log2(3)
          + (pow(2, labels[2]) - 1) / 2));
}
