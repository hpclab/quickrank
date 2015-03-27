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
