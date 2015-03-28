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
#define BOOST_TEST_HDATA_LINK
#include <boost/test/unit_test.hpp>

#include "metric/ir/dcg.h"
#include "metric/ir/ndcg.h"
#include "data/dataset.h"
#include "data/queryresults.h"
#include "io/svml.h"
#include <cmath>
#include <iomanip>

BOOST_AUTO_TEST_CASE( Dataset_IO_Test ) {

  // read and check dataset
  quickrank::io::Svml reader;
  std::unique_ptr<quickrank::data::Dataset> dataset = reader.read_horizontal(
      "quickranktestdata/msn1/msn1.fold1.train.5k.txt");

  // 226244459
  // std::cout << std::setprecision(16) << *(dataset->at(4329, 127)) << std::endl;
  // std::cout << std::setprecision(16) << 226244459.0f << std::endl;

  BOOST_CHECK_EQUAL(dataset->num_features(), 136);
  BOOST_CHECK_EQUAL(dataset->num_instances(), 5000);
  BOOST_CHECK_EQUAL(dataset->num_queries(), 43);

  // check query results
  std::unique_ptr<quickrank::data::QueryResults> qr = dataset->getQueryResults(
      0);

  BOOST_CHECK_EQUAL(qr->num_results(), 86);
  BOOST_CHECK_EQUAL(qr->labels()[0], 2);
  BOOST_CHECK_EQUAL(qr->labels()[1], 2);
  BOOST_CHECK_EQUAL(qr->labels()[2], 0);
  BOOST_CHECK_EQUAL(qr->features()[0], 3);
  BOOST_CHECK_EQUAL(qr->features()[dataset->num_features() + 1], 0);
  BOOST_CHECK_EQUAL(qr->features()[2 * dataset->num_features() + 2], 2);

  // check query results
  qr = dataset->getQueryResults(1);

  BOOST_CHECK_EQUAL(qr->num_results(), 106);
  BOOST_CHECK_EQUAL(qr->labels()[0], 0);
  BOOST_CHECK_EQUAL(qr->labels()[1], 0);
  BOOST_CHECK_EQUAL(qr->labels()[2], 0);
  BOOST_CHECK_EQUAL(qr->features()[0], 0);
  BOOST_CHECK_EQUAL(qr->features()[dataset->num_features() + 1], 0);
  BOOST_CHECK_EQUAL(qr->features()[2 * dataset->num_features() + 2], 5);

  // check some metrics on the given data
  qr = dataset->getQueryResults(0);
  quickrank::metric::ir::Dcg dcg_metric(3);

  quickrank::Score scores[86] = { 3, 2, 1 };
  BOOST_CHECK_EQUAL(
      dcg_metric.evaluate_result_list(qr.get(), scores),
      (pow(2, qr->labels()[0]) - 1) + (pow(2, qr->labels()[1]) - 1) / log2(3)
          + (pow(2, qr->labels()[2]) - 1) / 2);

  quickrank::Score scores2[86] = { 1, 2, 3 };
  BOOST_CHECK_EQUAL(
      dcg_metric.evaluate_result_list(qr.get(), scores2),
      (pow(2, qr->labels()[2]) - 1) + (pow(2, qr->labels()[1]) - 1) / log2(3)
          + (pow(2, qr->labels()[0]) - 1) / 2);

  quickrank::metric::ir::Ndcg ndcg_metric(3);  // ideal is 3,2,2
  BOOST_CHECK_EQUAL(
      ndcg_metric.evaluate_result_list(qr.get(), scores),
      ((pow(2, qr->labels()[0]) - 1) + (pow(2, qr->labels()[1]) - 1) / log2(3)
          + (pow(2, qr->labels()[2]) - 1) / 2)
          / ((pow(2, 3) - 1) + (pow(2, 2) - 1) / log2(3) + (pow(2, 2) - 1) / 2));
  BOOST_CHECK_EQUAL(
      ndcg_metric.evaluate_result_list(qr.get(), scores2),
      ((pow(2, qr->labels()[2]) - 1) + (pow(2, qr->labels()[1]) - 1) / log2(3)
          + (pow(2, qr->labels()[0]) - 1) / 2)
          / ((pow(2, 3) - 1) + (pow(2, 2) - 1) / log2(3) + (pow(2, 2) - 1) / 2));

  // check vertical dataset
  dataset->transpose();
  qr = dataset->getQueryResults(0);
  BOOST_CHECK_EQUAL(qr->num_results(), 86);
  BOOST_CHECK_EQUAL(qr->features()[0], 3);
  BOOST_CHECK_EQUAL(qr->features()[dataset->num_instances() + 1], 0);
  BOOST_CHECK_EQUAL(qr->features()[2 * dataset->num_instances() + 2], 2);

  // check horizontal dataset
  dataset->transpose();
  qr = dataset->getQueryResults(0);
  BOOST_CHECK_EQUAL(qr->num_results(), 86);
  BOOST_CHECK_EQUAL(qr->features()[0], 3);
  BOOST_CHECK_EQUAL(qr->features()[dataset->num_features() + 1], 0);
  BOOST_CHECK_EQUAL(qr->features()[2 * dataset->num_features() + 2], 2);
}
