#define BOOST_TEST_HDATA_LINK
#include <boost/test/unit_test.hpp>

#include "metric/ir/dcg.h"
#include "metric/ir/ndcg.h"
#include "data/dataset.h"
#include "data/queryresults.h"
#include "io/svml.h"
#include <cmath>

BOOST_AUTO_TEST_CASE( Horizontal_Dataset_Test ) {

  // read and check dataset
  quickrank::io::Svml reader;
  std::unique_ptr<quickrank::data::Dataset> dataset = reader.read_horizontal("tests/data/msn1.fold1.train.5k.txt");

  BOOST_CHECK_EQUAL(dataset->num_features(), 136);
  BOOST_CHECK_EQUAL(dataset->num_instances(), 5000);
  BOOST_CHECK_EQUAL(dataset->num_queries(), 43);

  // check query results
  std::unique_ptr<quickrank::data::QueryResults> qr = dataset->getQueryResults(0);

  BOOST_CHECK_EQUAL(qr->num_results(), 86);
  BOOST_CHECK_EQUAL(qr->labels()[0], 2);
  BOOST_CHECK_EQUAL(qr->labels()[1], 2);
  BOOST_CHECK_EQUAL(qr->labels()[2], 0);
  BOOST_CHECK_EQUAL(qr->features()[0], 3);
  BOOST_CHECK_EQUAL(qr->features()[dataset->num_features() +1], 0);
  BOOST_CHECK_EQUAL(qr->features()[2*dataset->num_features() +2], 2);

  // check query results
  qr = dataset->getQueryResults(1);

  BOOST_CHECK_EQUAL(qr->num_results(), 106);
  BOOST_CHECK_EQUAL(qr->labels()[0], 0);
  BOOST_CHECK_EQUAL(qr->labels()[1], 0);
  BOOST_CHECK_EQUAL(qr->labels()[2], 0);
  BOOST_CHECK_EQUAL(qr->features()[0], 0);
  BOOST_CHECK_EQUAL(qr->features()[dataset->num_features() +1], 0);
  BOOST_CHECK_EQUAL(qr->features()[2*dataset->num_features() +2], 5);

  // check some metrics on the given data
  qr = dataset->getQueryResults(0);
  qr::metric::ir::Dcg dcg_metric(3);

  qr::Score scores [86] = {3,2,1};
  BOOST_CHECK_EQUAL( dcg_metric.evaluate_result_list(qr.get(), scores ),
                     (pow(2,qr->labels()[0])-1) + (pow(2,qr->labels()[1])-1)/log2(3) + (pow(2,qr->labels()[2])-1)/2
  );

  qr::Score scores2 [86] = {1,2,3};
  BOOST_CHECK_EQUAL( dcg_metric.evaluate_result_list(qr.get(), scores2 ),
                     (pow(2,qr->labels()[2])-1) + (pow(2,qr->labels()[1])-1)/log2(3) + (pow(2,qr->labels()[0])-1)/2
  );

  qr::metric::ir::Ndcg ndcg_metric(3); // ideal is 3,2,2
  BOOST_CHECK_EQUAL( ndcg_metric.evaluate_result_list(qr.get(), scores ),
                     ( (pow(2,qr->labels()[0])-1) + (pow(2,qr->labels()[1])-1)/log2(3) + (pow(2,qr->labels()[2])-1)/2 ) /
                     ( (pow(2,3)-1) + (pow(2,2)-1)/log2(3) + (pow(2,2)-1)/2 )
  );
  BOOST_CHECK_EQUAL( ndcg_metric.evaluate_result_list(qr.get(), scores2 ),
                     ( (pow(2,qr->labels()[2])-1) + (pow(2,qr->labels()[1])-1)/log2(3) + (pow(2,qr->labels()[0])-1)/2 ) /
                     ( (pow(2,3)-1) + (pow(2,2)-1)/log2(3) + (pow(2,2)-1)/2 )
  );

  // check vertical dataset
  dataset->transpose();
  qr = dataset->getQueryResults(0);
  BOOST_CHECK_EQUAL(qr->num_results(), 86);
  BOOST_CHECK_EQUAL(qr->features()[0], 3);
  BOOST_CHECK_EQUAL(qr->features()[qr->num_results()+1], 0);
  BOOST_CHECK_EQUAL(qr->features()[2*qr->num_results()+2], 2);

  // check horizontal dataset
  dataset->transpose();
  qr = dataset->getQueryResults(0);
  BOOST_CHECK_EQUAL(qr->num_results(), 86);
  BOOST_CHECK_EQUAL(qr->features()[0], 3);
  BOOST_CHECK_EQUAL(qr->features()[dataset->num_features() +1], 0);
  BOOST_CHECK_EQUAL(qr->features()[2*dataset->num_features() +2], 2);
}
