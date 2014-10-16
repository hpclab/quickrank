#define BOOST_TEST_HDATA_LINK
#include <boost/test/unit_test.hpp>

#include "data/dataset.h"
#include "data/queryresults.h"
#include "io/svml.h"
#include <cmath>

BOOST_AUTO_TEST_CASE( horizontal_data_test ) {

  // read and check dataset
  quickrank::io::Svml reader;
  quickrank::data::Dataset* dataset= reader.read_horizontal("tests/data/msn1.fold1.train.5k.txt");

  BOOST_CHECK_EQUAL(dataset->num_features(), 136);
  BOOST_CHECK_EQUAL(dataset->num_instances(), 5000);
  BOOST_CHECK_EQUAL(dataset->num_queries(), 43);

  // check query results
  quickrank::data::QueryResults qr = dataset->getQueryResults(0);

  BOOST_CHECK_EQUAL(qr.num_results(), 86);
  BOOST_CHECK_EQUAL(qr.labels()[0], 2);
  BOOST_CHECK_EQUAL(qr.labels()[1], 2);
  BOOST_CHECK_EQUAL(qr.labels()[2], 0);
  BOOST_CHECK_EQUAL(qr.features()[0], 3);
  BOOST_CHECK_EQUAL(qr.features()[dataset->num_features() +1], 0);
  BOOST_CHECK_EQUAL(qr.features()[2*dataset->num_features() +2], 2);

  // check query results
  qr = dataset->getQueryResults(1);

  BOOST_CHECK_EQUAL(qr.num_results(), 106);
  BOOST_CHECK_EQUAL(qr.labels()[0], 0);
  BOOST_CHECK_EQUAL(qr.labels()[1], 0);
  BOOST_CHECK_EQUAL(qr.labels()[2], 0);
  BOOST_CHECK_EQUAL(qr.features()[0], 0);
  BOOST_CHECK_EQUAL(qr.features()[dataset->num_features() +1], 0);
  BOOST_CHECK_EQUAL(qr.features()[2*dataset->num_features() +2], 5);

}
