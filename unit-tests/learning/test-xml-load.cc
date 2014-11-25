#define BOOST_TEST_XML_LOAD
#include <boost/test/unit_test.hpp>

#include "learning/ltr_algorithm.h"

#include <cmath>

BOOST_AUTO_TEST_CASE( test_xml_load ) {

  // read model
  quickrank::learning::LTR_Algorithm* prova = quickrank::learning::LTR_Algorithm::load_model_from_file("tests/msn.fold1.quickrank.lmart.xml");




}
