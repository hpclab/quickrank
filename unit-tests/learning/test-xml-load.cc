#define BOOST_TEST_XML_LOAD
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <fstream>

#include "learning/ltr_algorithm.h"

#include <cmath>

BOOST_AUTO_TEST_CASE( test_xml_load ) {

  // read model
  auto model = quickrank::learning::LTR_Algorithm::load_model_from_file("tests/msn.fold1.quickrank.lmart.xml");

  std::ofstream out;
  out.open("tests/prova.out.xml", std::ofstream::out);
  model->save_model_to_file(out);

  out.close();
}
