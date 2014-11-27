#define BOOST_TEST_XML_IO
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <fstream>

#include "learning/ltr_algorithm.h"
#include "io/xml.h"

#include <cmath>

BOOST_AUTO_TEST_CASE( test_xml_io ) {

  quickrank::io::Xml xml;

  xml.generate_c_code("tests/msn.fold1.quickrank.lmart.xml", "tests/prova.c");

  /*
  // read model
  auto model = quickrank::learning::LTR_Algorithm::load_model_from_file("tests/msn.fold1.quickrank.lmart.xml");

  std::ofstream out;
  out.open("tests/prova.out.xml", std::ofstream::out);
  model->save_model_to_file(out);

  out.close();
  */
}
