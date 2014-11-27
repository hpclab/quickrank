#define BOOST_TEST_XML_IO
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <fstream>

#include "learning/ltr_algorithm.h"
#include "io/xml.h"

#include <cmath>

BOOST_AUTO_TEST_CASE( test_xml_io ) {

  quickrank::io::Xml xml;

  xml.generate_c_code("tests/msn.fold1.quickrank.lmart.xml", "tests/ranker.c");

}
