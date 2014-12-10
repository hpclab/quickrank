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
#define BOOST_TEST_XML_IO
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <fstream>

#include "learning/ltr_algorithm.h"
#include "io/xml.h"

#include <cmath>

BOOST_AUTO_TEST_CASE( test_xml_io ) {

  quickrank::io::Xml xml;

  xml.generate_c_code_baseline("tests/model.xml", "tests/lmart.cc");

  // xml.generate_c_code_oblivious_trees("tests/msn.fold1.quickrank.matrixnet.xml", "tests/matrixnet.cc");

  xml.generate_c_code_vectorized("tests/model.xml", "tests/lmart-fast.cc");

}
