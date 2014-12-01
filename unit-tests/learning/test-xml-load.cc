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
