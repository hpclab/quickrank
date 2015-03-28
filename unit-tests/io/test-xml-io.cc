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
#define BOOST_TEST_XML_IO
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <fstream>

#include "learning/ltr_algorithm.h"
#include "io/xml.h"

#include <cmath>

BOOST_AUTO_TEST_CASE( test_xml_io ) {

  quickrank::io::Xml xml;
  /// \todo TODO: To be rewritten or removed

  // xml.generate_c_code_baseline("tests/model.tree.1", "tests/prova_baseline.cc");

  // xml.generate_c_code_oblivious_trees("tests/msn.fold1.quickrank.matrixnet.xml", "tests/matrixnet.cc");

  //xml.generate_c_code_vectorized("tests/model.xml", "tests/lmart-fast.cc");

}
