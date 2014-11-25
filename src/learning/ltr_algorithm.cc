#include <fstream>

#include <boost/property_tree/xml_parser.hpp>
#include <boost/foreach.hpp>

#include "learning/ltr_algorithm.h"
#include "utils/mergesorter.h"

#include "learning/forests/mart.h"

#ifdef _OPENMP
#include <omp.h>
#else
#include "utils/omp-stubs.h"
#endif

namespace quickrank {
namespace learning {

void LTR_Algorithm::score_dataset(std::shared_ptr<data::Dataset> dataset,
                                  Score* scores) const {
  preprocess_dataset(dataset);
  for (unsigned int q = 0; q < dataset->num_queries(); q++) {
    std::shared_ptr<data::QueryResults> r = dataset->getQueryResults(q);
    score_query_results(r, scores, dataset->num_instances());
    scores += r->num_results();
  }
}

// assumes vertical dataset
// offset to next feature of the same instance
void LTR_Algorithm::score_query_results(
    std::shared_ptr<data::QueryResults> results, Score* scores,
    unsigned int offset) const {
  const quickrank::Feature* d = results->features();
  for (unsigned int i = 0; i < results->num_results(); i++) {
    scores[i] = score_document(d, offset);
    d++;
  }
}

// assumes vertical dataset
Score LTR_Algorithm::score_document(const quickrank::Feature* d,
                                    const unsigned int offset) const {
  return 0.0;
}

void LTR_Algorithm::save(std::string output_basename, int iteration) const {
  if (!output_basename.empty()) {
    std::string filename = output_basename;
    if (iteration != -1)
      filename += iteration + ".xml";
    std::ofstream output_stream;
    output_stream.open(filename);
    // Wrap actual model
    output_stream  << "<ranker>" << std::endl;

    // Save the actual model
    save_model_to_file(output_stream);

    output_stream  << "</ranker>" << std::endl;

    output_stream.close();
  }
}

LTR_Algorithm* LTR_Algorithm::load_model_from_file(std::string model_filename) {
  if (model_filename.empty()) {
    std::cerr << "!!! Model filename is empty." << std::endl;
    exit(EXIT_FAILURE);
  }

  boost::property_tree::ptree xml_tree;

  std::ifstream is;
  is.open(model_filename, std::ifstream::in);

  boost::property_tree::read_xml(is, xml_tree);

  is.close();

  boost::property_tree::ptree info_ptree;
  boost::property_tree::ptree ensemble_ptree;

  BOOST_FOREACH(const boost::property_tree::ptree::value_type& node, xml_tree.get_child("ranker")) {
    if (node.first=="info")
      info_ptree = node.second;
    else if (node.first=="ensemble")
      ensemble_ptree = node.second;
  }

  std::string ranker_type = info_ptree.get<std::string>("type");
  if (ranker_type=="MART")
      return new forests::Mart(info_ptree, ensemble_ptree);
/*
  else
      break;
    case "LAMBDAMART":
      return NULL;
      break;
    case "MATRIXNET":
      return NULL;
      break;
    default:
      break;
  }*/
  return NULL;
}

}  // namespace learning
}  // namespace quickrank
