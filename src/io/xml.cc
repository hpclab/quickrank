#include <boost/foreach.hpp>
#include <string>
#include <memory>

#include "io/xml.h"

namespace quickrank {
namespace io {

RTNode* RTNode_parse_xml(const boost::property_tree::ptree &split_xml) {
  RTNode* model_node = NULL;
  RTNode* left_child = NULL;
  RTNode* right_child = NULL;

  bool is_leaf = false;

  unsigned int feature_id = 0;
  float threshold = 0.0f;
  double prediction = 0.0;

  BOOST_FOREACH(const boost::property_tree::ptree::value_type& split_child, split_xml ) {
    if (split_child.first == "output") {
      prediction = split_child.second.get_value<double>();
      is_leaf = true;
      break;
    } else if (split_child.first == "feature") {
      feature_id = split_child.second.get_value<unsigned int>();
    } else if (split_child.first == "threshold") {
      threshold = split_child.second.get_value<float>();
    } else if (split_child.first == "split") {
      std::string pos = split_child.second.get<std::string>("<xmlattr>.pos");
      if (pos == "left")
        left_child = RTNode_parse_xml(split_child.second);
      else
        right_child = RTNode_parse_xml(split_child.second);
    }
  }

  if (is_leaf)
    model_node = new RTNode(prediction);
  else
    /// \todo TODO: this should be changed with item mapping
    model_node = new RTNode(threshold, feature_id - 1, feature_id, left_child,
                            right_child);

  return model_node;
}


}  // namespace io
}  // namespace quickrank
