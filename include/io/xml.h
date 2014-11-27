#ifndef QUICKRANK_IO_XML_H_
#define QUICKRANK_IO_XML_H_

#include <boost/property_tree/ptree.hpp>

#include "learning/tree/rt.h"

namespace quickrank {
namespace io {

RTNode* RTNode_parse_xml(const boost::property_tree::ptree &split_xml);

}  // namespace io
}  // namespace quickrank

#endif
