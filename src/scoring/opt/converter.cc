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
#include <string.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <queue>

#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/detail/ptree_utils.hpp>

#include "scoring/opt/converter.h"

namespace quickrank {
namespace scoring {

std::string trim(const boost::property_tree::ptree& node, const std::string& label)
{
	std::string res = node.get_child(label).data();
	boost::algorithm::trim(res);
	return res;
}

bool is_leaf(const boost::property_tree::ptree& node)
{
	return node.find("output") != node.not_found();
}

uint32_t find_depth(const boost::property_tree::ptree& split)
{
	uint32_t ld = 0, rd = 0;
	for (auto node: split) {
		if (node.first == "output")
			return 1;
		else if (node.first == "split") {
			std::string pos = node.second.get<std::string>("<xmlattr>.pos");
			if (pos == "left") {
				ld = 1 + find_depth(node.second);
			} else {
				rd = 1 + find_depth(node.second);
			}
		}
	}
	return std::max(ld,rd);
}

struct tree_node {
	boost::property_tree::ptree pnode;
    uint32_t id, pid;
    bool left;

    std::string feature, theta, leaf;

    tree_node(const boost::property_tree::ptree& pnode, uint32_t id, uint32_t pid, bool left, const std::string& parent_f = "")
    	: pnode(pnode)
    	, id(id)
    	, pid (pid)
    	, left(left)
    {
    	if (is_leaf(pnode)) {
    		this->feature = parent_f;
    		this->theta = "";
    		this->leaf = trim(pnode, "output");
    	} else {
    		this->feature = trim(pnode, "feature");
    		this->theta = trim(pnode, "threshold");
    		this->leaf = "";
    	}
    }
};

void generate_opt_trees_input(const std::string& ensemble_file, const std::string& /*output_file*/)
{
	// parse XML
	boost::property_tree::ptree xml_tree;
	std::ifstream is;
	is.open(ensemble_file, std::ifstream::in);
	boost::property_tree::read_xml(is, xml_tree);
	is.close();

	double learning_rate = xml_tree.get_child("ranker.info").get<double>("shrinkage");
	uint32_t num_trees = xml_tree.get_child("ranker.ensemble").count("tree");

	std::cout << num_trees << std::endl; // print number of trees in the ensemble

    // for each tree
	for (const auto& tree_it: xml_tree.get_child("ranker.ensemble")) {

		uint32_t depth = find_depth(tree_it.second.get_child("split")) - 1;
		std::cout << (depth) << std::endl; // print the tree depth

		uint32_t tree_size = std::pow(2, depth) - 1;
		auto split_node = tree_it.second.get_child("split");

		uint32_t local_id = 0;

		std::queue<tree_node> node_queue;

		// breadth first visit
		// TODO: Remove object copies with references
		for (node_queue.push(tree_node(split_node, local_id++, -1, false)); !node_queue.empty(); node_queue.pop()) {
			auto node = node_queue.front();
			if (is_leaf(node.pnode)) {
				if (node.id >= tree_size) {
					std::cout << "leaf"
						  << " " << node.id
						  << " " << node.pid
						  << " " << node.left
						  << " " << (learning_rate * std::stod(node.leaf)) << std::endl;
				} else {
					std::cout << "node"
					    	  << " " << node.id
						  << " " << node.pid
						  << " " << (std::stoi(node.feature) - 1)
						  << " " << node.left
						  << " " << (learning_rate * std::stod(node.leaf)) << std::endl;
				}
			} else {
				if (node.id == 0) {
					std::cout << "root"
							  << " " << node.id
							  << " " << (std::stoi(node.feature) - 1)
							  << " " << node.theta << std::endl; // print the root info
				} else {
					std::cout << "node"
					    	  << " " << node.id
							  << " " << node.pid
							  << " " << (std::stoi(node.feature) - 1)
							  << " " << node.left
							  << " " << node.theta << std::endl;
				}
				for (const auto& split_it: node.pnode) {
					if (split_it.first == "split") {
						std::string pos = split_it.second.get<std::string>("<xmlattr>.pos");
						node_queue.push(tree_node(split_it.second, local_id++, node.id, (pos == "left"), node.feature));
					}
				}
			}
		}

		std:: cout << "end" << std::endl;
	}
}

} // namespace scoring
} // namespace quickrank
