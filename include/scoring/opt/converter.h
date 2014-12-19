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
#ifndef QUICKRANK_SCORING_OPT_CONVERTER_H
#define QUICKRANK_SCORING_OPT_CONVERTER_H

namespace quickrank {
namespace scoring {

  void generate_opt_trees_input(const std::string &ensemble_file, const std::string &output_file);

} // namespace scoring
} // namesapce quickrank

#endif // QUICKRANK_SCORING_OPT_CONVERTER_H
