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
#ifndef QUICKRANK_IO_SVML_H_
#define QUICKRANK_IO_SVML_H_

#include <string>
#include <boost/noncopyable.hpp>

#include "data/dataset.h"

namespace quickrank {
namespace io {

/**
 * This class implements IO on Svml files.
 *
 * SVML format is as follows:
 * \verbatim
 <line> .=. <target> qid:<qid> <feature>:<value> <feature>:<value> ... <feature>:<value> # <info>
 <target> .=. <float>
 <qid> .=. <positive integer>
 <feature> .=. <positive integer>
 <value> .=. <float>
 <info> .=. <string>
 \endverbatim

 \todo TODO: handle feature filtering
 */
class Svml : private boost::noncopyable {
 public:
  /// Creates a new Svml IO reader/writer.
  ///
  /// \param k The cut-off threshold.
  Svml() {
  }
  virtual ~Svml() {
  }

  /// Reads the input dataset and returns in horizontal format.
  /// \param filename the input filename.
  /// \return The svml dataset in horizontal format.
  virtual std::unique_ptr<data::Dataset> read_horizontal(
      const std::string &file);

 private:
  double reading_time_ = 0.0;
  double processing_time_ = 0.0;
  long file_size_ = 0;

  /// The output stream operator.
  /// Prints the data reading time stats.
  friend std::ostream& operator<<(std::ostream& os, const Svml& me) {
    return me.put(os);
  }

  /// Prints the data reading time stats
  virtual std::ostream& put(std::ostream& os) const;

};

}  // namespace io
}  // namespace quickrank

#endif
