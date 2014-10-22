#ifndef QUICKRANK_IO_SVML_H_
#define QUICKRANK_IO_SVML_H_

#include <string>
#include <boost/noncopyable.hpp>

#include "data/ltrdata.h"
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
class Svml : private boost::noncopyable
{
 public:
  /// Creates a new Svml IO reader/writer.
  ///
  /// \param k The cut-off threshold.
  explicit Svml() {}
  virtual ~Svml() {}

  /// Reads the input dataset and returns in vertical format.
  /// \param filename the input filename.
  /// \return The svml dataset in vertical format.
  /// \todo TODO: add smart pointer here
  virtual LTR_VerticalDataset* read_vertical(const std::string &filename) const;


  /// Reads the input dataset and returns in horizontal format.
  /// \param filename the input filename.
  /// \return The svml dataset in horizontal format.
  virtual std::unique_ptr<data::Dataset> read_horizontal(const std::string &file) const;
};

} // namespace data
} // namespace quickrank


#endif
