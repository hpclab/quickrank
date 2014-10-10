#ifndef QUICKRANK_IO_SVML_H_
#define QUICKRANK_IO_SVML_H_

#include <boost/noncopyable.hpp>

#include "data/ltrdata.h"


namespace qr {
namespace io {

/**
 * This class implements IO on Svml files.
 */
class Svml : private boost::noncopyable
{
 public:
  /// Creates a new Svml IO reader/writer.
  ///
  /// \param k The cut-off threshold.
  explicit Svml() {}
  virtual ~Svml() {};

  /// Reads the input dataset and returns in vertical format.
  /// \param filename the input filename.
  /// \return The svml dataset in vertical format.
  /// \todo TODO: add smart pointer here
  virtual LTR_VerticalDataset* read_vertical(const char *filename) const;

};

} // namespace data
} // namespace qr


#endif
