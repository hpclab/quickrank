/*
 * Instance.h
 *
 *  Created on: Oct 3, 2014
 *      Author: claudio
 */

#ifndef INSTANCE_H_
#define INSTANCE_H_

#include <iostream>
#include <string>
#include <boost/noncopyable.hpp>

#include "types.h"

namespace quickrank {
namespace io {

class Instance : private boost::noncopyable {
 public:
  explicit Instance() : label_(0) {}
  virtual ~Instance() {}

  const std::string& comment() const { return comment_; }
  void set_comment(const std::string& comment) { comment_ = comment; }

  Feature get_feature(unsigned int i) const { return (i < features_.size() ? features_[i] : 0); }

  void set_features(unsigned int i, const Feature& f) {
    if (i >= features_.size())
      features_.resize(i + 1, 0);
    features_[i] = f;
  }

  Label getLabel() const {
    return label_;
  }

  void setLabel(Label label) {
    label_ = label;
  }

 private:
  friend std::ostream& operator<<(std::ostream&, const Instance&);
  friend std::istream& operator>>(std::istream&, Instance&);

  FeatureVector features_;
  Label label_;
  std::string comment_;
};

} // namespace io
} // namespace quickrank

#endif /* INSTANCE_H_ */
