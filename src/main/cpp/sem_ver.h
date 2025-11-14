#ifndef BAZEL_SRC_MAIN_CPP_SEMVER_H_
#define BAZEL_SRC_MAIN_CPP_SEMVER_H_

#include <iostream>
#include <optional>
#include <string>
#include <utility>

#include "absl/strings/string_view.h"

namespace blaze {

// Represents a semantic version.
class SemVer {
public:
  // Parse a string into a SemVer. Must be a valid semantic version string per
  // semver.org, otherwise, returns nullopt.
  static std::optional<SemVer> Parse(const std::string &v);

  // Returns a new SemVer with the next major version.  Eg. 8.3.4 -> 9.0.0.
  SemVer NextMajorVersion() const;

  // Returns a new SemVer with the next minor version. Eg. 8.3.4 -> 8.4.0.
  SemVer NextMinorVersion() const;

  bool operator==(const SemVer& other) const;

  bool operator<(const SemVer& other) const;

  bool operator!=(const SemVer& other) const;

  bool operator>(const SemVer& other) const;

  bool operator<=(const SemVer& other) const;

  bool operator>=(const SemVer& other) const;

private:
  // Private constructor - use Parse() instead.
  explicit SemVer(int major, int minor, int patch,
                  std::string prerelease,
                  std::string buildmetadata)
      : major_(major), minor_(minor), patch_(patch), prerelease_(std::move(prerelease)),
        buildmetadata_(std::move(buildmetadata)) {};

  // Comparator for SemVer.
  [[nodiscard]] int Compare(const SemVer &other) const;

  // Comparator function for prerelease strings.
  static int ComparePrerelease(absl::string_view x, absl::string_view y);

  // The individual semantic version parts.
  int major_;
  int minor_;
  int patch_;
  std::string prerelease_;
  std::string buildmetadata_;
};

} // namespace blaze
#endif // BAZEL_SRC_MAIN_CPP_SEMVER_H_
