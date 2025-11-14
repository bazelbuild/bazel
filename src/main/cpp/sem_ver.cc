#include "src/main/cpp/sem_ver.h"

#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include <optional>
#include <regex>
#include <string>

namespace blaze {
namespace {
// Semantic version regex copied verbatim from
// https://semver.org/#is-there-a-suggested-regular-expression-regex-to-check-a-semver-string
const std::regex kSemverRe(
    R"(^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$)");
} // namespace

/*static*/ std::optional<SemVer> SemVer::Parse(const std::string &v) {
  if (std::smatch m; std::regex_match(v, m, kSemverRe)) {
    return SemVer(std::stoi(m[1]), std::stoi(m[2]), std::stoi(m[3]), m[4],
                  m[5]);
  }
  return std::nullopt;
}

SemVer SemVer::NextMajorVersion() const {
  return SemVer(major_ + 1, 0, 0, "", "");
}

SemVer SemVer::NextMinorVersion() const {
  return SemVer(major_, minor_ + 1, 0, "", "");
}

int SemVer::Compare(const SemVer &other) const {
  auto major_diff = major_ - other.major_;
  if (major_diff != 0) {
    return major_diff;
  }
  auto minor_diff = minor_ - other.minor_;
  if (minor_diff != 0) {
    return minor_diff;
  }
  auto patch_diff = patch_ - other.patch_;
  if (patch_diff != 0) {
    return patch_diff;
  }
  return ComparePrerelease(prerelease_, other.prerelease_);
}

// Adapted from golang's implementation.
// https://cs.opensource.google/go/x/mod/+/refs/tags/v0.30.0:semver/semver.go;l=339;bpv=1
/*static*/ int SemVer::ComparePrerelease(absl::string_view x,
                                         absl::string_view y) {
  if (x == y) {
    return 0;
  }
  // A larger set of pre-release fields has a higher precedence than a smaller
  // set.
  if (x.empty()) {
    return 1;
  }
  if (y.empty()) {
    return -1;
  }

  // Examine each dot-separated part.
  const std::vector<absl::string_view> xv = absl::StrSplit(x, ".");
  const std::vector<absl::string_view> yv = absl::StrSplit(y, ".");
  std::size_t i = 0;
  while (i != xv.size() && i != yv.size()) {
    auto part_x = xv[i];
    auto part_y = yv[i];
    i++;
    if (part_x != part_y) {
      int ix;
      int iy;
      bool is_numeric_x = absl::SimpleAtoi(part_x, &ix);
      bool is_numeric_y = absl::SimpleAtoi(part_y, &iy);

      if (is_numeric_x != is_numeric_y) {
        // Numeric identifiers always have lower precedence than non-numeric
        // identifiers.
        if (is_numeric_x) {
          return -1;
        }
        return 1;
      }

      if (is_numeric_x) { // Both parts are numbers.
        if (ix < iy) {
          return -1;
        }
        return 1;
      }

      // Lexicographic ordering.
      if (part_x < part_y) {
        return -1;
      }
      return 1;
    }
  }
  // A larger set of pre-release fields has a higher precedence than a smaller
  // set.
  if (i == xv.size()) { // We ran out of x parts.
    return -1;
  }
  return 1;
}

bool SemVer::operator==(const SemVer &other) const {
  return Compare(other) == 0;
}

bool SemVer::operator<(const SemVer &other) const { return Compare(other) < 0; }

bool SemVer::operator!=(const SemVer &other) const {
  return Compare(other) != 0;
}

bool SemVer::operator>(const SemVer &other) const { return Compare(other) > 0; }

bool SemVer::operator<=(const SemVer &other) const {
  return Compare(other) <= 0;
}

bool SemVer::operator>=(const SemVer &other) const {
  return Compare(other) >= 0;
}

} // namespace blaze
