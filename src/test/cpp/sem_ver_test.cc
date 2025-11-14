#include "googletest/include/gtest/gtest.h"
#include "src/main/cpp/sem_ver.h"

#include "absl/strings/str_format.h"

namespace blaze {

TEST(SemVerTest, ComparisonOperators) {
  // Adapted from
  // https://cs.opensource.google/go/x/mod/+/refs/tags/v0.30.0:semver/semver_test.go;l=19
  // This list of semantic version strings are in order of precedence.

  // clang-format off: preserve one test version per line.
  std::vector<std::string> test_versions = {
    "1.0.0-alpha",
    "1.0.0-alpha.1",
    "1.0.0-alpha.beta",
    "1.0.0-beta",
    "1.0.0-beta.2",
    "1.0.0-beta.11",
    "1.0.0-rc.1",
    "1.0.0",
    "1.2.0",
    "1.2.3-456",
    "1.2.3-456.789",
    "1.2.3-456-789",
    "1.2.3-456a",
    "1.2.3-pre",
    "1.2.3-pre.1",
    "1.2.3-zzz",
    "1.2.3",
  };
  // clang-format on

  // Double loop over the test versions.  The index of each indicates the
  // precedence and the expected value.
  for (std::size_t i = 0; i < test_versions.size(); i++) {
    for (std::size_t j = 0; j < test_versions.size(); j++) {
      const std::string& is = test_versions[i];
      const std::string& js = test_versions[j];

      auto iv = SemVer::Parse(is);
      auto jv = SemVer::Parse(js);
      ASSERT_TRUE(iv.has_value()) << is;
      ASSERT_TRUE(jv.has_value()) << js;

      auto message = absl::StrFormat("comparing '%s' with '%s'", is, js);
      if (is == js) {
        EXPECT_TRUE(iv == jv) << message;
        EXPECT_FALSE(iv != jv) << message;
      } else if (i < j) {
        EXPECT_TRUE(iv < jv) << message;
        EXPECT_FALSE(iv >= jv) << message;
      } else {
        EXPECT_TRUE(iv > jv) << message;
        EXPECT_FALSE(iv <= jv) << message;
      }
    }
  }
}

TEST(SemVerTest, NextMajorVersion) {
  EXPECT_EQ(SemVer::Parse("0.0.0")->NextMajorVersion(), SemVer::Parse("1.0.0"));
  EXPECT_EQ(SemVer::Parse("8.0.0")->NextMajorVersion(), SemVer::Parse("9.0.0"));
  EXPECT_EQ(SemVer::Parse("8.2.4")->NextMajorVersion(), SemVer::Parse("9.0.0"));
  EXPECT_EQ(SemVer::Parse("8.2.4-beta")->NextMajorVersion(),
            SemVer::Parse("9.0.0"));
}

TEST(SemVerTest, NextMinorVersion) {
  EXPECT_EQ(SemVer::Parse("0.0.0")->NextMinorVersion(), SemVer::Parse("0.1.0"));
  EXPECT_EQ(SemVer::Parse("8.0.0")->NextMinorVersion(), SemVer::Parse("8.1.0"));
  EXPECT_EQ(SemVer::Parse("8.2.4")->NextMinorVersion(), SemVer::Parse("8.3.0"));
  EXPECT_EQ(SemVer::Parse("8.2.4-beta")->NextMinorVersion(),
            SemVer::Parse("8.3.0"));
}

} // namespace blaze
