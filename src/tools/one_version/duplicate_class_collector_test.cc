// Copyright 2024 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "src/tools/one_version/duplicate_class_collector.h"

#include <cstddef>
#include <string>
#include <vector>

#include "googletest/include/gtest/gtest.h"
#include "absl/strings/str_cat.h"

namespace one_version {

// The fixture is a friend of DuplicateClassCollector. Friendship is not
// inherited by the per-test subclasses, so private state is exposed to test
// bodies through static helpers here.
class DuplicateClassCollectorTest : public ::testing::Test {
 protected:
  static size_t ArenaBlockCount(const DuplicateClassCollector& vc) {
    return vc.arena_blocks_.size();
  }
};

TEST_F(DuplicateClassCollectorTest, NoDuplicates) {
  DuplicateClassCollector vc;
  vc.Add("com/google/Foo", 1, Label("//hello:foo", "hello/libfoo.jar"));
  vc.Add("com/google/Bar", 2, Label("//hello:bar", "hello/libbaz.jar"));
  vc.Add("com/google/Baz", 3, Label("//hello:baz", "hello/libbaz.jar"));
  vc.Add("com/google/Moo", 4, Label("//hello:moo", "hello/libmoo.jar"));
  EXPECT_TRUE(vc.Violations().empty());
}

TEST_F(DuplicateClassCollectorTest, Duplicates) {
  DuplicateClassCollector vc;
  vc.Add("com/google/Foo", 1, Label("//hello:foo", "hello/libfoo.jar"));
  vc.Add("com/google/Foo", 1, Label("//hello:bar", "hello/libbar.jar"));
  vc.Add("com/google/Baz", 3, Label("//hello:baz", "hello/libbaz.jar"));
  vc.Add("com/google/Moo", 4, Label("//hello:moo", "hello/libmoo.jar"));
  EXPECT_TRUE(vc.Violations().empty());
}

TEST_F(DuplicateClassCollectorTest, Violations) {
  DuplicateClassCollector vc;
  vc.Add("com/google/Foo", 2, Label("//hello:bar", "hello/libbar.jar"));
  vc.Add("com/google/Foo", 1, Label("//hello:foo", "hello/libfoo.jar"));
  vc.Add("com/google/Baz", 3, Label("//hello:baz", "hello/libbaz.jar"));
  vc.Add("com/google/Moo", 4, Label("//hello:moo", "hello/libmoo.jar"));
  std::string expected =
      "  com/google/Foo has incompatible definitions in:\n"
      "    crc32=1\n"
      "      //hello:foo [new]\n"
      "      via hello/libfoo.jar\n"
      "    crc32=2\n"
      "      //hello:bar [new]\n"
      "      via hello/libbar.jar\n";
  EXPECT_EQ(expected, DuplicateClassCollector::Report(vc.Violations()));
}

// Labels are interned by comparing against the most recently added label.
// Consecutive labels that differ in only one field (name, jar, or allowlisted)
// must each be stored, otherwise entries would be attributed to the previous
// label.
TEST_F(DuplicateClassCollectorTest, ConsecutiveLabelsDifferingInOneField) {
  DuplicateClassCollector vc;
  // Same jar and allowlisting, different name.
  vc.Add("com/google/Foo", 1, Label("//hello:a", "hello/lib.jar"));
  vc.Add("com/google/Foo", 2, Label("//hello:b", "hello/lib.jar"));
  // Same name and allowlisting, different jar.
  vc.Add("com/google/Bar", 1, Label("//hello:c", "hello/libc1.jar"));
  vc.Add("com/google/Bar", 2, Label("//hello:c", "hello/libc2.jar"));
  // Same name and jar, different allowlisting.
  vc.Add("com/google/Baz", 1,
         Label("//hello:d", "hello/libd.jar", /*allowlisted=*/false));
  vc.Add("com/google/Baz", 2,
         Label("//hello:d", "hello/libd.jar", /*allowlisted=*/true));
  std::string expected =
      "  com/google/Bar has incompatible definitions in:\n"
      "    crc32=1\n"
      "      //hello:c [new]\n"
      "      via hello/libc1.jar\n"
      "    crc32=2\n"
      "      //hello:c [new]\n"
      "      via hello/libc2.jar\n"
      "  com/google/Baz has incompatible definitions in:\n"
      "    crc32=1\n"
      "      //hello:d [new]\n"
      "      via hello/libd.jar\n"
      "    crc32=2\n"
      "      //hello:d [allowlisted]\n"
      "      via hello/libd.jar\n"
      "  com/google/Foo has incompatible definitions in:\n"
      "    crc32=1\n"
      "      //hello:a [new]\n"
      "      via hello/lib.jar\n"
      "    crc32=2\n"
      "      //hello:b [new]\n"
      "      via hello/lib.jar\n";
  EXPECT_EQ(expected, DuplicateClassCollector::Report(vc.Violations()));
}

// An empty class name (a jar entry named exactly ".class") as the very first
// entry must not touch the arena before a block has been allocated, and must
// still be tracked like any other name.
TEST_F(DuplicateClassCollectorTest, EmptyClassName) {
  DuplicateClassCollector vc;
  vc.Add("", 1, Label("//hello:foo", "hello/libfoo.jar"));
  vc.Add("", 2, Label("//hello:bar", "hello/libbar.jar"));
  vc.Add("com/google/Baz", 3, Label("//hello:baz", "hello/libbaz.jar"));
  std::vector<Violation> violations = vc.Violations();
  ASSERT_EQ(1, violations.size());
  EXPECT_EQ("", violations[0].class_name());
  EXPECT_EQ(2, violations[0].versions().size());
}

// Class names are copied into 1 MiB arena blocks. A name that does not fit in
// the remaining space of the current block must roll over into a new block,
// and the space it consumes must be accounted for so the following name rolls
// over again instead of overrunning the block. The names must survive intact.
TEST_F(DuplicateClassCollectorTest, ArenaRollover) {
  constexpr size_t kBlockSize = 1 << 20;
  const std::string big_name(kBlockSize, 'x');
  DuplicateClassCollector vc;
  EXPECT_EQ(0, ArenaBlockCount(vc));

  // The first name allocates the first block and takes a few bytes of it.
  vc.Add("com/google/Foo", 1, Label("//hello:foo", "hello/libfoo.jar"));
  EXPECT_EQ(1, ArenaBlockCount(vc));

  // A block-sized name no longer fits in the first block and gets a dedicated
  // block, which it fills completely.
  vc.Add(big_name, 1, Label("//hello:big", "hello/libbig.jar"));
  EXPECT_EQ(2, ArenaBlockCount(vc));

  // Nothing is left in the second block, so the next name needs a third one.
  vc.Add("com/google/Bar", 1, Label("//hello:bar", "hello/libbar.jar"));
  EXPECT_EQ(3, ArenaBlockCount(vc));

  // Subsequent small names fit in the third block.
  vc.Add("com/google/Baz", 1, Label("//hello:baz", "hello/libbaz.jar"));
  EXPECT_EQ(3, ArenaBlockCount(vc));

  // Existing names are found (no new arena copy) and were not corrupted by the
  // rollovers: conflicting crc32s surface them in the (sorted) violations.
  vc.Add("com/google/Foo", 2, Label("//hello:foo2", "hello/libfoo2.jar"));
  vc.Add(big_name, 2, Label("//hello:big2", "hello/libbig2.jar"));
  vc.Add("com/google/Bar", 2, Label("//hello:bar2", "hello/libbar2.jar"));
  EXPECT_EQ(3, ArenaBlockCount(vc));
  std::vector<Violation> violations = vc.Violations();
  ASSERT_EQ(3, violations.size());
  EXPECT_EQ("com/google/Bar", violations[0].class_name());
  EXPECT_EQ("com/google/Foo", violations[1].class_name());
  EXPECT_EQ(big_name, violations[2].class_name());
}

// A class seen with several crc32s, including repeats, must be reported
// exactly once (the collector records a class as conflicting only the first
// time a second crc32 is seen), with all versions and labels present.
TEST_F(DuplicateClassCollectorTest, RepeatedConflicts) {
  DuplicateClassCollector vc;
  // Two entries with the same crc32 before the first conflict ...
  vc.Add("com/google/Foo", 2, Label("//hello:a", "hello/liba.jar"));
  vc.Add("com/google/Foo", 2, Label("//hello:b", "hello/libb.jar"));
  // ... the first conflict ...
  vc.Add("com/google/Foo", 1, Label("//hello:c", "hello/libc.jar"));
  // ... another entry matching the first crc32, and a third crc32.
  vc.Add("com/google/Foo", 2, Label("//hello:d", "hello/libd.jar"));
  vc.Add("com/google/Foo", 3, Label("//hello:e", "hello/libe.jar"));
  vc.Add("com/google/Baz", 3, Label("//hello:baz", "hello/libbaz.jar"));
  std::vector<Violation> violations = vc.Violations();
  ASSERT_EQ(1, violations.size());
  EXPECT_EQ("com/google/Foo", violations[0].class_name());
  EXPECT_EQ(3, violations[0].versions().size());
  std::string expected =
      "  com/google/Foo has incompatible definitions in:\n"
      "    crc32=1\n"
      "      //hello:c [new]\n"
      "      via hello/libc.jar\n"
      "    crc32=2\n"
      "      //hello:a [new]\n"
      "      via hello/liba.jar\n"
      "      //hello:b [new]\n"
      "      via hello/libb.jar\n"
      "      //hello:d [new]\n"
      "      via hello/libd.jar\n"
      "    crc32=3\n"
      "      //hello:e [new]\n"
      "      via hello/libe.jar\n";
  EXPECT_EQ(expected, DuplicateClassCollector::Report(violations));
}

}  // namespace one_version
