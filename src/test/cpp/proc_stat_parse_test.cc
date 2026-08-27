// Copyright 2026 The Bazel Authors. All rights reserved.
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

#include <string>

#include "src/main/cpp/blaze_util_platform.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace {

std::string MakeStatLine(absl::string_view comm) {
  return absl::StrCat(
      "12345 (", comm,
      ") S 1 12345 12345 34816 12345 4194560 111 0 222 0 333 444 0 0 20 0 1 0 "
      "424242 10000 500\n");
}

TEST(ProcStatParse, ValidStatLine) {
  std::string start_time;
  for (const auto* comm : {"java", "blaze(a b)", ") 7 6 5 4 3 2 1", "a  b"}) {
    EXPECT_TRUE(blaze::ParseProcStat(MakeStatLine(comm), &start_time));
    EXPECT_EQ(start_time, "424242");
  }
}

TEST(ProcStatParse, InvalidStatLine) {
  std::string start_time;
  EXPECT_FALSE(blaze::ParseProcStat("12345 java S 1 12345", &start_time));
  EXPECT_FALSE(blaze::ParseProcStat("12345 (java) S 1 2 3", &start_time));
}

TEST(ProcStatParse, ZombieDiagnosis) {
  std::string statline =
      "12345 (blaze) Z 1 12345 12345 34816 12345 4194560 111 0 222 0 333 444 0 "
      "0 20 0 1 0 424242 10000 500\n";
  std::string diagnosis = blaze::ParseProcStatDiagnosis(statline, 12345);
  EXPECT_NE(diagnosis.find("is a zombie process (state 'Z')"),
            std::string::npos);
  EXPECT_NE(diagnosis.find("parent process (pid=1)"), std::string::npos);
}

TEST(ProcStatParse, UninterruptibleSleepDiagnosis) {
  std::string statline =
      "259127 (blaze) D 5360 259127 259127 0 -1 4194368 1690 657 0 0 45 1 0 0 "
      "20 0 2 0 8509179 15511289856 196892\n";
  std::string diagnosis = blaze::ParseProcStatDiagnosis(statline, 259127);
  EXPECT_NE(
      diagnosis.find("is in uninterruptible disk/kernel sleep (state 'D')"),
      std::string::npos);
}

TEST(ProcStatParse, OtherStateDiagnosis) {
  std::string statline =
      "12345 (blaze) S 1 12345 12345 34816 12345 4194560 111 0 222 0 333 444 0 "
      "0 20 0 1 0 424242 10000 500\n";
  std::string diagnosis = blaze::ParseProcStatDiagnosis(statline, 12345);
  EXPECT_EQ(diagnosis, "Process state: S (parent pid=1).");
}

}  // namespace
