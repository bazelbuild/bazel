// Copyright 2018 The Bazel Authors. All rights reserved.
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

#include "src/tools/runfiles/runfiles.h"

#include <memory>
#include <string>

#include "gtest/gtest.h"

#define _T(x) #x
#define T(x) _T(x)
#define LINE() T(__LINE__)

namespace bazel {
namespace runfiles {
namespace {

using std::string;
using std::unique_ptr;

TEST(RunfilesTest, DirectoryBasedRunfilesRlocation) {
  string error;
  unique_ptr<Runfiles> r(Runfiles::CreateDirectoryBased("whatever", &error));
  ASSERT_NE(r, nullptr);
  EXPECT_TRUE(error.empty());

  EXPECT_EQ(r->Rlocation("a/b"), "whatever/a/b");
  EXPECT_EQ(r->Rlocation("c/d"), "whatever/c/d");
  EXPECT_EQ(r->Rlocation(""), "");
  EXPECT_EQ(r->Rlocation("foo"), "whatever/foo");
  EXPECT_EQ(r->Rlocation("foo/"), "whatever/foo/");
  EXPECT_EQ(r->Rlocation("foo/bar"), "whatever/foo/bar");
  EXPECT_EQ(r->Rlocation("foo/.."), "");
  EXPECT_EQ(r->Rlocation("/Foo"), "/Foo");
  EXPECT_EQ(r->Rlocation("c:/Foo"), "c:/Foo");
  EXPECT_EQ(r->Rlocation("c:\\Foo"), "c:\\Foo");
}

}  // namespace
}  // namespace runfiles
}  // namespace bazel
