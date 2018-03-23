// Copyright 2016 The Bazel Authors. All rights reserved.
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
#include "src/main/cpp/util/numbers.h"
#include "src/main/cpp/util/port.h"
#include "googletest/include/gtest/gtest.h"

namespace blaze_util {

TEST(NumbersTest, TestSafeStrto32) {
  int value;
  ASSERT_TRUE(safe_strto32("0", &value));
  ASSERT_EQ(0, value);
  ASSERT_TRUE(safe_strto32("42", &value));
  ASSERT_EQ(42, value);
  ASSERT_TRUE(safe_strto32("007", &value));
  ASSERT_EQ(7, value);
  ASSERT_TRUE(safe_strto32("1234567", &value));
  ASSERT_EQ(1234567, value);
  ASSERT_TRUE(safe_strto32("-0", &value));
  ASSERT_EQ(0, value);
  ASSERT_TRUE(safe_strto32("-273", &value));
  ASSERT_EQ(-273, value);
  ASSERT_TRUE(safe_strto32("-0420", &value));
  ASSERT_EQ(-420, value);
}

}  // namespace blaze_util
