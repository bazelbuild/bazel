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
#include "src/main/cpp/util/strings.h"

#include <wchar.h>

#include <memory>
#include <string>
#include <vector>

#include "googletest/include/gtest/gtest.h"

namespace blaze_util {

static const char kAsciiLatin[] = {'A', 'b', 'c', '\0'};
static const wchar_t kUtf16Latin[] = {L'A', L'b', L'c', L'\0'};
static const char kUtf8Cyrillic[] = {
    'H',  'e',  'y', '=',  //
    0xd0, 0x9f,            // Cyrillic Capital Letter Pe
    0xd1, 0x80,            // Cyrillic Small Letter Er
    0xd0, 0xb8,            // Cyrillic Small Letter I
    0xd0, 0xb2,            // Cyrillic Small Letter Ve
    0xd0, 0xb5,            // Cyrillic Small Letter Ie
    0xd1, 0x82,            // Cyrillic Small Letter Te
    0,
};
static const wchar_t kUtf16Cyrillic[] = {
    L'H',  L'e', L'y', L'=',
    0x41F,  // Cyrillic Capital Letter Pe
    0x440,  // Cyrillic Small Letter Er
    0x438,  // Cyrillic Small Letter I
    0x432,  // Cyrillic Small Letter Ve
    0x435,  // Cyrillic Small Letter Ie
    0x442,  // Cyrillic Small Letter Te
    0x0,
};

TEST(BlazeUtil, WcsToAcpTest) {
  std::string actual;
  uint32_t win32_err;
  ASSERT_TRUE(WcsToAcp(kUtf16Latin, &actual, &win32_err));
  ASSERT_TRUE(actual == kAsciiLatin);
}

TEST(BlazeUtil, WcsToUtf8Test) {
  std::string actual;
  uint32_t win32_err;
  ASSERT_TRUE(WcsToUtf8(kUtf16Latin, &actual, &win32_err));
  ASSERT_TRUE(actual == kAsciiLatin);
  ASSERT_TRUE(WcsToUtf8(kUtf16Cyrillic, &actual, &win32_err));
  ASSERT_TRUE(actual == kUtf8Cyrillic);
}

TEST(BlazeUtil, AcpToWcsTest) {
  std::wstring actual;
  uint32_t win32_err;
  ASSERT_TRUE(AcpToWcs(kAsciiLatin, &actual, &win32_err));
  ASSERT_TRUE(actual == kUtf16Latin);
}

TEST(BlazeUtil, Utf8ToWcsTest) {
  std::wstring actual;
  uint32_t win32_err;
  ASSERT_TRUE(Utf8ToWcs(kAsciiLatin, &actual, &win32_err));
  ASSERT_TRUE(actual == kUtf16Latin);
  ASSERT_TRUE(Utf8ToWcs(kUtf8Cyrillic, &actual, &win32_err));
  ASSERT_TRUE(actual == kUtf16Cyrillic);
}

}  // namespace blaze_util
