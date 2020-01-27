// Copyright 2014 The Bazel Authors. All rights reserved.
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
#include "src/main/cpp/util/md5.h"
#include "src/main/cpp/util/port.h"
#include "googletest/include/gtest/gtest.h"

namespace blaze_util {

TEST(BlazeUtil, Basic) {
  const char *strs[] = {
    "",
    "a",
    "abc",
    "message digest",
    "abcdefghijklmnopqrstuvwxyz",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
    "12345678901234567890123456789012345678901234567890123456789012345678901234567890"
  };
  const char *md5s[] = {
    "d41d8cd98f00b204e9800998ecf8427e",
    "0cc175b9c0f1b6a831c399e269772661",
    "900150983cd24fb0d6963f7d28e17f72",
    "f96b697d7cb7938d525a2f31aaf161d0",
    "c3fcd3d76192e4007dfb496cca67e13b",
    "d174ab98d277d9f5a5611c2c9f419d9f",
    "57edf4a22be3c955ac49da2e2107b67a",
  };
  unsigned int n = arraysize(strs);
  ASSERT_EQ(n, arraysize(md5s));

  unsigned char buf[17];
  Md5Digest digest;
  for (unsigned int i = 0; i < n; i++) {
    digest.Reset();
    digest.Update(strs[i], strlen(strs[i]));
    digest.Finish(buf);
    ASSERT_EQ(md5s[i], digest.String());
  }
}

}  // namespace blaze_util
