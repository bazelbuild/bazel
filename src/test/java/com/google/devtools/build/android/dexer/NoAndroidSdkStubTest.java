// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.dexer;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Stub test to run if {@code android_sdk_repository} is not in the {@code WORKSPACE} file. */
@RunWith(JUnit4.class)
public class NoAndroidSdkStubTest {
  @Test
  public void printWarningMessageTest() {
    System.out.println(
        "Android tests are being skipped because no android_sdk_repository rule is set up in the "
            + "WORKSPACE file.");
  }
}
