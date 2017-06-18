// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.packages;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.packages.License.LicenseType;
import java.util.Arrays;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class LicenseTest {

  @Test
  public void testLeastRestrictive() {
    assertThat(License.leastRestrictive(Arrays.asList(LicenseType.RESTRICTED)))
        .isEqualTo(LicenseType.RESTRICTED);
    assertThat(
            License.leastRestrictive(
                Arrays.asList(LicenseType.RESTRICTED, LicenseType.BY_EXCEPTION_ONLY)))
        .isEqualTo(LicenseType.RESTRICTED);
    assertThat(License.leastRestrictive(Arrays.<LicenseType>asList()))
        .isEqualTo(LicenseType.BY_EXCEPTION_ONLY);
  }
}
