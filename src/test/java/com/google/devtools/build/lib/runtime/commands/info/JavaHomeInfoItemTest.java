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

package com.google.devtools.build.lib.runtime.commands.info;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.ISO_8859_1;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class JavaHomeInfoItemTest {
  @Test
  public void missingJavaHomeReportsUnknown() throws Exception {
    String javaHome = System.clearProperty("java.home");
    try {
      assertThat(new JavaHomeInfoItem().get(/* configurationSupplier= */ null, /* env= */ null))
          .isEqualTo("unknown\n".getBytes(ISO_8859_1));
    } finally {
      if (javaHome != null) {
        System.setProperty("java.home", javaHome);
      }
    }
  }
}
