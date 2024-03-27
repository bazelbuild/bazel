// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.bazel.google;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.buildtool.InstrumentationFilterSupport.getInstrumentedPrefix;

import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit test for Bazel coverage. */
@RunWith(JUnit4.class)
public final class CoverageCommandUnitTest extends BuildViewTestCase {
  @Test
  public void testGetInstrumentedPrefixJavatests() throws Exception {
    // Make sure that javatests dir still gets replaced even when immediately under top-level dir
    scratch.file(
        "javatests/com/google/foo/BUILD",
        """
        java_library(
            name = "l",
            srcs = ["foo.java"],
        )
        """);
    String packageName = getConfiguredTarget("//javatests/com/google/foo:l").getLabel()
        .getPackageName();
    assertThat(packageName).isEqualTo("javatests/com/google/foo"); // No leading slashes
    assertThat(getInstrumentedPrefix(packageName)).isEqualTo("java/com/google/foo");
  }

  @Test
  public void testGetInstrumentedPrefix() {
    assertThat(getInstrumentedPrefix("javatests/foo")).isEqualTo("java/foo");
    assertThat(getInstrumentedPrefix("third_party/foo/javatests/foo"))
        .isEqualTo("third_party/foo/java/foo");
    assertThat(getInstrumentedPrefix("third_party/foo/javatest/foo"))
        .isEqualTo("third_party/foo/javatest/foo"); // No substitution of javatest without the s
    assertThat(getInstrumentedPrefix("third_party/foo/src/test/java/foo"))
        .isEqualTo("third_party/foo/src/main/java/foo");
    assertThat(getInstrumentedPrefix("test/java/foo")).isEqualTo("main/java/foo");
    assertThat(getInstrumentedPrefix("foo/internal")).isEqualTo("foo");
    assertThat(getInstrumentedPrefix("foo/public")).isEqualTo("foo");
    assertThat(getInstrumentedPrefix("foo/tests")).isEqualTo("foo");
  }
}
