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

package com.google.devtools.build.lib.rules.java;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.vfs.PathFragment;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link JavaUtil} methods. */
@RunWith(JUnit4.class)
public class JavaUtilTest {

  @Test
  public void testGetJavaPath() {
    assertThat(
            JavaUtil.getJavaPath(PathFragment.create("java/com/google/foo/FooModule.java"))
                .getPathString())
        .isEqualTo("com/google/foo/FooModule.java");
    assertThat(JavaUtil.getJavaPath(PathFragment.create("org/foo/FooUtil.java"))).isNull();
  }

  @Test
  public void testDetokenization() {
    ImmutableList<String> options =
        ImmutableList.of(
            "-source",
            "8",
            "-target",
            "8",
            "-Xmx1G",
            "--arg=val",
            "-XepExcludedPaths:.*/\\\\$$?\\\\$$?AutoValue(Gson)?_.*\\.java");

    NestedSet<String> detokenized = JavaHelper.detokenizeJavaOptions(options);
    ImmutableList<String> retokenized = JavaHelper.tokenizeJavaOptions(detokenized);

    assertThat(detokenized.toList())
        .containsExactly(
            "-source 8 -target 8 -Xmx1G '--arg=val'"
                + " '-XepExcludedPaths:.*/\\\\$$?\\\\$$?AutoValue(Gson)?_.*\\.java'");
    assertThat(retokenized).isEqualTo(options);
  }
}
