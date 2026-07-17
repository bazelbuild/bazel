// Copyright 2023 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.docgen;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableMap;
import java.io.File;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link SourceUrlMapper}. */
@RunWith(JUnit4.class)
public final class SourceUrlMapperTest {

  SourceUrlMapper mapper =
      new SourceUrlMapper(
          "https://example.com/",
          "/tmp/io_bazel",
          ImmutableMap.of(
              "@//",
              "https://example.com/",
              "@_builtins//",
              "https://example.com/src/main/starlark/builtins_bzl/"));

  @Test
  public void urlOfFile() {
    assertThat(mapper.urlOfFile(new File("/tmp/io_bazel/src/FooBar.java")))
        .isEqualTo("https://example.com/src/FooBar.java");
  }

  @Test
  public void urlOfFile_throwsIfFileNotUnderSourceRoot() {
    IllegalArgumentException e =
        assertThrows(
            IllegalArgumentException.class,
            () -> mapper.urlOfFile(new File("/tmp/io_bazel_src/FooBar.java")));
    assertThat(e)
        .hasMessageThat()
        .contains("File '/tmp/io_bazel_src/FooBar.java' is expected to be under '/tmp/io_bazel'");
  }

  @Test
  public void urlOfLabel() {
    assertThat(mapper.urlOfLabel("@_builtins//:foo/bar.bzl"))
        .isEqualTo("https://example.com/src/main/starlark/builtins_bzl/foo/bar.bzl");
    assertThat(mapper.urlOfLabel("//not/in/builtins/foo:bar.bzl"))
        .isEqualTo("https://example.com/not/in/builtins/foo/bar.bzl");
  }
}
