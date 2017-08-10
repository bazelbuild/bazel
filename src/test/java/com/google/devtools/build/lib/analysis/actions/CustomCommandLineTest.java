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
package com.google.devtools.build.lib.analysis.actions;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine.Builder;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine.VectorArg;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.vfs.PathFragment;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class CustomCommandLineTest extends BuildViewTestCase {

  @Test
  public void testAddBeforeEachPath() {
    CustomCommandLine commandLine =
        new Builder()
            .add("foo")
            .add(
                VectorArg.of(
                        ImmutableList.of(
                            PathFragment.create("/path1"), PathFragment.create("/path2")))
                    .beforeEach("-I"))
            .add("bar")
            .add(VectorArg.of(ImmutableList.<PathFragment>of()).beforeEach("-I"))
            .add("baz")
            .build();
    assertThat(commandLine.arguments())
        .containsExactly("foo", "-I", "/path1", "-I", "/path2", "bar", "baz")
        .inOrder();
  }

  @Test
  public void testAddBeforeEach() {
    CustomCommandLine commandLine =
        new Builder()
            .add("foo")
            .add(VectorArg.of(ImmutableList.<String>of()).beforeEach("-D"))
            .add("bar")
            .add(
                VectorArg.of(ImmutableList.of("DEBUG=42", "ENABLE_QUANTUM", "__OBJC__"))
                    .beforeEach("-D"))
            .add("baz")
            .build();
    assertThat(commandLine.arguments())
        .containsExactly(
            "foo", "bar", "-D", "DEBUG=42", "-D", "ENABLE_QUANTUM", "-D", "__OBJC__", "baz")
        .inOrder();
  }

  @Test
  public void testAddBeforeEachExecPath() throws Exception {
    CustomCommandLine commandLine =
        new Builder()
            .add("foo")
            .add(
                VectorArg.of(
                        ImmutableList.of(
                            getSourceArtifact("pkg/util.a"), getSourceArtifact("pkg2/extra.a")))
                    .beforeEach("-l"))
            .add("bar")
            .add(VectorArg.of(ImmutableList.<Artifact>of()).beforeEach("-l"))
            .add("baz")
            .build();
    assertThat(commandLine.arguments())
        .containsExactly("foo", "-l", "pkg/util.a", "-l", "pkg2/extra.a", "bar", "baz")
        .inOrder();
  }

  @Test
  public void testAddFormatEach() {
    CustomCommandLine commandLine =
        new Builder()
            .add("foo")
            .add(VectorArg.of(ImmutableList.<String>of()).formatEach("-X'%s'"))
            .add("bar")
            .add(VectorArg.of(ImmutableList.of("42", "1011")).formatEach("-X'%s'"))
            .add("baz")
            .build();
    assertThat(commandLine.arguments())
        .containsExactly("foo", "bar", "-X'42'", "-X'1011'", "baz")
        .inOrder();
  }
}
