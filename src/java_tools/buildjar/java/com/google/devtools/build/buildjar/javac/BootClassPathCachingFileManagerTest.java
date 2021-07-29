// Copyright 2021 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.buildjar.javac;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.protobuf.ByteString;
import com.sun.tools.javac.util.Context;
import java.io.IOException;
import java.nio.file.Path;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** {@link BootClassPathCachingFileManager}Test */
@RunWith(JUnit4.class)
public class BootClassPathCachingFileManagerTest {
  private ImmutableList<Path> bootClassPathsCandidates;
  private BootClassPathCachingFileManager bootFileManager;

  @Rule public final TemporaryFolder temporaryFolder = new TemporaryFolder();

  @Before
  public void createBootFileManager() throws IOException {
    bootClassPathsCandidates =
        ImmutableList.of(
            temporaryFolder.newFolder().toPath().resolve("BootClassPath0.jar"),
            temporaryFolder.newFolder().toPath().resolve("BootClassPath1.jar"),
            temporaryFolder.newFolder().toPath().resolve("BootClassPath2.jar"));

    ImmutableList<Path> bootClassPaths =
        ImmutableList.of(bootClassPathsCandidates.get(0), bootClassPathsCandidates.get(1));

    ImmutableMap<String, ByteString> inputsAndDigest =
        ImmutableMap.<String, ByteString>builder()
            .put(
                bootClassPaths.get(0).toString(),
                ByteString.copyFromUtf8(bootClassPaths.get(0).toString()))
            .put(
                bootClassPaths.get(1).toString(),
                ByteString.copyFromUtf8(bootClassPaths.get(1).toString()))
            .build();

    BlazeJavacArguments arguments =
        BlazeJavacArguments.builder()
            .bootClassPath(bootClassPaths)
            .inputsAndDigest(inputsAndDigest)
            .classOutput(temporaryFolder.newFolder().toPath().resolve("output_0.jar"))
            .build();

    bootFileManager = new BootClassPathCachingFileManager(new Context(), arguments);
  }

  @Test
  public void testNeedsUpdate() throws IOException {
    ImmutableList<Path> bootClassPaths =
        ImmutableList.of(
            bootClassPathsCandidates.get(0),
            bootClassPathsCandidates.get(1),
            bootClassPathsCandidates.get(2));

    ImmutableMap<String, ByteString> inputsAndDigest =
        ImmutableMap.<String, ByteString>builder()
            .put(
                bootClassPaths.get(0).toString(),
                ByteString.copyFromUtf8(bootClassPaths.get(0).toString()))
            .put(
                bootClassPaths.get(1).toString(),
                ByteString.copyFromUtf8(bootClassPaths.get(1).toString()))
            .put(
                bootClassPaths.get(2).toString(),
                ByteString.copyFromUtf8(bootClassPaths.get(2).toString()))
            .build();

    BlazeJavacArguments arguments =
        BlazeJavacArguments.builder()
            .bootClassPath(bootClassPaths)
            .inputsAndDigest(inputsAndDigest)
            .classOutput(temporaryFolder.newFolder().toPath().resolve("output_1.jar"))
            .build();

    assertThat(bootFileManager.needsUpdate(arguments)).isFalse();
  }

  @Test
  public void testNeedsUpdate_withDifferentDigest() throws IOException {
    ImmutableList<Path> bootClassPaths =
        ImmutableList.of(bootClassPathsCandidates.get(0), bootClassPathsCandidates.get(1));

    ImmutableMap<String, ByteString> inputsAndDigest =
        ImmutableMap.<String, ByteString>builder()
            .put(bootClassPaths.get(0).toString(), ByteString.copyFromUtf8("different digest"))
            .put(
                bootClassPaths.get(1).toString(),
                ByteString.copyFromUtf8(bootClassPaths.get(1).toString()))
            .build();

    BlazeJavacArguments arguments =
        BlazeJavacArguments.builder()
            .bootClassPath(bootClassPaths)
            .inputsAndDigest(inputsAndDigest)
            .classOutput(temporaryFolder.newFolder().toPath().resolve("output_1.jar"))
            .build();

    assertThat(bootFileManager.needsUpdate(arguments)).isTrue();
  }

  @Test
  public void testAreArgumentsValid_withOneMissingDigest() throws IOException {
    ImmutableList<Path> bootClassPaths =
        ImmutableList.of(bootClassPathsCandidates.get(0), bootClassPathsCandidates.get(1));

    ImmutableMap<String, ByteString> inputsAndDigest =
        ImmutableMap.of(
            bootClassPaths.get(1).toString(),
            ByteString.copyFromUtf8(bootClassPaths.get(1).toString()));

    BlazeJavacArguments arguments =
        BlazeJavacArguments.builder()
            .bootClassPath(bootClassPaths)
            .inputsAndDigest(inputsAndDigest)
            .classOutput(temporaryFolder.newFolder().toPath().resolve("output_1.jar"))
            .build();

    assertThat(BootClassPathCachingFileManager.areArgumentsValid(arguments)).isFalse();
  }

  @Test
  public void testAreArgumentsValid_withEmptyInputsAndDigest() throws IOException {
    ImmutableList<Path> bootClassPaths =
        ImmutableList.of(bootClassPathsCandidates.get(0), bootClassPathsCandidates.get(1));

    BlazeJavacArguments arguments =
        BlazeJavacArguments.builder()
            .bootClassPath(bootClassPaths)
            .classOutput(temporaryFolder.newFolder().toPath().resolve("output_1.jar"))
            .build();

    assertThat(BootClassPathCachingFileManager.areArgumentsValid(arguments)).isFalse();
  }

  @Test
  public void testAreArgumentsValid_withEmptyBootClassPaths() throws IOException {
    assertThat(
            BootClassPathCachingFileManager.areArgumentsValid(
                BlazeJavacArguments.builder()
                    .classOutput(temporaryFolder.newFolder().toPath().resolve("output_1.jar"))
                    .build()))
        .isFalse();
  }
}
