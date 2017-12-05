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
package com.google.devtools.build.lib.remote;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.remoteexecution.v1test.Action;
import com.google.devtools.remoteexecution.v1test.Digest;
import com.google.devtools.remoteexecution.v1test.Platform;
import java.time.Duration;
import java.time.temporal.ChronoUnit;
import java.util.Arrays;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.mockito.MockitoAnnotations;

public class BuildActionTest {
  private FileSystem fs;
  private Path execRoot;

  @Before
  public final void setUp() throws Exception {
    MockitoAnnotations.initMocks(this);
    fs = new InMemoryFileSystem();
    execRoot = fs.getPath("/exec/root");
    FileSystemUtils.createDirectoryAndParents(execRoot);
  }

  @Test
  public void actionIsBuiltAccordingToOutputType() throws Exception {
    FileSystemUtils.createEmptyFile(execRoot.getRelative("qux"));
    FileSystemUtils.createDirectoryAndParents(execRoot.getRelative("bar"));

    List<ActionInput> outputs = Arrays.asList(
        ActionInputHelper.fromPath("qux"),
        ActionInputHelper.fromPath("bar"));

    Action action = RemoteSpawnRunner.buildAction(
        execRoot,
        outputs,
        Digest.newBuilder().setHash("commandDigest").build(), // command
        Digest.newBuilder().setHash("inputRootDigest").build(), // input root
        Platform.newBuilder().build(),
        Duration.of(1, ChronoUnit.SECONDS) // timeout
    );

    assertThat(action.getCommandDigest().getHash()).isEqualTo("commandDigest");
    assertThat(action.getInputRootDigest().getHash()).isEqualTo("inputRootDigest");
    assertThat(action.getTimeout()).isEqualTo(Duration.of(1, ChronoUnit.SECONDS));

    assertThat(action.getOutputFilesList()).containsExactly("qux");
    assertThat(action.getOutputDirectoriesList()).containsExactly("bar");
  }
}
