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
import static org.mockito.ArgumentMatchers.anyBoolean;
import static org.mockito.Mockito.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.exec.SpawnRunner.SpawnExecutionContext;
import com.google.devtools.build.lib.remote.common.RemotePathResolver;
import com.google.devtools.build.lib.remote.common.RemotePathResolver.SiblingRepositoryLayoutResolver;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.util.SortedMap;
import java.util.TreeMap;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link RemotePathResolver} */
@RunWith(JUnit4.class)
public class RemotePathResolverTest {

  private Path execRoot;
  private SpawnExecutionContext spawnExecutionContext;
  private ActionInput input;

  @Before
  public void setup() throws Exception {
    FileSystem fs = new InMemoryFileSystem(DigestHashFunction.SHA256);
    execRoot = fs.getPath("/execroot/main");

    input = ActionInputHelper.fromPath("foo");
    spawnExecutionContext = mock(SpawnExecutionContext.class);
    when(spawnExecutionContext.getInputMapping(any(), anyBoolean(), anyBoolean()))
        .thenAnswer(
            invocationOnMock -> {
              PathFragment baseDirectory = invocationOnMock.getArgument(0);
              TreeMap<PathFragment, ActionInput> inputMap = new TreeMap<>();
              inputMap.put(baseDirectory.getRelative(input.getExecPath()), input);
              return inputMap;
            });
  }

  @Test
  public void getWorkingDirectory_default_isInputRoot() {
    RemotePathResolver remotePathResolver = RemotePathResolver.createDefault(execRoot);

    String workingDirectory = remotePathResolver.getWorkingDirectory();

    assertThat(workingDirectory).isEqualTo("");
  }

  @Test
  public void getWorkingDirectory_sibling_isExecRootBaseName() {
    RemotePathResolver remotePathResolver = new SiblingRepositoryLayoutResolver(execRoot);

    String workingDirectory = remotePathResolver.getWorkingDirectory();

    assertThat(workingDirectory).isEqualTo("main");
  }

  @Test
  public void getInputMapping_default_inputsRelativeToExecRoot() throws Exception {
    RemotePathResolver remotePathResolver = RemotePathResolver.createDefault(execRoot);

    SortedMap<PathFragment, ActionInput> inputs =
        remotePathResolver.getInputMapping(spawnExecutionContext, false);

    assertThat(inputs).containsExactly(PathFragment.create("foo"), input);
  }

  @Test
  public void getInputMapping_sibling_inputsRelativeToInputRoot() throws Exception {
    RemotePathResolver remotePathResolver = new SiblingRepositoryLayoutResolver(execRoot);

    SortedMap<PathFragment, ActionInput> inputs =
        remotePathResolver.getInputMapping(spawnExecutionContext, false);

    assertThat(inputs).containsExactly(PathFragment.create("main/foo"), input);
  }

  @Test
  public void convertPaths_default_relativeToWorkingDirectory() {
    RemotePathResolver remotePathResolver = RemotePathResolver.createDefault(execRoot);

    String outputPath = remotePathResolver.localPathToOutputPath(PathFragment.create("bar"));
    Path localPath = remotePathResolver.outputPathToLocalPath(outputPath);

    assertThat(outputPath).isEqualTo("bar");
    assertThat(localPath).isEqualTo(execRoot.getRelative("bar"));
  }

  @Test
  public void convertPaths_siblingCompatible_relativeToWorkingDirectory() {
    RemotePathResolver remotePathResolver = new SiblingRepositoryLayoutResolver(execRoot);

    String outputPath = remotePathResolver.localPathToOutputPath(PathFragment.create("bar"));
    Path localPath = remotePathResolver.outputPathToLocalPath(outputPath);

    assertThat(outputPath).isEqualTo("bar");
    assertThat(localPath).isEqualTo(execRoot.getRelative("bar"));
  }
}
