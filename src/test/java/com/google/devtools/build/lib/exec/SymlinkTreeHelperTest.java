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

package com.google.devtools.build.lib.exec;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.shell.Command;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link SymlinkTreeHelper}. */
@RunWith(JUnit4.class)
public final class SymlinkTreeHelperTest {
  private final FileSystem fs = new InMemoryFileSystem();

  @Test
  public void checkCreatedSpawn() {
    Path execRoot = fs.getPath("/my/workspace");
    Path inputManifestPath = execRoot.getRelative("input_manifest");
    BinTools binTools =
        BinTools.forUnitTesting(execRoot, ImmutableList.of(SymlinkTreeHelper.BUILD_RUNFILES));
    Command command =
        new SymlinkTreeHelper(inputManifestPath, execRoot.getRelative("output/MANIFEST"), false)
            .createCommand(execRoot, binTools, ImmutableMap.of());
    assertThat(command.getEnvironmentVariables()).isEmpty();
    assertThat(command.getWorkingDirectory()).isEqualTo(execRoot.getPathFile());
    String[] commandLine = command.getCommandLineElements();
    assertThat(commandLine).hasLength(3);
    assertThat(commandLine[0]).endsWith(SymlinkTreeHelper.BUILD_RUNFILES);
    assertThat(commandLine[1]).isEqualTo("input_manifest");
    assertThat(commandLine[2]).isEqualTo("output/MANIFEST");
  }
}
