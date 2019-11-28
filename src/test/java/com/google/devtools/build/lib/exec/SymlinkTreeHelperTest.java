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
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.shell.Command;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Map;
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

  @Test
  public void readManifest() throws Exception {
    Path execRoot = fs.getPath("/my/workspace");
    execRoot.createDirectoryAndParents();
    Path inputManifestPath = execRoot.getRelative("input_manifest");
    FileSystemUtils.writeContentAsLatin1(inputManifestPath, "from to\nmetadata");
    Map<PathFragment, PathFragment> symlinks =
        SymlinkTreeHelper.readSymlinksFromFilesetManifest(inputManifestPath);
    assertThat(symlinks).containsExactly(PathFragment.create("from"), PathFragment.create("to"));
  }

  @Test
  public void readMultilineManifest() throws Exception {
    Path execRoot = fs.getPath("/my/workspace");
    execRoot.createDirectoryAndParents();
    Path inputManifestPath = execRoot.getRelative("input_manifest");
    FileSystemUtils.writeContentAsLatin1(
        inputManifestPath, "from to\nmetadata\n/foo /bar\nmetadata");
    Map<PathFragment, PathFragment> symlinks =
        SymlinkTreeHelper.readSymlinksFromFilesetManifest(inputManifestPath);
    assertThat(symlinks)
        .containsExactly(
            PathFragment.create("from"),
            PathFragment.create("to"),
            PathFragment.create("/foo"),
            PathFragment.create("/bar"));
  }

  @Test
  public void readCorruptManifest() throws Exception {
    Path execRoot = fs.getPath("/my/workspace");
    execRoot.createDirectoryAndParents();
    Path inputManifestPath = execRoot.getRelative("input_manifest");
    FileSystemUtils.writeContentAsLatin1(inputManifestPath, "from to");
    assertThrows(
        IOException.class,
        () -> SymlinkTreeHelper.readSymlinksFromFilesetManifest(inputManifestPath));
  }

  @Test
  public void readNonExistentManifestFails() {
    Path execRoot = fs.getPath("/my/workspace");
    Path inputManifestPath = execRoot.getRelative("input_manifest");
    assertThrows(
        FileNotFoundException.class,
        () -> SymlinkTreeHelper.readSymlinksFromFilesetManifest(inputManifestPath));
  }
}
