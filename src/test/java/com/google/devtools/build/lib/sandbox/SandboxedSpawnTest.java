// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.sandbox;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import java.io.IOException;
import java.util.Set;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link SandboxedSpawn}. */
@RunWith(JUnit4.class)
public class SandboxedSpawnTest extends SandboxTestCase {

  @Test
  public void testMoveOutputs() throws Exception {
    Path execRoot = testRoot.getRelative("execroot");
    execRoot.createDirectory();

    Path outputFile = execRoot.getRelative("very/output.txt");
    Path outputLink = execRoot.getRelative("very/output.link");
    Path outputDangling = execRoot.getRelative("very/output.dangling");
    Path outputDir = execRoot.getRelative("very/output.dir");
    Path outputInUncreatedTargetDir = execRoot.getRelative("uncreated/output.txt");

    Set<PathFragment> outputs = ImmutableSet.of(
        outputFile.relativeTo(execRoot),
        outputLink.relativeTo(execRoot),
        outputDangling.relativeTo(execRoot),
        outputDir.relativeTo(execRoot),
        outputInUncreatedTargetDir.relativeTo(execRoot));
    for (PathFragment path : outputs) {
      execRoot.getRelative(path).getParentDirectory().createDirectoryAndParents();
    }

    FileSystemUtils.createEmptyFile(outputFile);
    outputLink.createSymbolicLink(PathFragment.create("output.txt"));
    outputDangling.createSymbolicLink(PathFragment.create("doesnotexist"));
    outputDir.createDirectory();
    FileSystemUtils.createEmptyFile(outputDir.getRelative("test.txt"));
    FileSystemUtils.createEmptyFile(outputInUncreatedTargetDir);

    Path outputsDir = testRoot.getRelative("outputs");
    outputsDir.createDirectory();
    outputsDir.getRelative("very").createDirectory();
    SandboxedSpawn.moveOutputs(outputs, execRoot, outputsDir);

    assertThat(outputsDir.getRelative("very/output.txt").isFile(Symlinks.NOFOLLOW)).isTrue();
    assertThat(outputsDir.getRelative("very/output.link").isSymbolicLink()).isTrue();
    assertThat(outputsDir.getRelative("very/output.link").resolveSymbolicLinks())
        .isEqualTo(outputsDir.getRelative("very/output.txt"));
    assertThat(outputsDir.getRelative("very/output.dangling").isSymbolicLink()).isTrue();
    try {
      outputsDir.getRelative("very/output.dangling").resolveSymbolicLinks();
      fail("expected IOException");
    } catch (IOException e) {
      // Ignored.
    }
    assertThat(outputsDir.getRelative("very/output.dir").isDirectory(Symlinks.NOFOLLOW)).isTrue();
    assertThat(outputsDir.getRelative("very/output.dir/test.txt").isFile(Symlinks.NOFOLLOW))
        .isTrue();
    assertThat(outputsDir.getRelative("uncreated/output.txt").isFile(Symlinks.NOFOLLOW)).isTrue();
  }
}
