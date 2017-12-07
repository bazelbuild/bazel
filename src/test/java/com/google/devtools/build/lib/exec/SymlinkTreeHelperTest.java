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
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.ExecutionRequirements;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.analysis.config.BinTools;
import com.google.devtools.build.lib.exec.util.FakeOwner;
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
    ActionExecutionMetadata owner = new FakeOwner("SymlinkTree", "Creating it");
    Path execRoot = fs.getPath("/my/workspace");
    Path inputManifestPath = execRoot.getRelative("input_manifest");
    ActionInput inputManifest = ActionInputHelper.fromPath(inputManifestPath.asFragment());
    Spawn spawn =
        new SymlinkTreeHelper(
            inputManifestPath,
            fs.getPath("/my/workspace/output/MANIFEST"),
            false)
        .createSpawn(
            owner,
            execRoot,
            BinTools.forUnitTesting(execRoot, ImmutableList.of(SymlinkTreeHelper.BUILD_RUNFILES)),
            ImmutableMap.of(),
            inputManifest);
    assertThat(spawn.getResourceOwner()).isSameAs(owner);
    assertThat(spawn.getEnvironment()).isEmpty();
    assertThat(spawn.getExecutionInfo()).containsExactly(
        ExecutionRequirements.LOCAL, "",
        ExecutionRequirements.NO_CACHE, "",
        ExecutionRequirements.NO_SANDBOX, "");
    assertThat(spawn.getInputFiles()).containsExactly(inputManifest);
    // At this time, the spawn does not declare any output files.
    assertThat(spawn.getOutputFiles()).isEmpty();
  }
}
