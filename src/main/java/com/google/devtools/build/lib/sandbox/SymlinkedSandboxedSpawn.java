// Copyright 2016 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.sandbox;

import static com.google.common.base.Strings.isNullOrEmpty;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.exec.TreeDeleter;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxInputs;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxOutputs;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.LinkedHashSet;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * Creates an execRoot for a Spawn that contains input files as symlinks to their original
 * destination.
 */
public class SymlinkedSandboxedSpawn extends AbstractContainerizingSandboxedSpawn {

  /** Mnemonic of the action running in this spawn. */
  private final String mnemonic;

  public SymlinkedSandboxedSpawn(
      Path sandboxPath,
      Path sandboxExecRoot,
      ImmutableList<String> arguments,
      ImmutableMap<String, String> environment,
      SandboxInputs inputs,
      SandboxOutputs outputs,
      Set<Path> writableDirs,
      TreeDeleter treeDeleter,
      @Nullable Path sandboxDebugPath,
      @Nullable Path statisticsPath,
      String mnemonic) {
    super(
        sandboxPath,
        sandboxExecRoot,
        arguments,
        environment,
        inputs,
        outputs,
        writableDirs,
        treeDeleter,
        sandboxDebugPath,
        statisticsPath,
        mnemonic);
    this.mnemonic = isNullOrEmpty(mnemonic) ? mnemonic : "_NoMnemonic_";
  }

  @Override
  public void filterInputsAndDirsToCreate(
      Set<PathFragment> inputsToCreate, LinkedHashSet<PathFragment> dirsToCreate)
      throws IOException, InterruptedException {
    boolean gotStash = SandboxStash.takeStashedSandbox(sandboxPath, mnemonic);
    sandboxExecRoot.createDirectoryAndParents();
    if (gotStash) {
      // When reusing an old sandbox, we do a full traversal of the parent directory of
      // `sandboxExecRoot`. This will use what we computed above, delete anything unnecessary, and
      // update `inputsToCreate`/`dirsToCreate` if something can be left without changes (e.g., a,
      // symlink that already points to the right destination). We're traversing from
      // sandboxExecRoot's parent directory because external repositories can now be symlinked as
      // siblings of sandboxExecRoot when --experimental_sibling_repository_layout is set.
      SandboxHelpers.cleanExisting(
          sandboxExecRoot.getParentDirectory(),
          inputs,
          inputsToCreate,
          dirsToCreate,
          sandboxExecRoot);
    }
  }

  @Override
  protected void copyFile(Path source, Path target) throws IOException {
    target.createSymbolicLink(source);
  }

  @Override
  public void delete() {
    SandboxStash.stashSandbox(sandboxPath, mnemonic);
    super.delete();
  }
}
