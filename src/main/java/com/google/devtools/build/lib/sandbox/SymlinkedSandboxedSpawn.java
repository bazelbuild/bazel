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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.exec.TreeDeleter;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxInputs;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxOutputs;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.StashContents;
import com.google.devtools.build.lib.util.CommandDescriptionForm;
import com.google.devtools.build.lib.util.CommandFailureUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * Creates an execRoot for a Spawn that contains input files as symlinks to their original
 * destination.
 */
public class SymlinkedSandboxedSpawn extends AbstractContainerizingSandboxedSpawn {

  /** Mnemonic of the action running in this spawn. */
  private final String mnemonic;

  @Nullable private final ImmutableList<String> interactiveDebugArguments;

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
      @Nullable ImmutableList<String> interactiveDebugArguments,
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
    this.mnemonic = isNullOrEmpty(mnemonic) ? "_NoMnemonic_" : mnemonic;
    this.interactiveDebugArguments = interactiveDebugArguments;
  }

  @Override
  public StashContents filterInputsAndDirsToCreate(
      Set<PathFragment> inputsToCreate, Set<PathFragment> dirsToCreate)
      throws IOException, InterruptedException {
    StashContents oldStashContents = null;
    StashContents stashContents =
        SandboxStash.takeStashedSandbox(sandboxPath, mnemonic, getEnvironment(), outputs);
    sandboxExecRoot.createDirectoryAndParents();
    if (stashContents != null) {
      SandboxStash.statistics.putIfAbsent("reused_stashes", 0);
      SandboxStash.statistics.put("reused_stashes", SandboxStash.statistics.get("reused_stashes") + 1);
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
          sandboxExecRoot,
          treeDeleter,
          stashContents);
      oldStashContents = stashContents;
    } else {
      SandboxStash.statistics.putIfAbsent("stashes_from_scratch", 0);
      SandboxStash.statistics.put("stashes_from_scratch", SandboxStash.statistics.get("stashes_from_scratch") + 1);
    }
    Map<PathFragment, StashContents> stashContentsMap = new HashMap<>();
    for (var entry : Iterables.concat(
        inputs.getFiles().entrySet().stream().map(
            x -> Map.entry(x.getKey(), x.getValue() == null ? PathFragment.EMPTY_FRAGMENT : x.getValue().asPath().asFragment())).collect(ImmutableList.toImmutableList()), inputs.getSymlinks().entrySet()))  {
      PathFragment parent = entry.getKey().getParentDirectory();
      boolean parentWasPresent = true;
      if (!stashContentsMap.containsKey(parent)) {
        stashContentsMap.put(parent, new StashContents());
        parentWasPresent = false;
      }
      StashContents parentStashContents = stashContentsMap.get(parent);
      parentStashContents.symlinkEntryToTarget.put(entry.getKey().getBaseName(), entry.getValue());
      while (!parentWasPresent && parent.getParentDirectory() != null) {
        PathFragment parentParent = parent.getParentDirectory();
        if (stashContentsMap.containsKey(parentParent)) {
          parentWasPresent = true;
        } else {
          stashContentsMap.put(parentParent, new StashContents());
        }
        StashContents parentParentStashContents = stashContentsMap.get(parentParent);
        if (!parentParentStashContents.dirEntries.containsKey(parent.getBaseName())) {
          parentParentStashContents.dirEntries.put(parent.getBaseName(),
              stashContentsMap.get(parent));
        }
        parent = parentParent;
      }
    }
    for (var outputDir : Iterables.concat(outputs.files().values().stream().map(PathFragment::getParentDirectory).collect(
        ImmutableList.toImmutableList()), outputs.dirs().values())) {
      PathFragment parent = outputDir;
      boolean parentWasPresent = true;
      if (!stashContentsMap.containsKey(parent)) {
        stashContentsMap.put(parent, new StashContents());
        parentWasPresent = false;
      }
      while (!parentWasPresent && parent.getParentDirectory() != null) {
        PathFragment parentParent = parent.getParentDirectory();
        if (stashContentsMap.containsKey(parentParent)) {
          parentWasPresent = true;
        } else {
          stashContentsMap.put(parentParent, new StashContents());
        }
        StashContents parentParentStashContents = stashContentsMap.get(parentParent);
        if (!parentParentStashContents.dirEntries.containsKey(parent.getBaseName())) {
          parentParentStashContents.dirEntries.put(parent.getBaseName(),
              stashContentsMap.get(parent));
        }
        parent = parentParent;
      }
    }
    StashContents main = new StashContents();
    // TODO: Avoid having to do this "_main" thing
    main.dirEntries.put("_main", stashContentsMap.get(PathFragment.EMPTY_FRAGMENT));
    SandboxStash.setPathContents(sandboxPath, main);
    // TODO: delete return value, only for debugging
    return oldStashContents;
  }

  @Override
  protected void copyFile(Path source, Path target) throws IOException {
    SandboxStash.statistics.putIfAbsent("symlinks", 0);
    SandboxStash.statistics.put("symlinks", SandboxStash.statistics.get("symlinks") + 1);
    target.createSymbolicLink(source);
  }

  @Override
  public void delete() {
    SandboxStash.stashSandbox(sandboxPath, mnemonic, getEnvironment(), outputs, treeDeleter);
    super.delete();
  }

  @Nullable
  @Override
  public Optional<String> getInteractiveDebugInstructions() {
    if (interactiveDebugArguments == null) {
      return Optional.empty();
    }
    return Optional.of(
        "Run this command to start an interactive shell in an identical sandboxed environment:\n"
            + CommandFailureUtils.describeCommand(
                CommandDescriptionForm.COMPLETE,
                /* prettyPrintArgs= */ false,
                interactiveDebugArguments,
                getEnvironment(),
                /* environmentVariablesToClear= */ null,
                /* cwd= */ null,
                /* configurationChecksum= */ null,
                /* executionPlatformLabel= */ null));
  }
}
