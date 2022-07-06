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
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.exec.TreeDeleter;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxInputs;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxOutputs;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import javax.annotation.Nullable;

/**
 * Creates an execRoot for a Spawn that contains input files as symlinks to their original
 * destination.
 */
public class SymlinkedSandboxedSpawn extends AbstractContainerizingSandboxedSpawn {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  /** If true, we have already warned about an error causing us to turn off reuse. */
  private static final AtomicBoolean warnedAboutTurningOffReuse = new AtomicBoolean();

  /** Base for the entire sandbox system, needed for stashing reusable sandboxes. */
  private final Path sandboxBase;

  /**
   * Whether to attempt to reuse previously-created sandboxes. Not final because we may turn it off
   * in case of errors.
   */
  private boolean reuseSandboxDirectories;

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
      @Nullable Path statisticsPath,
      boolean reuseSandboxDirectories,
      Path sandboxBase,
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
        statisticsPath);
    this.sandboxBase = sandboxBase;
    this.reuseSandboxDirectories = reuseSandboxDirectories;
    this.mnemonic = isNullOrEmpty(mnemonic) ? mnemonic : "_NoMnemonic_";
  }

  @Override
  public void filterInputsAndDirsToCreate(
      Set<PathFragment> inputsToCreate, LinkedHashSet<PathFragment> dirsToCreate)
      throws IOException {
    if (reuseSandboxDirectories && takeStashedSandbox()) {
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

  /**
   * Attempts to take an existing stashed sandbox for reuse. Returns true if it succeeds. On certain
   * errors we disable sandbox reuse because it seems to just not work.
   */
  private boolean takeStashedSandbox() {
    Path sandboxes = getSandboxStashDir();
    if (sandboxes == null) {
      return false;
    }
    try {
      Collection<Path> stashes = sandboxes.getDirectoryEntries();
      // We have to remove the sandbox root to move a stash there, but it is currently empty
      // and we reinstate it if we don't get a sandbox.
      sandboxPath.deleteTree();
      for (Path stash : stashes) {
        try {
          stash.renameTo(sandboxPath);
          return true;
        } catch (FileNotFoundException e) {
          // Try the next one, somebody else took this one.
        } catch (IOException e) {
          turnOffReuse("Error renaming sandbox stash %s to %s: %s\n", stash, sandboxPath, e);
          return false;
        }
      }
    } catch (IOException e) {
      turnOffReuse("Failed to prepare for reusing stashed sandbox for %s: %s", sandboxPath, e);
      return false;
    } finally {
      if (!sandboxPath.exists()) {
        try {
          // If we failed somehow, recreate the empty sandbox.
          sandboxExecRoot.createDirectoryAndParents();
        } catch (IOException e) {
          System.err.printf("Failed to re-establish sandbox %s: %s\n", sandboxPath, e);
        }
      }
    }
    return false;
  }

  /** An incrementing count of stashes to avoid filename clashes. */
  static final AtomicInteger stash = new AtomicInteger(0);

  /** Atomically moves the sandboxPath directory aside for later reuse. */
  private boolean stashSandbox(Path path) {
    Path sandboxes = getSandboxStashDir();
    if (sandboxes == null) {
      return false;
    }
    String stashName;
    synchronized (stash) {
      stashName = Integer.toString(stash.incrementAndGet());
    }
    Path stashPath = sandboxes.getChild(stashName);
    if (!path.exists()) {
      return false;
    }
    try {
      path.renameTo(stashPath);
    } catch (IOException e) {
      // Since stash names are unique, this IOException indicates some other problem with stashing,
      // so we turn it off.
      turnOffReuse("Error stashing sandbox at %s: %s", stashPath, e);
      return false;
    }
    return true;
  }

  /**
   * Returns the sandbox stashing directory appropriate for this spawn. In order to maximize reuse,
   * we keep stashed sandboxes separated by mnemonic. May return null if there are errors, in which
   * case sandbox reuse also gets turned of.
   */
  @Nullable
  private Path getSandboxStashDir() {
    Path stashDir = sandboxBase.getChild("sandbox_stash");
    try {
      stashDir.createDirectory();
      if (!maybeClearExistingStash(stashDir)) {
        return null;
      }
    } catch (IOException e) {
      turnOffReuse(
          "Error creating sandbox stash dir %s, disabling sandbox reuse: %s\n",
          stashDir, e.getMessage());
      return null;
    }
    Path mnemonicStashDir = stashDir.getChild(mnemonic);
    try {
      mnemonicStashDir.createDirectory();
      return mnemonicStashDir;
    } catch (IOException e) {
      turnOffReuse("Error creating mnemonic stash dir %s: %s\n", mnemonicStashDir, e.getMessage());
      return null;
    }
  }

  /**
   * Clears away existing stash if this is the first access to the stash in this Blaze server
   * instance.
   *
   * @param stashPath Path of the stashes.
   * @return True unless there was an error deleting sandbox stashes.
   */
  private boolean maybeClearExistingStash(Path stashPath) {
    synchronized (stash) {
      if (stash.getAndIncrement() == 0) {
        try {
          for (Path directoryEntry : stashPath.getDirectoryEntries()) {
            directoryEntry.deleteTree();
          }
        } catch (IOException e) {
          turnOffReuse("Unable to clear old sandbox stash %s: %s\n", stashPath, e.getMessage());
          return false;
        }
      }
    }
    return true;
  }

  @Override
  protected void copyFile(Path source, Path target) throws IOException {
    target.createSymbolicLink(source);
  }

  @Override
  public void delete() {
    if (!reuseSandboxDirectories || !stashSandbox(sandboxPath)) {
      super.delete();
    }
  }

  private void turnOffReuse(String fmt, Object... args) {
    reuseSandboxDirectories = false;
    if (warnedAboutTurningOffReuse.compareAndSet(false, true)) {
      logger.atWarning().logVarargs("Turning off sandbox reuse: " + fmt, args);
    }
  }
}
