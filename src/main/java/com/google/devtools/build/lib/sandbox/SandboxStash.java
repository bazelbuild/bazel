// Copyright 2023 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Preconditions;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.exec.TreeDeleter;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxOutputs;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Collection;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import javax.annotation.Nullable;

/**
 * Singleton class for the `--reuse_sandbox_directories` flag: Controls a "stash" of old sandbox
 * directories. When a sandboxed runner needs its directory tree, it first tries to grab a stash by
 * just moving it. They are separated by mnemonic because that makes them much more likely to be
 * able to reuse things common for that mnemonic, e.g. standard libraries.
 */
public class SandboxStash {

  public static final String SANDBOX_STASH_BASE = "sandbox_stash";
  private static final String TEST_RUNNER_MNEMONIC = "TestRunner";
  private static final String TEST_SRCDIR = "TEST_SRCDIR";
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  /** An incrementing count of stashes to avoid filename clashes. */
  static final AtomicInteger stash = new AtomicInteger(0);

  /** If true, we have already warned about an error causing us to turn off reuse. */
  private final AtomicBoolean warnedAboutTurningOffReuse = new AtomicBoolean();

  /**
   * Whether to attempt to reuse previously-created sandboxes. Not final because we may turn it off
   * in case of errors.
   */
  static boolean reuseSandboxDirectories;

  private static SandboxStash instance;
  private final String workspaceName;
  private final Path sandboxBase;

  private final Map<Path, String> stashPathToRunfilesDir = new ConcurrentHashMap<>();

  public SandboxStash(String workspaceName, Path sandboxBase) {
    this.workspaceName = workspaceName;
    this.sandboxBase = sandboxBase;
  }

  static boolean takeStashedSandbox(
      Path sandboxPath, String mnemonic, Map<String, String> environment, SandboxOutputs outputs) {
    if (instance == null) {
      return false;
    }
    return instance.takeStashedSandboxInternal(sandboxPath, mnemonic, environment, outputs);
  }

  private boolean takeStashedSandboxInternal(
      Path sandboxPath, String mnemonic, Map<String, String> environment, SandboxOutputs outputs) {
    try {
      Path sandboxes = getSandboxStashDir(mnemonic, sandboxPath.getFileSystem());
      if (sandboxes == null || isTestXmlGenerationOrCoverageSpawn(mnemonic, outputs)) {
        return false;
      }
      Collection<Path> stashes = sandboxes.getDirectoryEntries();
      if (stashes.isEmpty()) {
        return false;
      }
      // We have to remove the sandbox execroot dir to move a stash there, but it is currently empty
      // and we reinstate it later if we don't get a sandbox. We can't just move the stash dir
      // fully, as we would then lose siblings of the execroot dir, such as hermetic-tmp dirs.
      Path sandboxExecroot = sandboxPath.getChild("execroot");
      sandboxExecroot.deleteTree();
      for (Path stash : stashes) {
        try {
          Path stashExecroot = stash.getChild("execroot");
          stashExecroot.renameTo(sandboxExecroot);
          stash.deleteTree();
          if (isTestAction(mnemonic)) {
            String relativeStashedRunfilesDir = stashPathToRunfilesDir.get(stashExecroot);
            if (relativeStashedRunfilesDir == null) {
              StringBuilder errorMessageBuilder = new StringBuilder();
              errorMessageBuilder.append(
                  String.format("Current stashExecroot:%s\n", stashExecroot));
              errorMessageBuilder.append("Stashes:\n");
              for (Path path : stashes) {
                errorMessageBuilder.append(path.getPathString()).append("\n");
              }
              errorMessageBuilder.append("Environment:\n");
              for (var entry : environment.entrySet()) {
                errorMessageBuilder.append(
                    String.format("%s : %s\n", entry.getKey(), entry.getValue()));
              }
              errorMessageBuilder.append("Entries runfiles map:\n");
              for (var entry : stashPathToRunfilesDir.entrySet()) {
                errorMessageBuilder.append(
                    String.format("%s : %s\n", entry.getKey(), entry.getValue()));
              }
              Preconditions.checkNotNull(
                  relativeStashedRunfilesDir, errorMessageBuilder.toString());
            }
            Path stashedRunfilesDir = sandboxExecroot.getRelative(relativeStashedRunfilesDir);
            Path currentRunfiles = sandboxExecroot.getRelative(getCurrentRunfilesDir(environment));
            currentRunfiles.getParentDirectory().createDirectoryAndParents();
            stashedRunfilesDir.renameTo(currentRunfiles);
            stashPathToRunfilesDir.remove(stashExecroot);
          }
          return true;
        } catch (FileNotFoundException e) {
          // Try the next one, somebody else took this one.
        } catch (IOException e) {
          turnOffReuse("Error renaming sandbox stash %s to %s: %s\n", stash, sandboxPath, e);
          return false;
        }
      }
      return false;
    } catch (IOException e) {
      turnOffReuse("Failed to prepare for reusing stashed sandbox for %s: %s", sandboxPath, e);
      return false;
    }
  }

  /** Atomically moves the sandboxPath directory aside for later reuse. */
  static void stashSandbox(
      Path path,
      String mnemonic,
      Map<String, String> environment,
      SandboxOutputs outputs,
      TreeDeleter treeDeleter) {
    if (instance == null) {
      return;
    }
    instance.stashSandboxInternal(path, mnemonic, environment, outputs, treeDeleter);
  }

  private void stashSandboxInternal(
      Path path,
      String mnemonic,
      Map<String, String> environment,
      SandboxOutputs outputs,
      TreeDeleter treeDeleter) {
    Path sandboxes = getSandboxStashDir(mnemonic, path.getFileSystem());
    if (sandboxes == null || isTestXmlGenerationOrCoverageSpawn(mnemonic, outputs)) {
      return;
    }
    String stashName = Integer.toString(stash.incrementAndGet());
    Path stashPath = sandboxes.getChild(stashName);
    if (!path.exists()) {
      return;
    }
    try {
      stashPath.createDirectory();
      Path stashPathExecroot = stashPath.getChild("execroot");
      if (isTestAction(mnemonic)) {
        if (environment.get("TEST_TMPDIR").startsWith("_tmp")) {
          treeDeleter.deleteTree(
              path.getRelative("execroot/" + environment.get("TEST_WORKSPACE") + "/_tmp"));
        }
      }
      if (isTestAction(mnemonic)) {
        // We do this before the rename operation to avoid a race condition.
        stashPathToRunfilesDir.put(stashPathExecroot, getCurrentRunfilesDir(environment));
      }
      path.getChild("execroot").renameTo(stashPathExecroot);
    } catch (IOException e) {
      // Since stash names are unique, this IOException indicates some other problem with stashing,
      // so we turn it off.
      turnOffReuse("Error stashing sandbox at %s: %s", stashPath, e);
    }
  }

  /**
   * Returns the sandbox stashing directory appropriate for this mnemonic. In order to maximize
   * reuse, we keep stashed sandboxes separated by mnemonic. May return null if there are errors, in
   * which case sandbox reuse also gets turned off.
   *
   * <p>TODO(bazel-team): Fix integration tests to instantiate FileSystem only once, so that passing
   * it in here (to avoid the cross-filesystem precondition check in renameTo) is no longer
   * necessary.
   */
  @Nullable
  private Path getSandboxStashDir(String mnemonic, FileSystem fileSystem) {
    Path stashDir = getStashBase(fileSystem.getPath(this.sandboxBase.getPathString()));
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

  private static Path getStashBase(Path sandboxBase) {
    return sandboxBase.getChild(SANDBOX_STASH_BASE);
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

  private void turnOffReuse(String fmt, Object... args) {
    reuseSandboxDirectories = false;
    if (warnedAboutTurningOffReuse.compareAndSet(false, true)) {
      logger.atWarning().logVarargs("Turning off sandbox reuse: " + fmt, args);
    }
  }

  public static void initialize(String workspaceName, Path sandboxBase, SandboxOptions options) {
    if (options.reuseSandboxDirectories) {
      if (instance == null) {
        instance = new SandboxStash(workspaceName, sandboxBase);
      } else if (!Objects.equals(workspaceName, instance.workspaceName)) {
        Path stashBase = getStashBase(instance.sandboxBase);
        try {
          for (Path directoryEntry : stashBase.getDirectoryEntries()) {
            directoryEntry.deleteTree();
          }
        } catch (IOException e) {
          instance.turnOffReuse(
              "Unable to clear old sandbox stash %s: %s\n", stashBase, e.getMessage());
        }
        instance = new SandboxStash(workspaceName, sandboxBase);
      }
    } else {
      instance = null;
    }
  }

  /** Cleans up the entire current stash, if any. Cleaning may be asynchronous. */
  static void clean(TreeDeleter treeDeleter, Path sandboxBase) {
    Path stashDir = getStashBase(sandboxBase);
    if (!stashDir.isDirectory()) {
      return;
    }
    Path stashTrashDir = stashDir.getChild("__trash");
    try {
      stashDir.renameTo(stashTrashDir);
    } catch (IOException e) {
      // If we couldn't move the stashdir away for deletion, we need to delete it synchronously
      // in place, so we can't use the treeDeleter.
      treeDeleter = null;
      stashTrashDir = stashDir;
    }
    try {
      if (treeDeleter != null) {
        treeDeleter.deleteTree(stashTrashDir);
      } else {
        stashTrashDir.deleteTree();
      }
    } catch (IOException e) {
      logger.atWarning().withCause(e).log("Failed to clean sandbox stash %s", stashDir);
    }
  }

  /**
   * Test actions are guaranteed to have a runfiles directory with the test name as part of the
   * name. The path to the directory is unique between tests. If two tests (foo and bar) have the
   * directory <source-root>/pkg/my_runfiles as part of their runfiles and this directory contains
   * 1000 files, we would be symlinking the 1000 files for each test since the paths do not
   * coincide. To make sure we can reuse the runfiles directory we must rename the old runfiles
   * directory for the action that was stashed to the path that is expected by the current test.
   */
  private static boolean isTestAction(String mnemonic) {
    return mnemonic.equals(TEST_RUNNER_MNEMONIC);
  }

  /**
   * Test actions are split in two spawns. The first one runs the test and the second generates the
   * XML output from the test log. We do not want the second spawn to reuse the stash because it
   * doesn't contain the inputs needed to run the test; if it reused it, it would be expensive in
   * two ways: it would have to clean up all the inputs, and it would destroy a valid stash that a
   * different test could potentially use. If we are running coverage, there might be a third spawn
   * for coverage where we apply the same reasoning.
   *
   * <p>We identify the second and third spawn because they have a single output.
   */
  private static boolean isTestXmlGenerationOrCoverageSpawn(
      String mnemonic, SandboxOutputs outputs) {
    return isTestAction(mnemonic) && outputs.files().size() == 1;
  }

  private static String getCurrentRunfilesDir(Map<String, String> environment) {
    return environment.get("TEST_WORKSPACE") + "/" + environment.get(TEST_SRCDIR);
  }
}
