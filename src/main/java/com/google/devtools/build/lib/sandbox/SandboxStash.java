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

import static com.google.devtools.build.lib.vfs.Dirent.Type.DIRECTORY;
import static com.google.devtools.build.lib.vfs.Dirent.Type.SYMLINK;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.flogger.GoogleLogger;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.exec.TreeDeleter;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxOutputs;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.StashContents;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
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

  // Used while we gather all the contents asynchronously.
  public static final String TEMPORARY_SANDBOX_STASH_BASE = "tmp_sandbox_stash";
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

  private static final int POOL_SIZE = Runtime.getRuntime().availableProcessors();
  private final ExecutorService stashFileListingPool =
      Executors.newFixedThreadPool(
          POOL_SIZE,
          new ThreadFactoryBuilder().setNameFormat("stash-file-listing-thread-%d").build());

  public final Map<Path, StashContents> pathToContents = new ConcurrentHashMap<>();
  private final Map<Path, Label> sandboxToTarget = new ConcurrentHashMap<>();
  private final Map<Path, Long> pathToLastModified = new ConcurrentHashMap<>();
  private boolean inMemoryStashes;

  public SandboxStash(String workspaceName, Path sandboxBase, boolean inMemoryStashes) {
    this.workspaceName = workspaceName;
    this.sandboxBase = sandboxBase;
    this.inMemoryStashes = inMemoryStashes;
  }

  @Nullable
  @SuppressWarnings("NullableOptional")
  static Optional<StashContents> takeStashedSandbox(
      Path sandboxPath,
      String mnemonic,
      Map<String, String> environment,
      SandboxOutputs outputs,
      Label target) {
    if (instance == null) {
      return null;
    }
    return instance.takeStashedSandboxInternal(sandboxPath, mnemonic, environment, outputs, target);
  }

  @Nullable
  @SuppressWarnings("NullableOptional")
  private Optional<StashContents> takeStashedSandboxInternal(
      Path sandboxPath,
      String mnemonic,
      Map<String, String> environment,
      SandboxOutputs outputs,
      Label target) {
    try {
      Path sandboxes = getSandboxStashDir(mnemonic, sandboxPath.getFileSystem());
      if (sandboxes == null || isTestXmlGenerationOrCoverageSpawn(mnemonic, outputs)) {
        return null;
      }

      Collection<Path> diskStashes = sandboxes.getDirectoryEntries();
      if (diskStashes.isEmpty()) {
        return null;
      }

      ImmutableList<Path> stashes = sortStashesByMatchingTargetSegments(target, diskStashes);
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
            Path stashedRunfilesDir = sandboxExecroot.getRelative(relativeStashedRunfilesDir);
            String relativeCurrentRunfilesDir = getCurrentRunfilesDir(environment);
            Path currentRunfiles = sandboxExecroot.getRelative(relativeCurrentRunfilesDir);
            currentRunfiles.getParentDirectory().createDirectoryAndParents();
            stashedRunfilesDir.renameTo(currentRunfiles);
            stashPathToRunfilesDir.remove(stashExecroot);
            if (useInMemoryStashes() && pathToContents.containsKey(stash)) {
              updateStashContentsAfterRunfilesMove(
                  relativeStashedRunfilesDir,
                  relativeCurrentRunfilesDir,
                  pathToContents.get(stash));
            }
          }
          sandboxToTarget.remove(stash);
          // If we switched the flag experimental_inmemory_sandbox_stashes from false to true
          // without restarting the Bazel server, we may have a stash but not its contents in
          // memory.
          return useInMemoryStashes() && pathToContents.containsKey(stash)
              ? Optional.of(pathToContents.remove(stash))
              : Optional.empty();
        } catch (FileNotFoundException e) {
          // Try the next one, somebody else took this one.
        } catch (IOException e) {
          turnOffReuse("Error renaming sandbox stash %s to %s: %s\n", stash, sandboxPath, e);
          return null;
        }
      }
      return null;
    } catch (IOException e) {
      turnOffReuse("Failed to prepare for reusing stashed sandbox for %s: %s", sandboxPath, e);
      return null;
    }
  }

  /** Atomically moves the sandboxPath directory aside for later reuse. */
  static void stashSandbox(
      Path path,
      String mnemonic,
      Map<String, String> environment,
      SandboxOutputs outputs,
      TreeDeleter treeDeleter,
      Label target) {
    if (instance == null) {
      return;
    }

    Path sandboxes = instance.getSandboxStashDir(mnemonic, path.getFileSystem());
    if (sandboxes == null
        || isTestXmlGenerationOrCoverageSpawn(mnemonic, outputs)
        || !path.exists()) {
      return;
    }
    String stashName = Integer.toString(stash.incrementAndGet());

    if (useInMemoryStashes()) {
      instance.stashSandboxInternalWithInMemoryStashes(
          stashName, sandboxes, path, mnemonic, environment, treeDeleter, target);
    } else {
      instance.stashSandboxInternal(
          stashName, sandboxes, path, mnemonic, environment, treeDeleter, target);
    }
  }

  @SuppressWarnings("FutureReturnValueIgnored")
  private void stashSandboxInternalWithInMemoryStashes(
      String stashName,
      Path sandboxes,
      Path path,
      String mnemonic,
      Map<String, String> environment,
      TreeDeleter treeDeleter,
      Label target) {
    Path temporaryStashes = sandboxBase.getChild(TEMPORARY_SANDBOX_STASH_BASE);
    Path temporaryStash = temporaryStashes.getChild(stashName);
    try {
      temporaryStashes.createDirectory();
      path.getChild("execroot").renameTo(temporaryStash);
    } catch (IOException e) {
      turnOffReuse("Error stashing sandbox at %s: %s", temporaryStash, e);
    }
    stashFileListingPool.submit(
        () -> {
          Path stashPath = sandboxes.getChild(stashName);
          try {
            StashContents stashContents = pathToContents.remove(path);
            Long lastModified = pathToLastModified.remove(path);
            Preconditions.checkNotNull(lastModified);
            listContentsRecursively(temporaryStash, lastModified, stashContents);
            stashPath.createDirectory();
            Path stashPathExecroot = stashPath.getChild("execroot");
            if (isTestAction(mnemonic)) {
              if (environment.get("TEST_TMPDIR").startsWith("_tmp")) {
                treeDeleter.deleteTree(
                    temporaryStash.getRelative(environment.get("TEST_WORKSPACE") + "/_tmp"));
              }
              // We do this before the rename operation to avoid a race condition.
              stashPathToRunfilesDir.put(stashPathExecroot, getCurrentRunfilesDir(environment));
            }
            setPathContents(stashPath, stashContents);
            temporaryStash.renameTo(stashPathExecroot);
            if (target != null) {
              sandboxToTarget.put(stashPath, target);
            }
          } catch (InterruptedException e) {
            // Finish the job without stashing the sandbox
          } catch (IOException e) {
            // TODO(bazel-team): Are we sure we don't want to surface this error?
            turnOffReuse("Error stashing sandbox at %s: %s", stashPath, e);
          }
        });
  }

  private void stashSandboxInternal(
      String stashName,
      Path sandboxes,
      Path path,
      String mnemonic,
      Map<String, String> environment,
      TreeDeleter treeDeleter,
      Label target) {
    Path stashPath = sandboxes.getChild(stashName);
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
      if (target != null) {
        sandboxToTarget.put(stashPath, target);
      }
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

  public static void initialize(
      String workspaceName, Path sandboxBase, SandboxOptions options, TreeDeleter treeDeleter) {
    if (options.reuseSandboxDirectories) {
      if (instance == null) {
        instance =
            new SandboxStash(
                workspaceName, sandboxBase, options.experimentalInMemorySandboxStashes);
      } else {
        if (!Objects.equals(workspaceName, instance.workspaceName)) {
          Path stashBase = getStashBase(instance.sandboxBase);
          try {
            for (Path directoryEntry : stashBase.getDirectoryEntries()) {
              treeDeleter.deleteTree(directoryEntry);
            }
          } catch (IOException e) {
            instance.turnOffReuse(
                "Unable to clear old sandbox stash %s: %s\n", stashBase, e.getMessage());
          }
          instance =
              new SandboxStash(
                  workspaceName, sandboxBase, options.experimentalInMemorySandboxStashes);
        }
        instance.inMemoryStashes = options.experimentalInMemorySandboxStashes;
      }
    } else {
      instance = null;
    }
  }

  public static boolean useInMemoryStashes() {
    Preconditions.checkNotNull(instance);
    return instance.inMemoryStashes;
  }

  public static void setPathContents(Path path, StashContents stashContents) {
    Preconditions.checkNotNull(instance);
    instance.pathToContents.put(path, stashContents);
  }

  public static void setLastModified(Path path, Long lastModified) {
    if (instance != null) {
      instance.pathToLastModified.put(path, lastModified);
    }
  }

  public static String getWorkspaceName() {
    Preconditions.checkNotNull(instance);
    return instance.workspaceName;
  }

  public static boolean gotInstance() {
    return instance != null;
  }

  public static void shutdown() {
    if (instance != null) {
      instance.stashFileListingPool.shutdown();
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

    if (instance != null) {
      instance.stashPathToRunfilesDir.clear();
      instance.pathToContents.clear();
      instance.sandboxToTarget.clear();
      instance.pathToLastModified.clear();
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

  /**
   * Before this function is called, stashContents will contain the inputs that were set up for the
   * action before executing it but the action might have written undeclared files into the sandbox
   * or deleted existing ones, therefore we need to crawl through all directories to see what's in
   * them and update stashContents.
   */
  private static void listContentsRecursively(
      Path root, Long timestamp, StashContents stashContents)
      throws IOException, InterruptedException {
    if (root.statIfFound().getLastChangeTime() > timestamp) {
      Set<String> dirsToKeep = new HashSet<>();
      Set<String> filesAndSymlinksToKeep = new HashSet<>();
      for (Dirent dirent : root.readdir(Symlinks.NOFOLLOW)) {
        if (Thread.interrupted()) {
          throw new InterruptedException();
        }
        Path absPath = root.getChild(dirent.getName());
        if (dirent.getType().equals(SYMLINK)) {
          if ((stashContents.filesToPath().containsKey(dirent.getName())
                  || stashContents.symlinksToPathFragment().containsKey(dirent.getName()))
              && absPath.stat().getLastChangeTime() <= timestamp) {
            filesAndSymlinksToKeep.add(dirent.getName());
          } else {
            absPath.delete();
          }
        } else if (dirent.getType().equals(DIRECTORY)) {
          if (stashContents.dirEntries().containsKey(dirent.getName())) {
            dirsToKeep.add(dirent.getName());
            listContentsRecursively(
                absPath, timestamp, stashContents.dirEntries().get(dirent.getName()));
          } else {
            absPath.deleteTree();
            stashContents.dirEntries().remove(dirent.getName());
          }
        } else {
          absPath.delete();
        }
      }

      stashContents.dirEntries().keySet().retainAll(dirsToKeep);
      stashContents.filesToPath().keySet().retainAll(filesAndSymlinksToKeep);
      stashContents.symlinksToPathFragment().keySet().retainAll(filesAndSymlinksToKeep);
    } else {
      for (var entry : stashContents.dirEntries().entrySet()) {
        Path absPath = root.getChild(entry.getKey());
        listContentsRecursively(absPath, timestamp, entry.getValue());
      }
    }
  }

  private ImmutableList<Path> sortStashesByMatchingTargetSegments(
      Label target, Collection<Path> stashes) {
    List<Path> sortedStashes = new ArrayList<>(stashes);
    Map<Path, Integer> countMap = new HashMap<>();
    String[] targetStr = null;
    if (target != null) {
      targetStr = target.getPackageName().split("/");
    }
    for (Path stash : stashes) {
      Label stashTarget = sandboxToTarget.getOrDefault(stash, /* defaultValue= */ null);
      if (target == null) {
        countMap.put(stash, stashTarget == null ? 1 : 0);
      } else {
        countMap.put(
            stash,
            stashTarget == null
                ? 0
                : Arrays.mismatch(targetStr, stashTarget.getPackageName().split("/")));
      }
    }
    return ImmutableList.sortedCopyOf(
        Comparator.comparingInt(countMap::get).reversed(), sortedStashes);
  }

  private void updateStashContentsAfterRunfilesMove(
      String stashedRunfiles, String currentRunfiles, StashContents stashContents) {
    ImmutableList<String> stashedRunfilesSegments =
        ImmutableList.copyOf(PathFragment.create(stashedRunfiles).segments());
    StashContents runfilesStashContents = stashContents;
    for (int i = 0; i < stashedRunfilesSegments.size() - 1; i++) {
      runfilesStashContents =
          Preconditions.checkNotNull(
              runfilesStashContents.dirEntries().get(stashedRunfilesSegments.get(i)));
    }
    runfilesStashContents =
        runfilesStashContents.dirEntries().remove(stashedRunfilesSegments.getLast());

    ImmutableList<String> currentRunfilesSegments =
        ImmutableList.copyOf(PathFragment.create(currentRunfiles).segments());
    StashContents currentStashContents = stashContents;
    for (int i = 0; i < currentRunfilesSegments.size() - 1; i++) {
      String segment = currentRunfilesSegments.get(i);
      currentStashContents.dirEntries().putIfAbsent(segment, new StashContents());
      currentStashContents = currentStashContents.dirEntries().get(segment);
    }
    currentStashContents.dirEntries().put(currentRunfilesSegments.getLast(), runfilesStashContents);
  }
}

