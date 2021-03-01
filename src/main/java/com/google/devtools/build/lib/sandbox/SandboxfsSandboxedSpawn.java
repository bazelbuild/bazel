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

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.base.Joiner;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.exec.TreeDeleter;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxInputs;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxOutputs;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import javax.annotation.Nullable;

/**
 * Creates an execRoot for a Spawn that contains all required input files by mounting a sandboxfs
 * FUSE filesystem on the provided path.
 */
class SandboxfsSandboxedSpawn implements SandboxedSpawn {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  /** Sequence number to assign a unique subtree to each action within the mount point. */
  private static final AtomicInteger lastId = new AtomicInteger();

  /** The sandboxfs instance to use for this spawn. */
  private final SandboxfsProcess process;

  /** Arguments to pass to the spawn, including the binary name. */
  private final List<String> arguments;

  /** Environment variables to pass to the spawn. */
  private final Map<String, String> environment;

  /** Collection of input files to be made available to the spawn in read-only mode. */
  private final SandboxInputs inputs;

  /** Collection of output files to expect from the spawn. */
  private final SandboxOutputs outputs;

  /** Collection of directories where the spawn can write files to relative to {@link #execRoot}. */
  private final Set<PathFragment> writableDirs;

  /** Map the targets of symlinks within the sandbox if true. */
  private final boolean mapSymlinkTargets;

  /** Scheduler for tree deletions. */
  private final TreeDeleter treeDeleter;

  /**
   * Writable directory where the spawn runner keeps control files and the execroot outside of the
   * sandboxfs instance.
   */
  private final Path sandboxPath;

  /**
   * Writable directory to support the writes performed by the command. This acts as the target
   * of all writable mappings in the sandboxfs instance.
   */
  private final Path sandboxScratchDir;

  /** Path to the working directory of the command. */
  private final Path execRoot;

  /**
   * Name of the sandbox within the sandboxfs mount point, which is just the basename of the
   * top-level directory where all execroot paths start.
   */
  private final String sandboxName;

  /** Path to the execroot within the sandbox. */
  private final PathFragment rootFragment;

  /** Flag to track whether the sandbox needs to be unmapped. */
  private boolean sandboxIsMapped;

  @Nullable private final Path statisticsPath;

  /**
   * Constructs a new sandboxfs-based spawn runner.
   *
   * @param process sandboxfs instance to use for this spawn
   * @param sandboxPath writable directory where the spawn runner keeps control files
   * @param arguments arguments to pass to the spawn, including the binary name
   * @param environment environment variables to pass to the spawn
   * @param inputs input files to be made available to the spawn in read-only mode
   * @param outputs output files to expect from the spawn
   * @param writableDirs directories where the spawn can write files to, relative to the sandbox's
   *     dynamically-allocated execroot
   * @param mapSymlinkTargets map the targets of symlinks within the sandbox if true
   * @param treeDeleter scheduler for tree deletions
   */
  SandboxfsSandboxedSpawn(
      SandboxfsProcess process,
      Path sandboxPath,
      String workspaceName,
      List<String> arguments,
      Map<String, String> environment,
      SandboxInputs inputs,
      SandboxOutputs outputs,
      Set<PathFragment> writableDirs,
      boolean mapSymlinkTargets,
      TreeDeleter treeDeleter,
      @Nullable Path statisticsPath) {
    this.process = process;
    this.arguments = arguments;
    this.environment = environment;
    this.inputs = inputs;
    for (PathFragment path : outputs.files()) {
      checkArgument(!path.isAbsolute(), "outputs %s must be relative", path);
    }
    for (PathFragment path : outputs.dirs()) {
      checkArgument(!path.isAbsolute(), "outputs %s must be relative", path);
    }
    this.outputs = outputs;
    for (PathFragment path : writableDirs) {
      checkArgument(!path.isAbsolute(), "writable directory %s must be relative", path);
    }
    this.writableDirs = writableDirs;
    this.mapSymlinkTargets = mapSymlinkTargets;
    this.treeDeleter = treeDeleter;

    this.sandboxPath = sandboxPath;
    this.sandboxScratchDir = sandboxPath.getRelative("scratch");

    int id = lastId.getAndIncrement();
    this.sandboxName = "" + id;
    this.sandboxIsMapped = false;
    this.statisticsPath = statisticsPath;

    // b/64689608: The execroot of the sandboxed process must end with the workspace name, just
    // like the normal execroot does. Some tools walk their path hierarchy looking for this
    // component and misbehave if they don't find it.
    this.execRoot =
        process.getMountPoint().getRelative(this.sandboxName).getRelative(workspaceName);
    this.rootFragment = PathFragment.create("/" + workspaceName);
  }

  @Override
  public Path getSandboxExecRoot() {
    return execRoot;
  }

  @Override
  public List<String> getArguments() {
    return arguments;
  }

  @Override
  public Map<String, String> getEnvironment() {
    return environment;
  }

  @Override
  public Path getStatisticsPath() {
    return statisticsPath;
  }

  @Override
  public void createFileSystem() throws IOException {
    sandboxScratchDir.createDirectory();

    inputs.materializeVirtualInputs(sandboxScratchDir);

    Set<PathFragment> dirsToCreate = new HashSet<>(writableDirs);
    for (PathFragment output : outputs.files()) {
      dirsToCreate.add(output.getParentDirectory());
    }
    dirsToCreate.addAll(outputs.dirs());
    for (PathFragment dir : dirsToCreate) {
      sandboxScratchDir.getRelative(dir).createDirectoryAndParents();
    }

    createSandbox(process, sandboxName, rootFragment, sandboxScratchDir, inputs, mapSymlinkTargets);
    sandboxIsMapped = true;
  }

  @Override
  public void copyOutputs(Path targetExecRoot) throws IOException {
    // TODO(jmmv): If we knew the targetExecRoot when setting up the spawn, we may be able to
    // configure sandboxfs so that the output files are written directly to their target locations.
    // This would avoid having to move them after-the-fact.
    SandboxHelpers.moveOutputs(outputs, sandboxScratchDir, targetExecRoot);
  }

  @Override
  public void delete() {
    // We can only ask sandboxfs to unmap a sandbox if we successfully finished creating it.
    // Otherwise, the request may fail, or we may fail our own checks that validate the lifecycle of
    // the sandboxes.
    if (sandboxIsMapped) {
      try {
        process.destroySandbox(sandboxName);
      } catch (IOException e) {
        // We use independent subdirectories for each action, so a failure to unmap one, while
        // annoying, is not a big deal.  The sandboxfs instance will be unmounted anyway after
        // the build, which will cause these to go away anyway.
        logger.atWarning().withCause(e).log("Cannot unmap %s", sandboxName);
      }
      sandboxIsMapped = false;
    }

    try {
      treeDeleter.deleteTree(sandboxPath);
    } catch (IOException e) {
      // This usually means that the Spawn itself exited but still has children running that
      // we couldn't wait for, which now block deletion of the sandbox directory.  (Those processes
      // may be creating new files in the directories we are trying to delete, preventing the
      // deletion.)  On Linux this should never happen: we use PID namespaces when available and the
      // subreaper feature when not to make sure all children have been reliably killed before
      // returning, but on other OSes this might not always work.  The SandboxModule will try to
      // delete them again when the build is all done, at which point it hopefully works... so let's
      // just go on here.
    }
  }

  /**
   * Maps the targets of relative symlinks into the sandbox.
   *
   * <p>Symlinks with relative targets are tricky business. Consider this simple case: the source
   * tree contains {@code dir/file.h} and {@code dir/symlink.h} where {@code dir/symlink.h}'s target
   * is {@code ./file.h}. If {@code dir/symlink.h} is supplied as an input, we must preserve its
   * target "as is" to avoid confusing any tooling: for example, the C compiler will understand that
   * both {@code dir/file.h} and {@code dir/symlink.h} are the same entity and handle them
   * appropriately. (We did encounter a case where the compiler complained about duplicate symbols
   * because we exposed symlinks as regular files.)
   *
   * <p>However, there is no guarantee that the target of the symlink is mapped in the sandbox. You
   * may think that this is a bug in the rules, and you would probably be right, but until those
   * rules are fixed, we must supply a workaround. Therefore, we must handle these two cases: if the
   * target is explicitly mapped, we do nothing. If it isn't, we have to compute where the target
   * lives within the sandbox and map that as well. Oh, and we have to do this recursively.
   *
   * @param path path to expose within the sandbox
   * @param symlink path to the target of the mapping specified by {@code path}
   * @param mappings mutable collection of mappings to extend with the new symlink entries. Note
   *     that the entries added to this map may correspond to explicitly-mapped entries, so the
   *     caller must check this to avoid duplicate mappings
   * @throws IOException if we fail to resolve symbolic links
   */
  private static void computeSymlinkMappings(
      PathFragment path, Path symlink, Map<PathFragment, PathFragment> mappings)
      throws IOException {
    for (; ; ) {
      PathFragment symlinkTarget = symlink.readSymbolicLinkUnchecked();
      if (!symlinkTarget.isAbsolute()) {
        PathFragment keyParent = path.getParentDirectory();
        if (keyParent == null) {
          throw new IOException("Cannot resolve " + symlinkTarget + " relative to " + path);
        }
        PathFragment key = keyParent.getRelative(symlinkTarget);

        Path valueParent = symlink.getParentDirectory();
        if (valueParent == null) {
          throw new IOException("Cannot resolve " + symlinkTarget + " relative to " + symlink);
        }
        Path value = valueParent.getRelative(symlinkTarget);
        mappings.put(key, value.asFragment());

        if (value.isSymbolicLink()) {
          path = key;
          symlink = value;
          continue;
        }
      }
      break;
    }
  }

  /**
   * Creates a new set of mappings to sandbox the given inputs.
   *
   * @param process the sandboxfs instance on which to create the sandbox
   * @param sandboxName the name of the sandbox to pass to sandboxfs
   * @param rootFragment path within the sandbox to the execroot to create
   * @param scratchDir writable used as the target for all writable mappings
   * @param inputs collection of paths to expose within the sandbox as read-only mappings, given as
   *     a map of mapped path to target path. The target path may be null, in which case an empty
   *     read-only file is mapped.
   * @param sandboxfsMapSymlinkTargets map the targets of symlinks within the sandbox if true
   * @return the collection of mappings to use for reconfiguration
   * @throws IOException if we fail to resolve symbolic links
   */
  private static void createSandbox(
      SandboxfsProcess process,
      String sandboxName,
      PathFragment rootFragment,
      Path scratchDir,
      SandboxInputs inputs,
      boolean sandboxfsMapSymlinkTargets)
      throws IOException {
    // Path to the empty file used as the target of mappings that don't provide one.  This is
    // lazily created and initialized only when we need such a mapping.  It's safe to share the
    // same empty file across all such mappings because this file is exposed as read-only.
    //
    // We cannot use /dev/null, as we used to do in the past, because exposing devices via a
    // FUSE file system (which sandboxfs is) requires root privileges.
    PathFragment emptyFile = null;

    // Collection of extra mappings needed to represent the targets of relative symlinks. Lazily
    // created once we encounter the first symlink in the list of inputs.
    Map<PathFragment, PathFragment> symlinks = null;

    for (Map.Entry<PathFragment, Path> entry : inputs.getFiles().entrySet()) {
      if (entry.getValue() == null) {
        if (emptyFile == null) {
          Path emptyFilePath = scratchDir.getRelative("empty");
          FileSystemUtils.createEmptyFile(emptyFilePath);
          emptyFile = emptyFilePath.asFragment();
        }
      } else {
        if (sandboxfsMapSymlinkTargets && entry.getValue().isSymbolicLink()) {
          if (symlinks == null) {
            symlinks = new HashMap<>();
          }
          computeSymlinkMappings(entry.getKey(), entry.getValue(), symlinks);
        }
      }
    }

    // IMPORTANT: Keep the code in the lambda passed to createSandbox() free from any operations
    // that may block. This includes doing any kind of I/O. We used to include the loop above in
    // this call and doing so cost 2-3% of the total build time measured on an iOS build with many
    // actions that have thousands of inputs each.
    @Nullable final PathFragment finalEmptyFile = emptyFile;
    @Nullable final Map<PathFragment, PathFragment> finalSymlinks = symlinks;
    process.createSandbox(
        sandboxName,
        (mapper) -> {
          mapper.map(rootFragment, scratchDir.asFragment(), true);

          for (Map.Entry<PathFragment, Path> entry : inputs.getFiles().entrySet()) {
            PathFragment target;
            if (entry.getValue() == null) {
              checkNotNull(finalEmptyFile, "Must have been initialized above by matching logic");
              target = finalEmptyFile;
            } else {
              target = entry.getValue().asFragment();
            }
            mapper.map(rootFragment.getRelative(entry.getKey()), target, false);
          }

          if (finalSymlinks != null) {
            for (Map.Entry<PathFragment, PathFragment> entry : finalSymlinks.entrySet()) {
              if (!inputs.getFiles().containsKey(entry.getKey())) {
                mapper.map(rootFragment.getRelative(entry.getKey()), entry.getValue(), false);
              }
            }
          }
        });

    // sandboxfs probably doesn't support symlinks.
    // TODO(jmmv): This claim is simply not true. Figure out why this code snippet was added and
    // address the real problem.
    if (!inputs.getSymlinks().isEmpty()) {
      throw new IOException(
          "sandboxfs sandbox does not support unresolved symlinks "
              + Joiner.on(", ").join(inputs.getSymlinks().keySet()));
    }
  }
}
