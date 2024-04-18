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

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.collect.ImmutableMap.toImmutableMap;

import com.google.auto.value.AutoValue;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.actions.cache.VirtualActionInput;
import com.google.devtools.build.lib.actions.cache.VirtualActionInput.EmptyActionInput;
import com.google.devtools.build.lib.analysis.test.TestConfiguration;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.exec.TreeDeleter;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.Sandbox;
import com.google.devtools.build.lib.server.FailureDetails.Sandbox.Code;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.FileSystemUtils.MoveResult;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.OptionsParsingResult;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Optional;
import java.util.Set;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.concurrent.atomic.AtomicBoolean;
import javax.annotation.Nullable;

/**
 * Helper methods that are shared by the different sandboxing strategies.
 *
 * <p>All sandboxed strategies within a build should share the same instance of this object.
 */
public final class SandboxHelpers {

  public static final String INACCESSIBLE_HELPER_DIR = "inaccessibleHelperDir";
  public static final String INACCESSIBLE_HELPER_FILE = "inaccessibleHelperFile";

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private static final AtomicBoolean warnedAboutMovesBeingCopies = new AtomicBoolean(false);

  /**
   * Moves all given outputs from a root to another.
   *
   * @param outputs outputs to move as relative paths to a root
   * @param sourceRoot source directory from which to resolve outputs
   * @param targetRoot target directory to which to move the resolved outputs from the source
   * @throws IOException if any of the moves fails
   */
  public static void moveOutputs(SandboxOutputs outputs, Path sourceRoot, Path targetRoot)
      throws IOException {
    for (Entry<PathFragment, PathFragment> output :
        Iterables.concat(outputs.files().entrySet(), outputs.dirs().entrySet())) {
      Path source = sourceRoot.getRelative(output.getValue());
      Path target = targetRoot.getRelative(output.getKey());
      if (source.isFile() || source.isSymbolicLink()) {
        // Ensure the target directory exists in the target. The directories for the action outputs
        // have already been created, but the spawn outputs may be different from the overall action
        // outputs. This is the case for test actions.
        target.getParentDirectory().createDirectoryAndParents();
        if (FileSystemUtils.moveFile(source, target).equals(MoveResult.FILE_COPIED)) {
          if (warnedAboutMovesBeingCopies.compareAndSet(false, true)) {
            logger.atWarning().log(
                "Moving files out of the sandbox (e.g. from %s to %s"
                    + ") had to be done with a file copy, which is detrimental to performance; are "
                    + "the two trees in different file systems?",
                source, target);
          }
        }
      } else if (source.isDirectory()) {
        try {
          source.renameTo(target);
        } catch (IOException e) {
          // Failed to move directory directly, thus move it recursively.
          target.createDirectory();
          FileSystemUtils.moveTreesBelow(source, target);
        }
      } else if (!source.exists()) {
        // This will show up as an error later
      } else {
        logger.atWarning().log(
            "Sandbox file %s for output %s is neither file nor symlink nor directory.",
            source, target);
      }
    }
  }

  /**
   * Cleans the existing sandbox at {@code root} to match the {@code inputs}, updating {@code
   * inputsToCreate} and {@code dirsToCreate} to not contain existing inputs and dir. Existing
   * directories or files that are either not needed {@code inputs} or doesn't have the right
   * content or symlink destination are removed.
   */
  public static void cleanExisting(
      Path root,
      SandboxInputs inputs,
      Set<PathFragment> inputsToCreate,
      Set<PathFragment> dirsToCreate,
      Path workDir)
      throws IOException, InterruptedException {
    cleanExisting(root, inputs, inputsToCreate, dirsToCreate, workDir, /* treeDeleter= */ null);
  }

  public static void cleanExisting(
      Path root,
      SandboxInputs inputs,
      Set<PathFragment> inputsToCreate,
      Set<PathFragment> dirsToCreate,
      Path workDir,
      @Nullable TreeDeleter treeDeleter)
      throws IOException, InterruptedException {
    cleanExisting(
        root,
        inputs,
        inputsToCreate,
        dirsToCreate,
        workDir,
        treeDeleter,
        /* stashContents= */ null);
  }

  public static void cleanExisting(
      Path root,
      SandboxInputs inputs,
      Set<PathFragment> inputsToCreate,
      Set<PathFragment> dirsToCreate,
      Path workDir,
      @Nullable TreeDeleter treeDeleter,
      @Nullable StashContents stashContents)
      throws IOException, InterruptedException {
    Path inaccessibleHelperDir = workDir.getRelative(INACCESSIBLE_HELPER_DIR);
    // Setting the permissions is necessary when we are using an asynchronous tree deleter in order
    // to move the directory first. This is not necessary for a synchronous tree deleter because the
    // permissions are only needed in the parent directory in that case.
    if (inaccessibleHelperDir.exists()) {
      inaccessibleHelperDir.setExecutable(true);
      inaccessibleHelperDir.setWritable(true);
      inaccessibleHelperDir.setReadable(true);
    }

    // To avoid excessive scanning of dirsToCreate for prefix dirs, we prepopulate this set of
    // prefixes.
    Set<PathFragment> prefixDirs = new HashSet<>();
    for (PathFragment dir : dirsToCreate) {
      if (Thread.interrupted()) {
        throw new InterruptedException();
      }
      PathFragment parent = dir.getParentDirectory();
      while (parent != null && !prefixDirs.contains(parent)) {
        prefixDirs.add(parent);
        parent = parent.getParentDirectory();
      }
    }
    cleanRecursively(
        root,
        inputs,
        inputsToCreate,
        dirsToCreate,
        workDir,
        prefixDirs,
        treeDeleter,
        stashContents);
  }

  /**
   * Deletes unnecessary files/directories and updates the sets if something on disk is already
   * correct and doesn't need any changes.
   */
  private static void cleanRecursively(
      Path root,
      SandboxInputs inputs,
      Set<PathFragment> inputsToCreate,
      Set<PathFragment> dirsToCreate,
      Path workDir,
      Set<PathFragment> prefixDirs,
      @Nullable TreeDeleter treeDeleter,
      @Nullable StashContents stashContents)
      throws IOException, InterruptedException {
    Path execroot = workDir.getParentDirectory();
    Preconditions.checkNotNull(stashContents);
    for (var fileDirent : stashContents.filesToRootedPath.entrySet()) {
      if (Thread.interrupted()) {
        throw new InterruptedException();
      }
      Path absPath = root.getChild(fileDirent.getKey());
      PathFragment pathRelativeToWorkDir;
      if (absPath.startsWith(workDir)) {
        // path is under workDir, i.e. execroot/<workspace name>. Simply get the relative path.
        pathRelativeToWorkDir = absPath.relativeTo(workDir);
      } else {
        // path is not under workDir, which means it belongs to one of external repositories
        // symlinked directly under execroot. Get the relative path based on there and prepend it
        // with the designated prefix, '../', so that it's still a valid relative path to workDir.
        pathRelativeToWorkDir =
            LabelConstants.EXPERIMENTAL_EXTERNAL_PATH_PREFIX.getRelative(
                absPath.relativeTo(execroot));
      }
      Optional<Path> destination =
          getExpectedSymlinkDestinationForFiles(pathRelativeToWorkDir, inputs);
      if (destination.isPresent() && fileDirent.getValue().equals(destination.get())) {
        inputsToCreate.remove(pathRelativeToWorkDir);
      } else {
        absPath.delete();
      }
    }
    for (var symlinkDirent : stashContents.symlinksToPathFragment.entrySet()) {
      if (Thread.interrupted()) {
        throw new InterruptedException();
      }
      Path absPath = root.getChild(symlinkDirent.getKey());
      PathFragment pathRelativeToWorkDir;
      if (absPath.startsWith(workDir)) {
        // path is under workDir, i.e. execroot/<workspace name>. Simply get the relative path.
        pathRelativeToWorkDir = absPath.relativeTo(workDir);
      } else {
        // path is not under workDir, which means it belongs to one of external repositories
        // symlinked directly under execroot. Get the relative path based on there and prepend it
        // with the designated prefix, '../', so that it's still a valid relative path to workDir.
        pathRelativeToWorkDir =
            LabelConstants.EXPERIMENTAL_EXTERNAL_PATH_PREFIX.getRelative(
                absPath.relativeTo(execroot));
      }
      Optional<PathFragment> destination =
          getExpectedSymlinkDestinationForSymlinks(pathRelativeToWorkDir, inputs);
      if (destination.isPresent() && symlinkDirent.getValue().equals(destination.get())) {
        inputsToCreate.remove(pathRelativeToWorkDir);
      } else {
        absPath.delete();
      }
    }
    for (var dirent : stashContents.dirEntries.entrySet()) {
      if (Thread.interrupted()) {
        throw new InterruptedException();
      }
      Path absPath = root.getChild(dirent.getKey());
      PathFragment pathRelativeToWorkDir;
      if (absPath.startsWith(workDir)) {
        // path is under workDir, i.e. execroot/<workspace name>. Simply get the relative path.
        pathRelativeToWorkDir = absPath.relativeTo(workDir);
      } else {
        // path is not under workDir, which means it belongs to one of external repositories
        // symlinked directly under execroot. Get the relative path based on there and prepend it
        // with the designated prefix, '../', so that it's still a valid relative path to workDir.
        pathRelativeToWorkDir =
            LabelConstants.EXPERIMENTAL_EXTERNAL_PATH_PREFIX.getRelative(
                absPath.relativeTo(execroot));
      }
      if (dirsToCreate.contains(pathRelativeToWorkDir)
          || prefixDirs.contains(pathRelativeToWorkDir)) {
        cleanRecursively(
            absPath,
            inputs,
            inputsToCreate,
            dirsToCreate,
            workDir,
            prefixDirs,
            treeDeleter,
            dirent.getValue());
        dirsToCreate.remove(pathRelativeToWorkDir);
      } else {
        if (treeDeleter == null) {
          // TODO(bazel-team): Use async tree deleter for workers too
          absPath.deleteTree();
        } else {
          treeDeleter.deleteTree(absPath);
        }
      }
    }
  }

  /**
   * Returns what the destination of the symlink {@code file} should be, according to {@code
   * inputs}.
   */
  static Optional<Path> getExpectedSymlinkDestinationForFiles(
      PathFragment fragment, SandboxInputs inputs) {
    return Optional.ofNullable(inputs.getFiles().get(fragment));
  }

  static Optional<PathFragment> getExpectedSymlinkDestinationForSymlinks(
      PathFragment fragment, SandboxInputs inputs) {
    return Optional.ofNullable(inputs.getSymlinks().get(fragment));
  }

  /** Populates the provided sets with the inputs and directories that need to be created. */
  public static void populateInputsAndDirsToCreate(
      Set<PathFragment> writableDirs,
      Set<PathFragment> inputsToCreate,
      Set<PathFragment> dirsToCreate,
      Iterable<PathFragment> inputFiles,
      SandboxOutputs outputs) {
    // Add all worker files, input files, and the parent directories.
    for (PathFragment input : inputFiles) {
      inputsToCreate.add(input);
      dirsToCreate.add(input.getParentDirectory());
    }

    // And all parent directories of output files. Note that we don't add the files themselves --
    // any pre-existing files that have the same path as an output should get deleted.
    for (PathFragment file : outputs.files().values()) {
      dirsToCreate.add(file.getParentDirectory());
    }

    // Add all output directories.
    dirsToCreate.addAll(outputs.dirs().values());

    // Add some directories that should be writable, and thus exist.
    dirsToCreate.addAll(writableDirs);
  }

  /**
   * Creates directory and all ancestors for it at a given path.
   *
   * <p>This method uses (and updates) the set of already known directories in order to minimize the
   * I/O involved with creating directories. For example a path of {@code 1/2/3/4} created after
   * {@code 1/2/3/5} only calls for creating {@code 1/2/3/5}. We can use the set of known
   * directories to discover that {@code 1/2/3} already exists instead of deferring to the
   * filesystem for it.
   */
  public static void createDirectoryAndParentsInSandboxRoot(
      Path path, Set<Path> knownDirectories, Path sandboxExecRoot) throws IOException {
    if (knownDirectories.contains(path)) {
      return;
    }
    createDirectoryAndParentsInSandboxRoot(
        checkNotNull(
            path.getParentDirectory(),
            "Path %s is not under/siblings of sandboxExecRoot: %s",
            path,
            sandboxExecRoot),
        knownDirectories,
        sandboxExecRoot);
    path.createDirectory();
    knownDirectories.add(path);
  }

  /**
   * Creates all directories needed for the sandbox.
   *
   * <p>No input can be a child of another input, because otherwise we might try to create a symlink
   * below another symlink we created earlier - which means we'd actually end up writing somewhere
   * in the workspace.
   *
   * <p>If all inputs were regular files, this situation could naturally not happen - but
   * unfortunately, we might get the occasional action that has directories in its inputs.
   *
   * <p>Creating all parent directories first ensures that we can safely create symlinks to
   * directories, too, because we'll get an IOException with EEXIST if inputs happen to be nested
   * once we start creating the symlinks for all inputs.
   *
   * @param strict If true, absolute directories or directories with multiple up-level references
   *     are disallowed, for stricter sandboxing.
   */
  public static void createDirectories(
      Iterable<PathFragment> dirsToCreate, Path dir, boolean strict)
      throws IOException, InterruptedException {
    Set<Path> knownDirectories = new HashSet<>();
    // Add sandboxExecRoot and it's parent -- all paths must fall under the parent of
    // sandboxExecRoot and we know that sandboxExecRoot exists. This stops the recursion in
    // createDirectoryAndParentsInSandboxRoot.
    knownDirectories.add(dir);
    knownDirectories.add(dir.getParentDirectory());

    for (PathFragment path : dirsToCreate) {
      if (Thread.interrupted()) {
        throw new InterruptedException();
      }
      if (strict) {
        Preconditions.checkArgument(!path.isAbsolute(), path);
        if (path.containsUplevelReferences() && path.isMultiSegment()) {
          // Allow a single up-level reference to allow inputs from the siblings of the main
          // repository in the sandbox execution root, but forbid multiple up-level references.
          // PathFragment is normalized, so up-level references are guaranteed to be at the
          // beginning.
          Preconditions.checkArgument(
              !PathFragment.containsUplevelReferences(path.getSegment(1)),
              "%s escapes the sandbox exec root.",
              path);
        }
      }

      createDirectoryAndParentsInSandboxRoot(dir.getRelative(path), knownDirectories, dir);
    }
  }

  static FailureDetail createFailureDetail(String message, Code detailedCode) {
    return FailureDetail.newBuilder()
        .setMessage(message)
        .setSandbox(Sandbox.newBuilder().setCode(detailedCode))
        .build();
  }

  /** Adds additional bind mounts entries from {@code paths} to {@code bindMounts}. */
  public static void mountAdditionalPaths(
      List<Entry<String, String>> paths, Path sandboxExecRoot, SortedMap<Path, Path> bindMounts)
      throws UserExecException {
    FileSystem fs = sandboxExecRoot.getFileSystem();
    for (Map.Entry<String, String> additionalMountPath : paths) {
      try {
        final Path mountTarget = fs.getPath(additionalMountPath.getValue());
        // If source path is relative, treat it as a relative path inside the execution root
        final Path mountSource = sandboxExecRoot.getRelative(additionalMountPath.getKey());
        // If a target has more than one source path, the latter one will take effect.
        bindMounts.put(mountTarget, mountSource);
      } catch (IllegalArgumentException e) {
        throw new UserExecException(
            createFailureDetail(
                String.format("Error occurred when analyzing bind mount pairs. %s", e.getMessage()),
                Code.BIND_MOUNT_ANALYSIS_FAILURE));
      }
    }
  }

  /** Wrapper class for the inputs of a sandbox. */
  public static final class SandboxInputs {
    private final Map<PathFragment, Path> files;
    private final Map<VirtualActionInput, byte[]> virtualInputs;
    private final Map<PathFragment, PathFragment> symlinks;

    private static final SandboxInputs EMPTY_INPUTS =
        new SandboxInputs(ImmutableMap.of(), ImmutableMap.of(), ImmutableMap.of());

    public SandboxInputs(
        Map<PathFragment, Path> files,
        Map<VirtualActionInput, byte[]> virtualInputs,
        Map<PathFragment, PathFragment> symlinks) {
      this.files = files;
      this.virtualInputs = virtualInputs;
      this.symlinks = symlinks;
    }

    public static SandboxInputs getEmptyInputs() {
      return EMPTY_INPUTS;
    }

    public Map<PathFragment, Path> getFiles() {
      return files;
    }

    public Map<PathFragment, PathFragment> getSymlinks() {
      return symlinks;
    }

    public ImmutableMap<VirtualActionInput, byte[]> getVirtualInputDigests() {
      return ImmutableMap.copyOf(virtualInputs);
    }

    /**
     * Returns a new SandboxInputs instance with only the inputs/symlinks listed in {@code allowed}
     * included.
     */
    public SandboxInputs limitedCopy(Set<PathFragment> allowed) {
      return new SandboxInputs(
          Maps.filterKeys(files, allowed::contains),
          ImmutableMap.of(),
          Maps.filterKeys(symlinks, allowed::contains));
    }

    @Override
    public String toString() {
      return "Files: " + files + "\nVirtualInputs: " + virtualInputs + "\nSymlinks: " + symlinks;
    }
  }

  /**
   * Returns the inputs of a Spawn as a map of PathFragments relative to an execRoot to paths in the
   * host filesystem where the input files can be found.
   *
   * @param inputMap the map of action inputs and where they should be visible in the action
   * @param execRoot the exec root
   * @throws IOException if processing symlinks fails
   */
  @CanIgnoreReturnValue
  public SandboxInputs processInputFiles(Map<PathFragment, ActionInput> inputMap, Path execRoot)
      throws IOException, InterruptedException {
    Map<PathFragment, Path> inputFiles = new TreeMap<>();
    Map<PathFragment, PathFragment> inputSymlinks = new TreeMap<>();
    Map<VirtualActionInput, byte[]> virtualInputs = new HashMap<>();

    for (Map.Entry<PathFragment, ActionInput> e : inputMap.entrySet()) {
      if (Thread.interrupted()) {
        throw new InterruptedException();
      }
      PathFragment pathFragment = e.getKey();
      ActionInput actionInput = e.getValue();
      if (actionInput instanceof VirtualActionInput input) {
        byte[] digest = input.atomicallyWriteRelativeTo(execRoot);
        virtualInputs.put(input, digest);
      }

      if (actionInput.isSymlink()) {
        Path inputPath = execRoot.getRelative(actionInput.getExecPath());
        inputSymlinks.put(pathFragment, inputPath.readSymbolicLink());
      } else {
        Path inputPath =
            actionInput instanceof EmptyActionInput
                ? null
                : execRoot.getRelative(actionInput.getExecPath());
        inputFiles.put(pathFragment, inputPath);
      }
    }
    return new SandboxInputs(inputFiles, virtualInputs, inputSymlinks);
  }

  /** The file and directory outputs of a sandboxed spawn. */
  @AutoValue
  public abstract static class SandboxOutputs {

    /** A map from output file exec paths to paths in the sandbox. */
    public abstract ImmutableMap<PathFragment, PathFragment> files();

    /** A map from output directory exec paths to paths in the sandbox. */
    public abstract ImmutableMap<PathFragment, PathFragment> dirs();

    private static final SandboxOutputs EMPTY_OUTPUTS =
        SandboxOutputs.create(ImmutableMap.of(), ImmutableMap.of());

    public static SandboxOutputs create(
        ImmutableMap<PathFragment, PathFragment> files,
        ImmutableMap<PathFragment, PathFragment> dirs) {
      return new AutoValue_SandboxHelpers_SandboxOutputs(files, dirs);
    }

    public static SandboxOutputs create(
        ImmutableSet<PathFragment> files, ImmutableSet<PathFragment> dirs) {
      return new AutoValue_SandboxHelpers_SandboxOutputs(
          files.stream().collect(toImmutableMap(f -> f, f -> f)),
          dirs.stream().collect(toImmutableMap(d -> d, d -> d)));
    }

    public static SandboxOutputs getEmptyInstance() {
      return EMPTY_OUTPUTS;
    }
  }

  public SandboxOutputs getOutputs(Spawn spawn) {
    ImmutableMap.Builder<PathFragment, PathFragment> files = ImmutableMap.builder();
    ImmutableMap.Builder<PathFragment, PathFragment> dirs = ImmutableMap.builder();
    for (ActionInput output : spawn.getOutputFiles()) {
      PathFragment mappedPath = spawn.getPathMapper().map(output.getExecPath());
      if (output instanceof Artifact && ((Artifact) output).isTreeArtifact()) {
        dirs.put(output.getExecPath(), mappedPath);
      } else {
        files.put(output.getExecPath(), mappedPath);
      }
    }
    return SandboxOutputs.create(files.build(), dirs.build());
  }

  /**
   * Returns true if the build options are set in a way that requires network access for all
   * actions. This is separate from {@link
   * com.google.devtools.build.lib.actions.Spawns#requiresNetwork} to avoid having to keep a
   * reference to the full set of build options (and also for performance, since this only needs to
   * be checked once-per-build).
   */
  boolean shouldAllowNetwork(OptionsParsingResult buildOptions) {
    // Allow network access, when --java_debug is specified, otherwise we can't connect to the
    // remote debug server of the test. This intentionally overrides the "block-network" execution
    // tag.
    return buildOptions
        .getOptions(TestConfiguration.TestOptions.class)
        .testArguments
        .contains("--wrapper_script_flag=--debug");
  }

  public static class StashContents {
    Map<String, Path> filesToRootedPath = new HashMap<>();
    Map<String, PathFragment> symlinksToPathFragment = new HashMap<>();
    Map<String, StashContents> dirEntries = new HashMap<>();
  }
}
