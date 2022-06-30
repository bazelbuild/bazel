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
import static com.google.common.base.Preconditions.checkState;
import static com.google.devtools.build.lib.vfs.Dirent.Type.DIRECTORY;
import static com.google.devtools.build.lib.vfs.Dirent.Type.SYMLINK;

import com.google.auto.value.AutoValue;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.common.flogger.GoogleLogger;
import com.google.common.hash.HashingOutputStream;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.cache.VirtualActionInput;
import com.google.devtools.build.lib.actions.cache.VirtualActionInput.EmptyActionInput;
import com.google.devtools.build.lib.analysis.test.TestConfiguration;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.FileSystemUtils.MoveResult;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.common.options.OptionsParsingResult;
import java.io.IOException;
import java.io.OutputStream;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.TreeMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Helper methods that are shared by the different sandboxing strategies.
 *
 * <p>All sandboxed strategies within a build should share the same instance of this object.
 */
public final class SandboxHelpers {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private static final AtomicBoolean warnedAboutMovesBeingCopies = new AtomicBoolean(false);
  /**
   * If true, materialize virtual inputs only inside the sandbox, not the output tree. This flag
   * exists purely to support rolling this out as the defaut in a controlled manner.
   */
  private final boolean delayVirtualInputMaterialization;

  /**
   * Constructs a new collection of helpers.
   *
   * @param delayVirtualInputMaterialization whether to materialize virtual inputs only inside the
   *     sandbox
   */
  public SandboxHelpers(boolean delayVirtualInputMaterialization) {
    this.delayVirtualInputMaterialization = delayVirtualInputMaterialization;
  }

  /**
   * Writes a virtual input file so that the final file is always consistent to all readers.
   *
   * <p>This function exists to aid dynamic scheduling. Param files are inputs, so they need to be
   * written without holding the output lock. When we have competing unsandboxed spawn runners (like
   * persistent workers), it's possible for them to clash in these writes, either encountering
   * missing file errors or encountering incomplete data. But given that we can assume both spawn
   * runners will write the same contents, we can write those as temporary files and then perform a
   * rename, which has atomic semantics on Unix, and thus keep all readers always seeing consistent
   * contents.
   *
   * @param input the virtual input file to write
   * @param outputPath final path where the virtual input file ought to live
   * @param uniqueSuffix a filename extension that is different between the local spawn runners and
   *     the remote ones
   * @return digest of written virtual input
   * @throws IOException if we fail to write the virtual input file
   */
  // TODO(b/150963503): We are using atomic file system moves for synchronization... but Bazel
  // should not be able to reach this state. Which means we should probably be doing some other
  // form of synchronization in-process before touching the file system.
  public static byte[] atomicallyWriteVirtualInput(
      VirtualActionInput input, Path outputPath, String uniqueSuffix) throws IOException {
    Path tmpPath = outputPath.getFileSystem().getPath(outputPath.getPathString() + uniqueSuffix);
    tmpPath.getParentDirectory().createDirectoryAndParents();
    try {
      byte[] digest = writeVirtualInputTo(input, tmpPath);
      // We expect the following to replace the params file atomically in case we are using
      // the dynamic scheduler and we are racing the remote strategy writing this same file.
      tmpPath.renameTo(outputPath);
      tmpPath = null; // Avoid unnecessary deletion attempt.
      return digest;
    } finally {
      try {
        if (tmpPath != null) {
          // Make sure we don't leave temp files behind if we are interrupted.
          tmpPath.delete();
        }
      } catch (IOException e) {
        // Ignore.
      }
    }
  }

  /**
   * Moves all given outputs from a root to another.
   *
   * <p>This is a support function to help with the implementation of {@link
   * SandboxfsSandboxedSpawn#copyOutputs(Path)}.
   *
   * @param outputs outputs to move as relative paths to a root
   * @param sourceRoot source directory from which to resolve outputs
   * @param targetRoot target directory to which to move the resolved outputs from the source
   * @throws IOException if any of the moves fails
   */
  public static void moveOutputs(SandboxOutputs outputs, Path sourceRoot, Path targetRoot)
      throws IOException {
    for (PathFragment output : Iterables.concat(outputs.files(), outputs.dirs())) {
      Path source = sourceRoot.getRelative(output);
      Path target = targetRoot.getRelative(output);
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
      throws IOException {
    // To avoid excessive scanning of dirsToCreate for prefix dirs, we prepopulate this set of
    // prefixes.
    Set<PathFragment> prefixDirs = new HashSet<>();
    for (PathFragment dir : dirsToCreate) {
      PathFragment parent = dir.getParentDirectory();
      while (parent != null && !prefixDirs.contains(parent)) {
        prefixDirs.add(parent);
        parent = parent.getParentDirectory();
      }
    }

    cleanRecursively(root, inputs, inputsToCreate, dirsToCreate, workDir, prefixDirs);
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
      Set<PathFragment> prefixDirs)
      throws IOException {
    Path execroot = workDir.getParentDirectory();
    for (Dirent dirent : root.readdir(Symlinks.NOFOLLOW)) {
      Path absPath = root.getChild(dirent.getName());
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
      Optional<String> destination = getExpectedSymlinkDestination(pathRelativeToWorkDir, inputs);
      if (destination.isPresent()) {
        if (SYMLINK.equals(dirent.getType()) && absPath.readSymbolicLink()
            .equals(destination.get())) {
          inputsToCreate.remove(pathRelativeToWorkDir);
        } else {
          absPath.delete();
        }
      } else if (DIRECTORY.equals(dirent.getType())) {
        if (dirsToCreate.contains(pathRelativeToWorkDir)
            || prefixDirs.contains(pathRelativeToWorkDir)) {
          cleanRecursively(absPath, inputs, inputsToCreate, dirsToCreate, workDir, prefixDirs);
          dirsToCreate.remove(pathRelativeToWorkDir);
        } else {
          absPath.deleteTree();
        }
      } else if (!inputsToCreate.contains(pathRelativeToWorkDir)) {
        absPath.delete();
      }
    }
  }

  /**
   * Returns what the destination of the symlink {@code file} should be, according to
   * {@code inputs}.
   */
  static Optional<String> getExpectedSymlinkDestination(
      PathFragment fragment, SandboxInputs inputs) {
    Path file = inputs.getFiles().get(fragment);
    if (file != null) {
      return Optional.of(file.asFragment().getPathString());
    }
    return Optional.ofNullable(inputs.getSymlinks().get(fragment));
  }

  /** Populates the provided sets with the inputs and directories that need to be created. */
  public static void populateInputsAndDirsToCreate(
      Set<PathFragment> writableDirs,
      Set<PathFragment> inputsToCreate,
      LinkedHashSet<PathFragment> dirsToCreate,
      Iterable<PathFragment> inputFiles,
      ImmutableSet<PathFragment> outputFiles,
      ImmutableSet<PathFragment> outputDirs) {
    // Add all worker files, input files, and the parent directories.
    for (PathFragment input : inputFiles) {
      inputsToCreate.add(input);
      dirsToCreate.add(input.getParentDirectory());
    }

    // And all parent directories of output files. Note that we don't add the files themselves --
    // any pre-existing files that have the same path as an output should get deleted.
    for (PathFragment file : outputFiles) {
      dirsToCreate.add(file.getParentDirectory());
    }

    // Add all output directories.
    dirsToCreate.addAll(outputDirs);

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
      Iterable<PathFragment> dirsToCreate, Path dir, boolean strict) throws IOException {
    Set<Path> knownDirectories = new HashSet<>();
    // Add sandboxExecRoot and it's parent -- all paths must fall under the parent of
    // sandboxExecRoot and we know that sandboxExecRoot exists. This stops the recursion in
    // createDirectoryAndParentsInSandboxRoot.
    knownDirectories.add(dir);
    knownDirectories.add(dir.getParentDirectory());

    for (PathFragment path : dirsToCreate) {
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

  /** Wrapper class for the inputs of a sandbox. */
  public static final class SandboxInputs {

    private static final AtomicInteger tempFileUniquifierForVirtualInputWrites =
        new AtomicInteger();

    private final Map<PathFragment, Path> files;
    // Virtual inputs that are not materialized during {@link #processInputFiles}
    private final Set<VirtualActionInput> virtualInputsWithDelayedMaterialization;
    // Virtual inputs that are materialized during {@link #processInputFiles}.
    private final Map<VirtualActionInput, byte[]> materializedVirtualInputs;
    private final Map<PathFragment, String> symlinks;

    private static final SandboxInputs EMPTY_INPUTS =
        new SandboxInputs(
            ImmutableMap.of(), ImmutableSet.of(), ImmutableMap.of(), ImmutableMap.of());

    public SandboxInputs(
        Map<PathFragment, Path> files,
        Set<VirtualActionInput> virtualInputsWithDelayedMaterialization,
        Map<VirtualActionInput, byte[]> materializedVirtualInputs,
        Map<PathFragment, String> symlinks) {
      checkState(
          virtualInputsWithDelayedMaterialization.isEmpty() || materializedVirtualInputs.isEmpty(),
          "Either virtualInputsWithDelayedMaterialization or materializedVirtualInputs should be"
              + " empty.");
      this.files = files;
      this.virtualInputsWithDelayedMaterialization = virtualInputsWithDelayedMaterialization;
      this.materializedVirtualInputs = materializedVirtualInputs;
      this.symlinks = symlinks;
    }

    public SandboxInputs(
        Map<PathFragment, Path> files,
        Set<VirtualActionInput> virtualInputsWithDelayedMaterialization,
        Map<PathFragment, String> symlinks) {
      this(files, virtualInputsWithDelayedMaterialization, ImmutableMap.of(), symlinks);
    }

    public static SandboxInputs getEmptyInputs() {
      return EMPTY_INPUTS;
    }

    public Map<PathFragment, Path> getFiles() {
      return files;
    }

    public Map<PathFragment, String> getSymlinks() {
      return symlinks;
    }

    /**
     * Materializes a single virtual input inside the given execroot.
     *
     * <p>When materializing inputs under a new sandbox exec root, we can expect the input to not
     * exist, but we cannot make the same assumption for the non-sandboxed exec root therefore, we
     * may need to delete existing files.
     *
     * @param input virtual input to materialize
     * @param execroot path to the execroot under which to materialize the virtual input
     * @param isExecRootSandboxed whether the execroot is sandboxed.
     * @return digest of written virtual input
     * @throws IOException if the virtual input cannot be materialized
     */
    private static byte[] materializeVirtualInput(
        VirtualActionInput input, Path execroot, boolean isExecRootSandboxed) throws IOException {
      if (input instanceof EmptyActionInput) {
        // TODO(b/150963503): We can turn this into an unreachable code path when the old
        //  !delayVirtualInputMaterialization code path is deleted.
        return new byte[0];
      }

      Path outputPath = execroot.getRelative(input.getExecPath());
      if (isExecRootSandboxed) {
        return atomicallyWriteVirtualInput(
            input,
            outputPath,
            // When 2 actions try to atomically create the same virtual input, they need to have a
            // different suffix for the temporary file in order to avoid racy write to the same one.
            ".sandbox" + tempFileUniquifierForVirtualInputWrites.incrementAndGet());
      }

      if (outputPath.exists()) {
        outputPath.delete();
      }
      outputPath.getParentDirectory().createDirectoryAndParents();
      return writeVirtualInputTo(input, outputPath);
    }

    /**
     * Materializes virtual files inside the sandboxed execroot once it is known.
     *
     * <p>These are files that do not have to exist in the execroot: we can materialize them only
     * inside the sandbox, which means we can create them <i>before</i> we grab the output tree lock
     * (but assuming we do so inside the sandbox only).
     *
     * @param sandboxExecRoot the path to the <i>sandboxed</i> execroot
     * @return digests of written virtual inputs
     * @throws IOException if any virtual input cannot be materialized
     */
    public ImmutableMap<VirtualActionInput, byte[]> materializeVirtualInputs(Path sandboxExecRoot)
        throws IOException {
      if (!materializedVirtualInputs.isEmpty()) {
        return ImmutableMap.copyOf(materializedVirtualInputs);
      }

      ImmutableMap.Builder<VirtualActionInput, byte[]> digests =
          ImmutableMap.builderWithExpectedSize(virtualInputsWithDelayedMaterialization.size());
      for (VirtualActionInput input : virtualInputsWithDelayedMaterialization) {
        byte[] digest =
            materializeVirtualInput(input, sandboxExecRoot, /*isExecRootSandboxed=*/ false);
        digests.put(input, digest);
      }
      return digests.buildOrThrow();
    }

    /**
     * Returns a new SandboxInputs instance with only the inputs/symlinks listed in {@code allowed}
     * included.
     */
    public SandboxInputs limitedCopy(Set<PathFragment> allowed) {
      return new SandboxInputs(
          Maps.filterKeys(files, allowed::contains),
          ImmutableSet.of(),
          ImmutableMap.of(),
          Maps.filterKeys(symlinks, allowed::contains));
    }

    @Override
    public String toString() {
      return "Files: "
          + files
          + "\nVirtualInputs: "
          + virtualInputsWithDelayedMaterialization
          + "\nSymlinks: "
          + symlinks;
    }
  }

  private static byte[] writeVirtualInputTo(VirtualActionInput input, Path target)
      throws IOException {
    byte[] digest;
    try (OutputStream out = target.getOutputStream();
        HashingOutputStream hashingOut =
            new HashingOutputStream(
                target.getFileSystem().getDigestFunction().getHashFunction(), out)) {
      input.writeTo(hashingOut);
      digest = hashingOut.hash().asBytes();
    }
    // Some of the virtual inputs can be executed, e.g. embedded tools. Setting executable flag for
    // other is fine since that is only more permissive. Please note that for action outputs (e.g.
    // file write, where the user can specify executable flag), we will have artifacts which do not
    // go through this code path.
    target.setExecutable(true);
    return digest;
  }

  /**
   * Returns the inputs of a Spawn as a map of PathFragments relative to an execRoot to paths in the
   * host filesystem where the input files can be found.
   *
   * <p>This does not (and must not) write any {@link VirtualActionInput}s found because we do not
   * yet know where they should be written to. We have a path to an {@code execRoot}, but this path
   * should be treated as read-only because we may not be holding its lock. The caller should use
   * {@link SandboxInputs#materializeVirtualInputs(Path)} to later write these inputs when it knows
   * where they should be written to.
   *
   * @throws IOException if processing symlinks fails
   */
  public SandboxInputs processInputFiles(Map<PathFragment, ActionInput> inputMap, Path execRoot)
      throws IOException {
    Map<PathFragment, Path> inputFiles = new TreeMap<>();
    Set<VirtualActionInput> virtualInputsWithDelayedMaterialization = new HashSet<>();
    Map<PathFragment, String> inputSymlinks = new TreeMap<>();
    Map<VirtualActionInput, byte[]> materializedVirtualInputs = new HashMap<>();

    for (Map.Entry<PathFragment, ActionInput> e : inputMap.entrySet()) {
      PathFragment pathFragment = e.getKey();
      ActionInput actionInput = e.getValue();

      // TODO(b/150963503): Make delayVirtualInputMaterialization the default and remove the
      // alternate code path.
      if (delayVirtualInputMaterialization) {
        if (actionInput instanceof VirtualActionInput) {
          if (actionInput instanceof EmptyActionInput) {
            inputFiles.put(pathFragment, null);
          } else {
            virtualInputsWithDelayedMaterialization.add((VirtualActionInput) actionInput);
          }
        } else if (actionInput.isSymlink()) {
          Path inputPath = execRoot.getRelative(actionInput.getExecPath());
          inputSymlinks.put(pathFragment, inputPath.readSymbolicLink());
        } else {
          Path inputPath = execRoot.getRelative(actionInput.getExecPath());
          inputFiles.put(pathFragment, inputPath);
        }
      } else {
        if (actionInput instanceof VirtualActionInput) {
          byte[] digest =
              SandboxInputs.materializeVirtualInput(
                  (VirtualActionInput) actionInput, execRoot, /* isExecRootSandboxed=*/ true);
          materializedVirtualInputs.put((VirtualActionInput) actionInput, digest);
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
    }
    return new SandboxInputs(
        inputFiles,
        virtualInputsWithDelayedMaterialization,
        materializedVirtualInputs,
        inputSymlinks);
  }

  /** The file and directory outputs of a sandboxed spawn. */
  @AutoValue
  public abstract static class SandboxOutputs {
    public abstract ImmutableSet<PathFragment> files();

    public abstract ImmutableSet<PathFragment> dirs();

    private static final SandboxOutputs EMPTY_OUTPUTS =
        SandboxOutputs.create(ImmutableSet.of(), ImmutableSet.of());

    public static SandboxOutputs create(
        ImmutableSet<PathFragment> files, ImmutableSet<PathFragment> dirs) {
      return new AutoValue_SandboxHelpers_SandboxOutputs(files, dirs);
    }

    public static SandboxOutputs getEmptyInstance() {
      return EMPTY_OUTPUTS;
    }
  }

  public SandboxOutputs getOutputs(Spawn spawn) {
    ImmutableSet.Builder<PathFragment> files = ImmutableSet.builder();
    ImmutableSet.Builder<PathFragment> dirs = ImmutableSet.builder();
    for (ActionInput output : spawn.getOutputFiles()) {
      PathFragment path = PathFragment.create(output.getExecPathString());
      if (output instanceof Artifact && ((Artifact) output).isTreeArtifact()) {
        dirs.add(path);
      } else {
        files.add(path);
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
}
