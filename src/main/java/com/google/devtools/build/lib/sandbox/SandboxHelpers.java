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

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.cache.VirtualActionInput;
import com.google.devtools.build.lib.actions.cache.VirtualActionInput.EmptyActionInput;
import com.google.devtools.build.lib.analysis.test.TestConfiguration;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.FileSystemUtils.MoveResult;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.OptionsParsingResult;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
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
   * @throws IOException if we fail to write the virtual input file
   */
  // TODO(b/150963503): We are using atomic file system moves for synchronization... but Bazel
  // should not be able to reach this state. Which means we should probably be doing some other
  // form of synchronization in-process before touching the file system.
  public static void atomicallyWriteVirtualInput(
      VirtualActionInput input, Path outputPath, String uniqueSuffix) throws IOException {
    Path tmpPath = outputPath.getFileSystem().getPath(outputPath.getPathString() + uniqueSuffix);
    tmpPath.getParentDirectory().createDirectoryAndParents();
    try {
      writeVirtualInputTo(input, tmpPath);
      // We expect the following to replace the params file atomically in case we are using
      // the dynamic scheduler and we are racing the remote strategy writing this same file.
      tmpPath.renameTo(outputPath);
      tmpPath = null; // Avoid unnecessary deletion attempt.
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
      }
    }
  }

  /** Wrapper class for the inputs of a sandbox. */
  public static final class SandboxInputs {

    private static final AtomicInteger tempFileUniquifierForVirtualInputWrites =
        new AtomicInteger();

    private final Map<PathFragment, Path> files;
    private final Set<VirtualActionInput> virtualInputs;
    private final Map<PathFragment, PathFragment> symlinks;

    public SandboxInputs(
        Map<PathFragment, Path> files,
        Set<VirtualActionInput> virtualInputs,
        Map<PathFragment, PathFragment> symlinks) {
      this.files = files;
      this.virtualInputs = virtualInputs;
      this.symlinks = symlinks;
    }

    public Map<PathFragment, Path> getFiles() {
      return files;
    }

    public Map<PathFragment, PathFragment> getSymlinks() {
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
     * @throws IOException if the virtual input cannot be materialized
     */
    private static void materializeVirtualInput(
        VirtualActionInput input, Path execroot, boolean isExecRootSandboxed) throws IOException {
      if (input instanceof EmptyActionInput) {
        // TODO(b/150963503): We can turn this into an unreachable code path when the old
        //  !delayVirtualInputMaterialization code path is deleted.
        return;
      }

      Path outputPath = execroot.getRelative(input.getExecPath());
      if (isExecRootSandboxed) {
        atomicallyWriteVirtualInput(
            input,
            outputPath,
            // When 2 actions try to atomically create the same virtual input, they need to have a
            // different suffix for the temporary file in order to avoid racy write to the same one.
            ".sandbox" + tempFileUniquifierForVirtualInputWrites.incrementAndGet());
        return;
      }

      if (outputPath.exists()) {
        outputPath.delete();
      }
      outputPath.getParentDirectory().createDirectoryAndParents();
      writeVirtualInputTo(input, outputPath);
    }

    /**
     * Materializes virtual files inside the sandboxed execroot once it is known.
     *
     * <p>These are files that do not have to exist in the execroot: we can materialize them only
     * inside the sandbox, which means we can create them <i>before</i> we grab the output tree lock
     * (but assuming we do so inside the sandbox only).
     *
     * @param sandboxExecRoot the path to the <i>sandboxed</i> execroot
     * @throws IOException if any virtual input cannot be materialized
     */
    public void materializeVirtualInputs(Path sandboxExecRoot) throws IOException {
      for (VirtualActionInput input : virtualInputs) {
        materializeVirtualInput(input, sandboxExecRoot, /*isExecRootSandboxed=*/ false);
      }
    }
  }

  private static void writeVirtualInputTo(VirtualActionInput input, Path target)
      throws IOException {
    try (OutputStream out = target.getOutputStream()) {
      input.writeTo(out);
    }
    // Some of the virtual inputs can be executed, e.g. embedded tools. Setting executable flag for
    // other is fine since that is only more permissive. Please note that for action outputs (e.g.
    // file write, where the user can specify executable flag), we will have artifacts which do not
    // go through this code path.
    target.setExecutable(true);
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
  public SandboxInputs processInputFiles(
      Map<PathFragment, ActionInput> inputMap,
      Spawn spawn,
      ArtifactExpander artifactExpander,
      Path execRoot)
      throws IOException {
    // SpawnInputExpander#getInputMapping uses ArtifactExpander#expandArtifacts to expand
    // middlemen and tree artifacts, which expands empty tree artifacts to no entry. However,
    // actions that accept TreeArtifacts as inputs generally expect that the empty directory is
    // created. So we add those explicitly here.
    // TODO(ulfjack): Move this code to SpawnInputExpander.
    for (ActionInput input : spawn.getInputFiles().toList()) {
      if (input instanceof Artifact && ((Artifact) input).isTreeArtifact()) {
        List<Artifact> containedArtifacts = new ArrayList<>();
        artifactExpander.expand((Artifact) input, containedArtifacts);
        // Attempting to mount a non-empty directory results in ERR_DIRECTORY_NOT_EMPTY, so we
        // only mount empty TreeArtifacts as directories.
        if (containedArtifacts.isEmpty()) {
          inputMap.put(input.getExecPath(), input);
        }
      }
    }

    Map<PathFragment, Path> inputFiles = new TreeMap<>();
    Set<VirtualActionInput> virtualInputs = new HashSet<>();
    Map<PathFragment, PathFragment> inputSymlinks = new TreeMap<>();

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
            virtualInputs.add((VirtualActionInput) actionInput);
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
          SandboxInputs.materializeVirtualInput(
              (VirtualActionInput) actionInput, execRoot, /* isExecRootSandboxed=*/ true);
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
    return new SandboxInputs(inputFiles, virtualInputs, inputSymlinks);
  }

  /** The file and directory outputs of a sandboxed spawn. */
  @AutoValue
  public abstract static class SandboxOutputs {
    public abstract ImmutableSet<PathFragment> files();

    public abstract ImmutableSet<PathFragment> dirs();

    public static SandboxOutputs create(
        ImmutableSet<PathFragment> files, ImmutableSet<PathFragment> dirs) {
      return new AutoValue_SandboxHelpers_SandboxOutputs(files, dirs);
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
