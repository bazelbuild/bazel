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
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.CommandLines.ParamFileActionInput;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.cache.VirtualActionInput;
import com.google.devtools.build.lib.actions.cache.VirtualActionInput.EmptyActionInput;
import com.google.devtools.build.lib.analysis.test.TestConfiguration;
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

/**
 * Helper methods that are shared by the different sandboxing strategies.
 *
 * <p>All sandboxed strategies within a build should share the same instance of this object.
 */
public final class SandboxHelpers {

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

  /** Wrapper class for the inputs of a sandbox. */
  public static final class SandboxInputs {
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
     * @param input virtual input to materialize
     * @param execroot path to the execroot under which to materialize the virtual input
     * @param needsDelete whether to attempt to delete a previous instance of this virtual input.
     *     When materializing under a new sandbox execroot, we can expect the input to not exist,
     *     but we cannot make the same assumption for the non-sandboxed execroot.
     * @throws IOException if the virtual input cannot be materialized
     */
    private static void materializeVirtualInput(
        VirtualActionInput input, Path execroot, boolean needsDelete) throws IOException {
      if (input instanceof ParamFileActionInput) {
        ParamFileActionInput paramFileInput = (ParamFileActionInput) input;
        Path outputPath = execroot.getRelative(paramFileInput.getExecPath());
        if (needsDelete && outputPath.exists()) {
          outputPath.delete();
        }

        outputPath.getParentDirectory().createDirectoryAndParents();
        try (OutputStream outputStream = outputPath.getOutputStream()) {
          paramFileInput.writeTo(outputStream);
        }
      } else {
        // TODO(b/150963503): We can turn this into an unreachable code path when the old
        // !delayVirtualInputMaterialization code path is deleted.
        // TODO(ulfjack): Handle all virtual inputs, e.g., by writing them to a file.
        Preconditions.checkState(input instanceof EmptyActionInput);
      }
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
        materializeVirtualInput(input, sandboxExecRoot, /*needsDelete=*/ false);
      }
    }
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
              (VirtualActionInput) actionInput, execRoot, /*needsDelete=*/ true);
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
