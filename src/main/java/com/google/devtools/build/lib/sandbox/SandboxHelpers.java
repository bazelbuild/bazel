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
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * Helper methods that are shared by the different sandboxing strategies.
 *
 * <p>All sandboxed strategies within a build should share the same instance of this object.
 */
public final class SandboxHelpers {
  /** Wrapper class for the inputs of a sandbox. */
  public static final class SandboxInputs {
    private final Map<PathFragment, Path> files;
    private final Map<PathFragment, PathFragment> symlinks;

    public SandboxInputs(Map<PathFragment, Path> files, Map<PathFragment, PathFragment> symlinks) {
      this.files = files;
      this.symlinks = symlinks;
    }

    public Map<PathFragment, Path> getFiles() {
      return files;
    }

    public Map<PathFragment, PathFragment> getSymlinks() {
      return symlinks;
    }
  }

  /**
   * Returns the inputs of a Spawn as a map of PathFragments relative to an execRoot to paths in the
   * host filesystem where the input files can be found.
   *
   * <p>Also writes any supported {@link VirtualActionInput}s found.
   *
   * @throws IOException If any files could not be written.
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
    Map<PathFragment, PathFragment> inputSymlinks = new TreeMap<>();

    for (Map.Entry<PathFragment, ActionInput> e : inputMap.entrySet()) {
      PathFragment pathFragment = e.getKey();
      ActionInput actionInput = e.getValue();
      if (actionInput instanceof VirtualActionInput) {
        if (actionInput instanceof ParamFileActionInput) {
          ParamFileActionInput paramFileInput = (ParamFileActionInput) actionInput;
          Path outputPath = execRoot.getRelative(paramFileInput.getExecPath());
          if (outputPath.exists()) {
            outputPath.delete();
          }

          outputPath.getParentDirectory().createDirectoryAndParents();
          try (OutputStream outputStream = outputPath.getOutputStream()) {
            paramFileInput.writeTo(outputStream);
          }
        } else {
          // TODO(ulfjack): Handle all virtual inputs, e.g., by writing them to a file.
          Preconditions.checkState(actionInput instanceof EmptyActionInput);
        }
      }

      if (actionInput.isSymlink()) {
        Path inputPath = execRoot.getRelative(actionInput.getExecPath());
        // TODO(lberki): This does I/O. This method already throws IOException, so I suppose that is
        // A-OK?
        inputSymlinks.put(pathFragment, inputPath.readSymbolicLink());
      } else {
        Path inputPath =
            actionInput instanceof EmptyActionInput
                ? null
                : execRoot.getRelative(actionInput.getExecPath());
        inputFiles.put(pathFragment, inputPath);
      }
    }
    return new SandboxInputs(inputFiles, inputSymlinks);
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
