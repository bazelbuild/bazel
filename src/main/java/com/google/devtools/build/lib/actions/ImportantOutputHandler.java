// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.actions;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.skyframe.DetailedException;
import com.google.devtools.build.lib.skyframe.rewinding.LostInputOwners;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.time.Duration;
import java.util.Collection;
import java.util.Map;
import java.util.Optional;

/** Context to be informed of top-level outputs and their runfiles. */
public interface ImportantOutputHandler extends ActionContext {

  /**
   * A threshold to pass to {@link com.google.devtools.build.lib.profiler.GoogleAutoProfilerUtils}
   * for profiling {@link ImportantOutputHandler} operations.
   */
  Duration LOG_THRESHOLD = Duration.ofMillis(100);

  /**
   * Informs this handler that top-level outputs have been built.
   *
   * <p>The handler may verify that remotely stored outputs are still available. Returns a map from
   * digest to output for any artifacts that need to be regenerated via action rewinding.
   *
   * @param importantOutputs top-level outputs, excluding {@linkplain
   *     com.google.devtools.build.lib.analysis.OutputGroupInfo#HIDDEN_OUTPUT_GROUP_PREFIX hidden
   *     output groups}
   * @param importantMetadataProvider provides metadata for artifacts in {@code importantOutputs}
   *     and their expansions
   * @param fullMetadataProvider like {@code importantMetadataProvider}, but additionally provides
   *     metadata for artifacts in {@linkplain
   *     com.google.devtools.build.lib.analysis.OutputGroupInfo#HIDDEN_OUTPUT_GROUP_PREFIX hidden
   *     output groups} and their expansions
   * @return any artifacts that need to be regenerated via action rewinding
   * @throws ImportantOutputException for an issue processing the outputs, not including lost
   *     outputs which are reported in the returned {@link LostArtifacts}
   */
  // TODO: jhorvitz - Find a cleaner way than passing two InputMetadataProviders.
  LostArtifacts processOutputsAndGetLostArtifacts(
      Iterable<Artifact> importantOutputs,
      InputMetadataProvider importantMetadataProvider,
      InputMetadataProvider fullMetadataProvider)
      throws ImportantOutputException, InterruptedException;

  /**
   * Informs this handler that the runfiles of a top-level target have been built.
   *
   * <p>The handler may verify that remotely stored outputs are still available. Returns a map from
   * digest to output for any artifacts that need to be regenerated via action rewinding.
   *
   * @param runfilesDir exec path of the runfiles directory
   * @param runfiles mapping from {@code runfilesDir}-relative path to target artifact; values may
   *     be {@code null} to represent an empty file (can happen with {@code __init__.py} files, see
   *     {@link com.google.devtools.build.lib.rules.python.PythonUtils.GetInitPyFiles})
   * @param metadataProvider provides metadata for artifacts in {@code runfiles} and their
   *     expansions
   * @param inputManifestExtension the file extension of the input manifest
   * @return any artifacts that need to be regenerated via action rewinding
   * @throws ImportantOutputException for an issue processing the runfiles, not including lost
   *     outputs which are reported in the returned {@link LostArtifacts}
   */
  LostArtifacts processRunfilesAndGetLostArtifacts(
      PathFragment runfilesDir,
      Map<PathFragment, Artifact> runfiles,
      InputMetadataProvider metadataProvider,
      String inputManifestExtension)
      throws ImportantOutputException, InterruptedException;

  /**
   * Informs this handler of outputs from a completed test attempt.
   *
   * <p>The given paths are under the exec root and are backed by an {@link
   * com.google.devtools.build.lib.vfs.OutputService#createActionFileSystem action filesystem} if
   * applicable.
   *
   * <p>Test outputs should never be lost. Test actions are not shareable across servers (see {@link
   * Actions#dependsOnBuildId}), so outputs passed to this method come from a just-executed test
   * action.
   */
  void processTestOutputs(Collection<Path> testOutputs)
      throws ImportantOutputException, InterruptedException;

  /**
   * Informs this handler of outputs from {@link
   * com.google.devtools.build.lib.analysis.WorkspaceStatusAction}.
   *
   * <p>The given paths are under the exec root and are backed by an {@link
   * com.google.devtools.build.lib.vfs.OutputService#createActionFileSystem action filesystem} if
   * applicable.
   *
   * <p>Workspace status outputs should never be lost. {@link
   * com.google.devtools.build.lib.analysis.WorkspaceStatusAction} is not shareable across servers
   * (see {@link Actions#dependsOnBuildId}), so outputs passed to this method come from a
   * just-executed action.
   */
  void processWorkspaceStatusOutputs(Path stableOutput, Path volatileOutput)
      throws ImportantOutputException, InterruptedException;

  /**
   * Represents artifacts that need to be regenerated via action rewinding, optionally along with
   * their owners if known. If {@code owners} is present, the ownership information must be
   * complete.
   */
  record LostArtifacts(
      ImmutableMap<String, ActionInput> byDigest, Optional<LostInputOwners> owners) {

    /** An empty instance of {@link LostArtifacts}. */
    public static final LostArtifacts EMPTY =
        new LostArtifacts(ImmutableMap.of(), Optional.of(new LostInputOwners()));

    public LostArtifacts {
      checkNotNull(byDigest);
      checkNotNull(owners);
    }

    public boolean isEmpty() {
      return byDigest.isEmpty();
    }

    /** Throws {@link LostInputsExecException} if this instance is not empty. */
    public void throwIfNotEmpty() throws LostInputsExecException {
      if (!isEmpty()) {
        throw new LostInputsExecException(byDigest, owners, /* cause= */ null);
      }
    }
  }

  /** Represents an exception encountered during processing of important outputs. */
  final class ImportantOutputException extends Exception implements DetailedException {
    private final FailureDetail failureDetail;

    public ImportantOutputException(Throwable cause, FailureDetail failureDetail) {
      super(failureDetail.getMessage(), cause);
      this.failureDetail = failureDetail;
    }

    public FailureDetail getFailureDetail() {
      return failureDetail;
    }

    @Override
    public DetailedExitCode getDetailedExitCode() {
      return DetailedExitCode.of(failureDetail);
    }
  }
}
