// Copyright 2025 The Bazel Authors. All rights reserved.
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
//
package com.google.devtools.build.lib.bazel.repository;

import static java.nio.charset.StandardCharsets.ISO_8859_1;

import com.google.common.base.Preconditions;
import com.google.common.base.Splitter;
import com.google.common.collect.Collections2;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.bazel.bzlmod.GsonTypeAdapterUtil;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.rules.repository.RepoRecordedInput;
import com.google.devtools.build.lib.rules.repository.RepoRecordedInput.NeverUpToDateRepoRecordedInput;
import com.google.devtools.build.lib.skyframe.RepoEnvironmentFunction;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import java.io.IOException;
import java.util.List;
import java.util.Optional;
import javax.annotation.Nullable;
import net.starlark.java.eval.StarlarkSemantics;

/** Handles writing and reading of repo marker files. */
public class DigestWriter {

  // The marker file version is inject in the rule key digest so the rule key is always different
  // when we decide to update the format.
  private static final int MARKER_FILE_VERSION = 7;
  // Input value list to force repo invalidation upon an invalid marker file.
  private static final ImmutableList<RepoRecordedInput.WithValue> PARSE_FAILURE =
      ImmutableList.of(
          new RepoRecordedInput.WithValue(NeverUpToDateRepoRecordedInput.PARSE_FAILURE, ""));

  private final BlazeDirectories directories;
  final String predeclaredInputHash;
  final Path markerPath;

  private DigestWriter(
      BlazeDirectories directories, RepositoryName repositoryName, String predeclaredInputHash) {
    this.directories = directories;
    this.predeclaredInputHash = predeclaredInputHash;
    this.markerPath = getMarkerPath(directories, repositoryName);
  }

  /** Returns null if and only if a Skyframe restart is needed. */
  @Nullable
  public static DigestWriter create(
      Environment env,
      BlazeDirectories directories,
      RepositoryName repositoryName,
      RepoDefinition repoDefinition,
      StarlarkSemantics starlarkSemantics)
      throws InterruptedException {
    String predeclaredInputHash =
        computePredeclaredInputHash(env, repoDefinition, starlarkSemantics);
    if (predeclaredInputHash == null) {
      return null;
    }
    return new DigestWriter(directories, repositoryName, predeclaredInputHash);
  }

  void writeMarkerFile(List<RepoRecordedInput.WithValue> recordedInputValues)
      throws RepositoryFunctionException {
    StringBuilder builder = new StringBuilder();
    builder.append(predeclaredInputHash).append('\n');
    for (var recordedInputValue : recordedInputValues) {
      builder.append(recordedInputValue).append('\n');
    }
    String content = builder.toString();
    try {
      FileSystemUtils.writeContent(markerPath, ISO_8859_1, content);
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }
  }

  sealed interface RepoDirectoryState {
    record UpToDate() implements RepoDirectoryState {}

    record OutOfDate(String reason) implements RepoDirectoryState {}

    /**
     * Opaque state indicating that a Skyframe restart is needed and carrying state to be preserved
     * across it.
     */
    final class Indeterminate implements RepoDirectoryState {
      private final ImmutableList<ImmutableList<RepoRecordedInput.WithValue>> batches;

      private Indeterminate(ImmutableList<ImmutableList<RepoRecordedInput.WithValue>> batches) {
        this.batches = batches;
      }
    }
  }

  RepoDirectoryState areRepositoryAndMarkerFileConsistent(
      Environment env, @Nullable RepoDirectoryState.Indeterminate indeterminateState)
      throws InterruptedException, RepositoryFunctionException {
    return areRepositoryAndMarkerFileConsistent(env, markerPath, indeterminateState);
  }

  /**
   * Checks if the state of the repository in the file system is consistent with the rule in the
   * WORKSPACE file.
   *
   * <p>Returns {@link RepoDirectoryState.Indeterminate} if a Skyframe restart is needed.
   *
   * <p>We check the repository root for existence here, but we can't depend on the FileValue,
   * because it's possible that we eventually create that directory in which case the FileValue and
   * the state of the file system would be inconsistent.
   */
  RepoDirectoryState areRepositoryAndMarkerFileConsistent(
      Environment env,
      Path markerPath,
      @Nullable RepoDirectoryState.Indeterminate intermediateState)
      throws RepositoryFunctionException, InterruptedException {
    if (!markerPath.exists()) {
      return new RepoDirectoryState.OutOfDate("repo hasn't been fetched yet");
    }

    try {
      // Avoid reading the marker file repeatedly.
      if (intermediateState == null) {
        String content = FileSystemUtils.readContent(markerPath, ISO_8859_1);
        var recordedInputValues =
            readMarkerFile(content, Preconditions.checkNotNull(predeclaredInputHash));
        if (recordedInputValues.isEmpty()) {
          return new RepoDirectoryState.OutOfDate(
              "Bazel version, flags, repo rule definition or attributes changed");
        }
        // Check inputs in batches to prevent Skyframe cycles caused by outdated dependencies.
        intermediateState =
            new RepoDirectoryState.Indeterminate(
                RepoRecordedInput.WithValue.splitIntoBatches(recordedInputValues.get()));
      }
      for (var batch : intermediateState.batches) {
        RepoRecordedInput.prefetch(
            env, directories, Collections2.transform(batch, RepoRecordedInput.WithValue::input));
        if (env.valuesMissing()) {
          return intermediateState;
        }
        Optional<String> outdatedReason =
            RepoRecordedInput.isAnyValueOutdated(env, directories, batch);
        if (outdatedReason.isPresent()) {
          return new RepoDirectoryState.OutOfDate(outdatedReason.get());
        }
      }
      return new RepoDirectoryState.UpToDate();
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }
  }

  /**
   * Returns a list of recorded inputs with their values parsed from the given marker file if the
   * predeclared input hash matches, or {@code Optional.empty()} if the hash doesn't match or any
   * error occurs during parsing.
   */
  public static Optional<ImmutableList<RepoRecordedInput.WithValue>> readMarkerFile(
      String content, String predeclaredInputHash) {
    Iterable<String> lines = Splitter.on('\n').split(content);

    var recordedInputValues = ImmutableList.<RepoRecordedInput.WithValue>builder();
    boolean firstLineVerified = false;
    for (String line : lines) {
      if (line.isEmpty()) {
        continue;
      }
      if (!firstLineVerified) {
        if (!line.equals(predeclaredInputHash)) {
          // Break early, need to reload anyway. This also detects marker file version changes
          // so that unknown formats are not parsed.
          return Optional.empty();
        }
        firstLineVerified = true;
      } else {
        var inputAndValue = RepoRecordedInput.WithValue.parse(line);
        if (inputAndValue.isEmpty()) {
          // On parse failure, just forget everything else and mark the whole input out of date.
          return Optional.empty();
        }
        recordedInputValues.add(inputAndValue.get());
      }
    }
    if (!firstLineVerified) {
      return Optional.empty();
    }
    return Optional.of(recordedInputValues.build());
  }

  @Nullable
  static String computePredeclaredInputHash(
      Environment env, RepoDefinition repoDefinition, StarlarkSemantics starlarkSemantics)
      throws InterruptedException {
    var environ =
        RepoEnvironmentFunction.getEnvironmentView(env, repoDefinition.repoRule().environ());
    if (environ == null) {
      return null;
    }
    var environInputs = RepoRecordedInput.EnvVar.wrap(environ);
    var fp =
        new Fingerprint()
            .addInt(MARKER_FILE_VERSION)
            .addBytes(BuildLanguageOptions.stableFingerprint(starlarkSemantics))
            .addString(repoDefinition.repoRule().id().bzlFileLabel().toString())
            .addString(repoDefinition.repoRule().id().ruleName())
            .addBytes(repoDefinition.repoRule().transitiveBzlDigest())
            .addString(repoDefinition.name())
            .addString(
                GsonTypeAdapterUtil.SINGLE_EXTENSION_USAGES_VALUE_GSON.toJson(
                    repoDefinition.attrValues()));
    fp.addInt(environInputs.size());
    environInputs.forEach(
        (key, value) -> fp.addString(key.toString()).addNullableString(value.orElse(null)));
    fp.addInt(repoDefinition.repoRule().recordedRepoMappingEntries().cellSet().size());
    repoDefinition
        .repoRule()
        .recordedRepoMappingEntries()
        .cellSet()
        .forEach(
            entry -> {
              fp.addString(entry.getRowKey().getName());
              fp.addString(entry.getColumnKey());
              fp.addString(entry.getValue().getName());
            });
    return fp.hexDigestAndReset();
  }

  private static Path getMarkerPath(BlazeDirectories directories, RepositoryName repo) {
    return RepositoryUtils.getExternalRepositoryDirectory(directories)
        .getChild(repo.getMarkerFileName());
  }

  static void clearMarkerFile(BlazeDirectories directories, RepositoryName repo)
      throws RepositoryFunctionException {
    try {
      getMarkerPath(directories, repo).delete();
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }
  }
}
