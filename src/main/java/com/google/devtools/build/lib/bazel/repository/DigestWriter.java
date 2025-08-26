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

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.bazel.bzlmod.GsonTypeAdapterUtil;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.rules.repository.RepoRecordedInput;
import com.google.devtools.build.lib.rules.repository.RepoRecordedInput.NeverUpToDateRepoRecordedInput;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import java.io.IOException;
import java.util.Map;
import java.util.Optional;
import java.util.TreeMap;
import javax.annotation.Nullable;
import net.starlark.java.eval.StarlarkSemantics;

/** Handles writing and reading of repo marker files. */
class DigestWriter {

  // The marker file version is inject in the rule key digest so the rule key is always different
  // when we decide to update the format.
  private static final int MARKER_FILE_VERSION = 7;
  // Input value map to force repo invalidation upon an invalid marker file.
  private static final ImmutableMap<RepoRecordedInput, String> PARSE_FAILURE =
      ImmutableMap.of(NeverUpToDateRepoRecordedInput.PARSE_FAILURE, "");

  private final BlazeDirectories directories;
  final Path markerPath;
  private final String predeclaredInputHash;

  DigestWriter(
      BlazeDirectories directories,
      RepositoryName repositoryName,
      RepoDefinition repoDefinition,
      StarlarkSemantics starlarkSemantics) {
    this.directories = directories;
    predeclaredInputHash = computePredeclaredInputHash(repoDefinition, starlarkSemantics);
    markerPath = getMarkerPath(directories, repositoryName);
  }

  // Escape a value for the marker file
  @VisibleForTesting
  static String escape(String str) {
    return str == null ? "\\0" : str.replace("\\", "\\\\").replace("\n", "\\n").replace(" ", "\\s");
  }

  // Unescape a value from the marker file
  @Nullable
  @VisibleForTesting
  static String unescape(String str) {
    if (str.equals("\\0")) {
      return null; // \0 == null string
    }
    StringBuilder result = new StringBuilder();
    boolean escaped = false;
    for (int i = 0; i < str.length(); i++) {
      char c = str.charAt(i);
      if (escaped) {
        if (c == 'n') { // n means new line
          result.append("\n");
        } else if (c == 's') { // s means space
          result.append(" ");
        } else { // Any other escaped characters are just un-escaped
          result.append(c);
        }
        escaped = false;
      } else if (c == '\\') {
        escaped = true;
      } else {
        result.append(c);
      }
    }
    return result.toString();
  }

  void writeMarkerFile(Map<? extends RepoRecordedInput, String> recordedInputValues)
      throws RepositoryFunctionException {
    StringBuilder builder = new StringBuilder();
    builder.append(predeclaredInputHash).append("\n");
    for (Map.Entry<RepoRecordedInput, String> recordedInput :
        new TreeMap<RepoRecordedInput, String>(recordedInputValues).entrySet()) {
      String key = recordedInput.getKey().toString();
      String value = recordedInput.getValue();
      builder.append(escape(key)).append(" ").append(escape(value)).append("\n");
    }
    String content = builder.toString();
    try {
      FileSystemUtils.writeContent(markerPath, UTF_8, content);
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }
  }

  sealed interface RepoDirectoryState {
    record UpToDate() implements RepoDirectoryState {}

    record OutOfDate(String reason) implements RepoDirectoryState {}
  }

  RepoDirectoryState areRepositoryAndMarkerFileConsistent(Environment env)
      throws InterruptedException, RepositoryFunctionException {
    return areRepositoryAndMarkerFileConsistent(env, markerPath);
  }

  /**
   * Checks if the state of the repository in the file system is consistent with the rule in the
   * WORKSPACE file.
   *
   * <p>Returns null if a Skyframe status is needed.
   *
   * <p>We check the repository root for existence here, but we can't depend on the FileValue,
   * because it's possible that we eventually create that directory in which case the FileValue and
   * the state of the file system would be inconsistent.
   */
  @Nullable
  RepoDirectoryState areRepositoryAndMarkerFileConsistent(Environment env, Path markerPath)
      throws RepositoryFunctionException, InterruptedException {
    if (!markerPath.exists()) {
      return new RepoDirectoryState.OutOfDate("repo hasn't been fetched yet");
    }

    try {
      String content = FileSystemUtils.readContent(markerPath, UTF_8);
      Map<RepoRecordedInput, String> recordedInputValues =
          readMarkerFile(content, Preconditions.checkNotNull(predeclaredInputHash));
      Optional<String> outdatedReason =
          RepoRecordedInput.isAnyValueOutdated(env, directories, recordedInputValues);
      if (env.valuesMissing()) {
        return null;
      }
      if (outdatedReason.isPresent()) {
        return new RepoDirectoryState.OutOfDate(outdatedReason.get());
      }
      return new RepoDirectoryState.UpToDate();
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }
  }

  private static Map<RepoRecordedInput, String> readMarkerFile(
      String content, String predeclaredInputHash) {
    Iterable<String> lines = Splitter.on('\n').split(content);

    @Nullable Map<RepoRecordedInput, String> recordedInputValues = null;
    boolean firstLineVerified = false;
    for (String line : lines) {
      if (line.isEmpty()) {
        continue;
      }
      if (!firstLineVerified) {
        if (!line.equals(predeclaredInputHash)) {
          // Break early, need to reload anyway. This also detects marker file version changes
          // so that unknown formats are not parsed.
          return ImmutableMap.of(
              new NeverUpToDateRepoRecordedInput(
                  "Bazel version, flags, repo rule definition or attributes changed"),
              "");
        }
        firstLineVerified = true;
        recordedInputValues = new TreeMap<>();
      } else {
        int sChar = line.indexOf(' ');
        if (sChar > 0) {
          RepoRecordedInput input = RepoRecordedInput.parse(unescape(line.substring(0, sChar)));
          if (!input.equals(NeverUpToDateRepoRecordedInput.PARSE_FAILURE)) {
            recordedInputValues.put(input, unescape(line.substring(sChar + 1)));
            continue;
          }
        }
        // On parse failure, just forget everything else and mark the whole input out of date.
        return PARSE_FAILURE;
      }
    }
    if (!firstLineVerified) {
      return PARSE_FAILURE;
    }
    return Preconditions.checkNotNull(recordedInputValues);
  }

  static String computePredeclaredInputHash(
      RepoDefinition repoDefinition, StarlarkSemantics starlarkSemantics) {
    return new Fingerprint()
        .addInt(MARKER_FILE_VERSION)
        // TODO: Using the hashCode() method for StarlarkSemantics here is suboptimal as
        //   it doesn't include any default values.
        .addInt(starlarkSemantics.hashCode())
        .addString(repoDefinition.repoRule().id().bzlFileLabel().toString())
        .addString(repoDefinition.repoRule().id().ruleName())
        .addBytes(repoDefinition.repoRule().transitiveBzlDigest())
        .addString(repoDefinition.name())
        .addString(
            GsonTypeAdapterUtil.SINGLE_EXTENSION_USAGES_VALUE_GSON.toJson(
                repoDefinition.attrValues()))
        .hexDigestAndReset();
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
