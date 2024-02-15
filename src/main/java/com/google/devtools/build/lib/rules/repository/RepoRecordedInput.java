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

package com.google.devtools.build.lib.rules.repository;

import static com.google.common.collect.ImmutableSet.toImmutableSet;

import com.google.common.base.Preconditions;
import com.google.common.base.Splitter;
import com.google.common.io.BaseEncoding;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.cmdline.LabelValidator;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.skyframe.ActionEnvironmentFunction;
import com.google.devtools.build.lib.skyframe.ClientEnvironmentValue;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.skyframe.RepositoryMappingValue;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyKey;
import java.io.IOException;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import javax.annotation.Nullable;

/**
 * Represents a "recorded input" of a repo fetch. We define the "input" of a repo fetch as any
 * entity that could affect the output of the repo fetch (i.e. the repo contents). A "recorded
 * input" is thus any input we can record during the fetch and thus know about only after the fetch.
 * This contrasts with "predeclared inputs", which are known before fetching the repo, and
 * "undiscoverable inputs", which are used during the fetch but is not recorded or recordable.
 *
 * <p>Recorded inputs are of particular interest, since in order to determine whether a fetched repo
 * is still up-to-date, the identity of all recorded inputs need to be stored in addition to their
 * values. This contrasts with predeclared inputs; the whole set of predeclared inputs are known
 * before the fetch, so we can simply hash all predeclared input values.
 *
 * <p>Recorded inputs and their values are stored in <i>marker files</i> for repos. Each recorded
 * input is stored as a string, with a prefix denoting its type, followed by a colon, and then the
 * information identifying that specific input.
 */
public abstract class RepoRecordedInput implements Comparable<RepoRecordedInput> {
  /** Represents a parser for a specific type of recorded inputs. */
  public abstract static class Parser {
    /**
     * The prefix that identifies the type of the recorded inputs: for example, the {@code ENV} part
     * of {@code ENV:MY_ENV_VAR}.
     */
    public abstract String getPrefix();

    /**
     * Parses a recorded input from the post-colon substring that identifies the specific input: for
     * example, the {@code MY_ENV_VAR} part of {@code ENV:MY_ENV_VAR}.
     */
    public abstract RepoRecordedInput parse(String s);
  }

  private static final Comparator<RepoRecordedInput> COMPARATOR =
      Comparator.comparing((RepoRecordedInput rri) -> rri.getParser().getPrefix())
          .thenComparing(RepoRecordedInput::toStringInternal);

  /**
   * Parses a recorded input from its string representation.
   *
   * @param s the string representation
   * @return The parsed recorded input object, or {@code null} if the string representation is
   *     invalid
   */
  @Nullable
  public static RepoRecordedInput parse(String s) {
    List<String> parts = Splitter.on(':').limit(2).splitToList(s);
    for (Parser parser : new Parser[] {File.PARSER, EnvVar.PARSER, RecordedRepoMapping.PARSER}) {
      if (parts.get(0).equals(parser.getPrefix())) {
        return parser.parse(parts.get(1));
      }
    }
    return null;
  }

  /**
   * Returns whether all values are still up-to-date for each recorded input. If Skyframe values are
   * missing, the return value should be ignored; callers are responsible for checking {@code
   * env.valuesMissing()} and triggering a Skyframe restart if needed.
   */
  public static boolean areAllValuesUpToDate(
      Environment env,
      BlazeDirectories directories,
      Map<? extends RepoRecordedInput, String> recordedInputValues)
      throws InterruptedException {
    env.getValuesAndExceptions(
        recordedInputValues.keySet().stream()
            .map(rri -> rri.getSkyKey(directories))
            .filter(Objects::nonNull)
            .collect(toImmutableSet()));
    if (env.valuesMissing()) {
      return false;
    }
    for (Map.Entry<? extends RepoRecordedInput, String> recordedInputValue :
        recordedInputValues.entrySet()) {
      if (!recordedInputValue
          .getKey()
          .isUpToDate(env, directories, recordedInputValue.getValue())) {
        return false;
      }
    }
    return true;
  }

  @Override
  public abstract boolean equals(Object obj);

  @Override
  public abstract int hashCode();

  @Override
  public final String toString() {
    return getParser().getPrefix() + ":" + toStringInternal();
  }

  @Override
  public int compareTo(RepoRecordedInput o) {
    return COMPARATOR.compare(this, o);
  }

  /**
   * Returns the post-colon substring that identifies the specific input: for example, the {@code
   * MY_ENV_VAR} part of {@code ENV:MY_ENV_VAR}.
   */
  public abstract String toStringInternal();

  /** Returns the parser object for this type of recorded inputs. */
  public abstract Parser getParser();

  /**
   * Returns the {@link SkyKey} that is necessary to determine {@link #isUpToDate}. Can be null if
   * no SkyKey is needed.
   */
  @Nullable
  public abstract SkyKey getSkyKey(BlazeDirectories directories);

  /**
   * Returns whether the given {@code oldValue} is still up-to-date for this recorded input. This
   * method can assume that {@link #getSkyKey(BlazeDirectories)} is already evaluated; it can
   * request further Skyframe evaluations, and if any values are missing, this method can return any
   * value (doesn't matter what) and will be reinvoked after a Skyframe restart.
   */
  public abstract boolean isUpToDate(
      Environment env, BlazeDirectories directories, @Nullable String oldValue)
      throws InterruptedException;

  /** Represents a file input accessed during the repo fetch. */
  public abstract static class File extends RepoRecordedInput {
    public static final Parser PARSER =
        new Parser() {
          @Override
          public String getPrefix() {
            return "FILE";
          }

          @Override
          public RepoRecordedInput parse(String s) {
            if (LabelValidator.isAbsolute(s)) {
              int doubleSlash = s.indexOf("//");
              int skipAts = s.startsWith("@@") ? 2 : s.startsWith("@") ? 1 : 0;
              return new FileInsideWorkspace(
                  RepositoryName.createUnvalidated(s.substring(skipAts, doubleSlash)),
                  // For backwards compatibility, treat colons as slashes.
                  PathFragment.create(s.substring(doubleSlash + 2).replace(':', '/')));
            }
            return new FileOutsideWorkspace(PathFragment.create(s));
          }
        };

    @Override
    public Parser getParser() {
      return PARSER;
    }

    /**
     * Convert to a {@link com.google.devtools.build.lib.actions.FileValue} to a String appropriate
     * for placing in a repository marker file. The file need not exist, and can be a file or a
     * directory.
     */
    public static String fileValueToMarkerValue(FileValue fileValue) throws IOException {
      if (fileValue.isDirectory()) {
        return "DIR";
      }
      if (!fileValue.exists()) {
        return "ENOENT";
      }
      // Return the file content digest in hex. fileValue may or may not have the digest available.
      byte[] digest = fileValue.realFileStateValue().getDigest();
      if (digest == null) {
        // Fast digest not available, or it would have been in the FileValue.
        digest = fileValue.realRootedPath().asPath().getDigest();
      }
      return BaseEncoding.base16().lowerCase().encode(digest);
    }
  }

  /**
   * Represents a file input accessed during the repo fetch that is within the current Bazel
   * workspace.
   *
   * <p>This is <em>almost</em> like being addressable by a label, but includes the extra corner
   * case of files inside a repo but not within any package due to missing BUILD files. For example,
   * the file {@code @@foo//:abc.bzl} is addressable by a label if the file {@code @@foo//:BUILD}
   * exists. But if the BUILD file doesn't exist, the {@code abc.bzl} file should still be
   * "watchable"; it's just that {@code @@foo//:abc.bzl} is technically not a valid label.
   */
  public static final class FileInsideWorkspace extends File {
    private final RepositoryName repoName;
    private final PathFragment pathFragment;

    public FileInsideWorkspace(RepositoryName repoName, PathFragment pathFragment) {
      this.repoName = repoName;
      this.pathFragment = pathFragment;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (!(o instanceof FileInsideWorkspace)) {
        return false;
      }
      FileInsideWorkspace that = (FileInsideWorkspace) o;
      return Objects.equals(repoName, that.repoName)
          && Objects.equals(pathFragment, that.pathFragment);
    }

    @Override
    public int hashCode() {
      return Objects.hash(repoName, pathFragment);
    }

    @Override
    public String toStringInternal() {
      // We store `@@foo//abc/def:ghi.bzl` as just `@@foo//abc/def/ghi.bzl`. See class javadoc for
      // more context.
      return repoName + "//" + pathFragment;
    }

    @Override
    @Nullable
    public SkyKey getSkyKey(BlazeDirectories directories) {
      return repoName.isMain() ? null : RepositoryDirectoryValue.key(repoName);
    }

    @Override
    public boolean isUpToDate(
        Environment env, BlazeDirectories directories, @Nullable String oldValue)
        throws InterruptedException {
      Root root;
      if (repoName.isMain()) {
        root = Root.fromPath(directories.getWorkspace());
      } else {
        RepositoryDirectoryValue repoDirValue =
            (RepositoryDirectoryValue) env.getValue(getSkyKey(directories));
        if (repoDirValue == null || !repoDirValue.repositoryExists()) {
          return false;
        }
        root = Root.fromPath(repoDirValue.getPath());
      }
      RootedPath rootedPath = RootedPath.toRootedPath(root, pathFragment);
      SkyKey fileKey = FileValue.key(rootedPath);
      try {
        FileValue fileValue = (FileValue) env.getValueOrThrow(fileKey, IOException.class);
        if (fileValue == null || !fileValue.isFile() || fileValue.isSpecialFile()) {
          return false;
        }
        return oldValue.equals(fileValueToMarkerValue(fileValue));
      } catch (IOException e) {
        return false;
      }
    }
  }

  /**
   * Represents a file input accessed during the repo fetch that is outside the current Bazel
   * workspace. This file is addressed by its absolute path.
   */
  public static final class FileOutsideWorkspace extends File {
    private final PathFragment path;

    public FileOutsideWorkspace(PathFragment path) {
      Preconditions.checkArgument(path.isAbsolute());
      this.path = path;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (!(o instanceof FileOutsideWorkspace)) {
        return false;
      }
      FileOutsideWorkspace that = (FileOutsideWorkspace) o;
      return Objects.equals(path, that.path);
    }

    @Override
    public int hashCode() {
      return path.hashCode();
    }

    @Override
    public String toStringInternal() {
      return path.getPathString();
    }

    @Override
    public SkyKey getSkyKey(BlazeDirectories directories) {
      return FileValue.key(
          RootedPath.toRootedPath(
              Root.absoluteRoot(directories.getOutputBase().getFileSystem()), path));
    }

    @Override
    public boolean isUpToDate(
        Environment env, BlazeDirectories directories, @Nullable String oldValue)
        throws InterruptedException {
      try {
        FileValue fileValue =
            (FileValue) env.getValueOrThrow(getSkyKey(directories), IOException.class);
        if (fileValue == null || !fileValue.isFile() || fileValue.isSpecialFile()) {
          return false;
        }
        return oldValue.equals(fileValueToMarkerValue(fileValue));
      } catch (IOException e) {
        return false;
      }
    }
  }

  /** Represents an environment variable accessed during the repo fetch. */
  public static final class EnvVar extends RepoRecordedInput {
    static final Parser PARSER =
        new Parser() {
          @Override
          public String getPrefix() {
            return "ENV";
          }

          @Override
          public RepoRecordedInput parse(String s) {
            return new EnvVar(s);
          }
        };

    final String name;

    public EnvVar(String name) {
      this.name = name;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (!(o instanceof EnvVar)) {
        return false;
      }
      EnvVar envVar = (EnvVar) o;
      return Objects.equals(name, envVar.name);
    }

    @Override
    public int hashCode() {
      return name.hashCode();
    }

    @Override
    public Parser getParser() {
      return PARSER;
    }

    @Override
    public String toStringInternal() {
      return name;
    }

    @Override
    public SkyKey getSkyKey(BlazeDirectories directories) {
      return ActionEnvironmentFunction.key(name);
    }

    @Override
    public boolean isUpToDate(
        Environment env, BlazeDirectories directories, @Nullable String oldValue)
        throws InterruptedException {
      String v = PrecomputedValue.REPO_ENV.get(env).get(name);
      if (v == null) {
        v = ((ClientEnvironmentValue) env.getValue(getSkyKey(directories))).getValue();
      }
      // Note that `oldValue` can be null if the env var was not set.
      return Objects.equals(oldValue, v);
    }
  }

  /** Represents a repo mapping entry that was used during the repo fetch. */
  public static final class RecordedRepoMapping extends RepoRecordedInput {
    static final Parser PARSER =
        new Parser() {
          @Override
          public String getPrefix() {
            return "REPO_MAPPING";
          }

          @Override
          public RepoRecordedInput parse(String s) {
            List<String> parts = Splitter.on(',').limit(2).splitToList(s);
            return new RecordedRepoMapping(
                RepositoryName.createUnvalidated(parts.get(0)), parts.get(1));
          }
        };

    final RepositoryName sourceRepo;
    final String apparentName;

    public RecordedRepoMapping(RepositoryName sourceRepo, String apparentName) {
      this.sourceRepo = sourceRepo;
      this.apparentName = apparentName;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (!(o instanceof RecordedRepoMapping)) {
        return false;
      }
      RecordedRepoMapping that = (RecordedRepoMapping) o;
      return Objects.equals(sourceRepo, that.sourceRepo)
          && Objects.equals(apparentName, that.apparentName);
    }

    @Override
    public int hashCode() {
      return Objects.hash(sourceRepo, apparentName);
    }

    @Override
    public Parser getParser() {
      return PARSER;
    }

    @Override
    public String toStringInternal() {
      return sourceRepo.getName() + ',' + apparentName;
    }

    @Override
    public SkyKey getSkyKey(BlazeDirectories directories) {
      return RepositoryMappingValue.key(sourceRepo);
    }

    @Override
    public boolean isUpToDate(
        Environment env, BlazeDirectories directories, @Nullable String oldValue)
        throws InterruptedException {
      RepositoryMappingValue repoMappingValue =
          (RepositoryMappingValue) env.getValue(getSkyKey(directories));
      return repoMappingValue != RepositoryMappingValue.NOT_FOUND_VALUE
          && RepositoryName.createUnvalidated(oldValue)
              .equals(repoMappingValue.getRepositoryMapping().get(apparentName));
    }
  }
}
