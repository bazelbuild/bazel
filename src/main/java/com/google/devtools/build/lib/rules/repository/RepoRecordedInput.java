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

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.collect.ImmutableSet.toImmutableSet;

import com.google.auto.value.AutoValue;
import com.google.common.base.Preconditions;
import com.google.common.base.Splitter;
import com.google.common.io.BaseEncoding;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.cmdline.LabelValidator;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.skyframe.ActionEnvironmentFunction;
import com.google.devtools.build.lib.skyframe.ClientEnvironmentValue;
import com.google.devtools.build.lib.skyframe.DirectoryListingValue;
import com.google.devtools.build.lib.skyframe.DirectoryTreeDigestValue;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.skyframe.RepositoryMappingValue;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.Path;
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
import java.util.Optional;
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
    for (Parser parser :
        new Parser[] {
          File.PARSER, Dirents.PARSER, DirTree.PARSER, EnvVar.PARSER, RecordedRepoMapping.PARSER
        }) {
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

  /** Returns the {@link SkyKey} that is necessary to determine {@link #isUpToDate}. */
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

  /**
   * Represents a filesystem path stored in a way that is repo-cache-friendly. That is, if the path
   * happens to point inside the current Bazel workspace (in either the main repo or an external
   * repo), we store the appropriate repo name and the path fragment relative to the repo root,
   * instead of the entire absolute path.
   *
   * <p>This is <em>almost</em> like storing a label, but includes the extra corner case of files
   * inside a repo but not within any package due to missing BUILD files. For example, the file
   * {@code @@foo//:abc.bzl} is addressable by a label if the file {@code @@foo//:BUILD} exists. But
   * if the BUILD file doesn't exist, the {@code abc.bzl} file should still be "watchable"; it's
   * just that {@code @@foo//:abc.bzl} is technically not a valid label.
   *
   * <p>Of course, when the path is outside the current Bazel workspace, we just store the absolute
   * path.
   */
  @AutoValue
  public abstract static class RepoCacheFriendlyPath {
    public abstract Optional<RepositoryName> repoName();

    public abstract PathFragment path();

    public static RepoCacheFriendlyPath createInsideWorkspace(
        RepositoryName repoName, PathFragment path) {
      Preconditions.checkArgument(
          !path.isAbsolute(), "the provided path should be relative to the repo root");
      return new AutoValue_RepoRecordedInput_RepoCacheFriendlyPath(Optional.of(repoName), path);
    }

    public static RepoCacheFriendlyPath createOutsideWorkspace(PathFragment path) {
      Preconditions.checkArgument(
          path.isAbsolute(), "the provided path should be absolute in the filesystem");
      return new AutoValue_RepoRecordedInput_RepoCacheFriendlyPath(Optional.empty(), path);
    }

    @Override
    public final String toString() {
      // We store `@@foo//abc/def:ghi.bzl` as just `@@foo//abc/def/ghi.bzl`. See class javadoc for
      // more context.
      return repoName().map(repoName -> repoName + "//" + path()).orElse(path().toString());
    }

    public static RepoCacheFriendlyPath parse(String s) {
      if (LabelValidator.isAbsolute(s)) {
        int doubleSlash = s.indexOf("//");
        int skipAts = s.startsWith("@@") ? 2 : s.startsWith("@") ? 1 : 0;
        return createInsideWorkspace(
            RepositoryName.createUnvalidated(s.substring(skipAts, doubleSlash)),
            PathFragment.create(s.substring(doubleSlash + 2)));
      }
      return createOutsideWorkspace(PathFragment.create(s));
    }

    /** Returns the rooted path corresponding to this "repo-friendly path". */
    public final RootedPath getRootedPath(BlazeDirectories directories) {
      Root root;
      if (repoName().isEmpty()) {
        root = Root.absoluteRoot(directories.getOutputBase().getFileSystem());
      } else if (repoName().get().isMain()) {
        root = Root.fromPath(directories.getWorkspace());
      } else {
        // This path is from an external repo. We just directly fabricate the path here instead of
        // requesting the appropriate RepositoryDirectoryValue, since we can rely on the various
        // other SkyFunctions (such as FileStateFunction and DirectoryListingStateFunction) to do
        // that for us instead. This also sidesteps an awkward situation when the external repo in
        // question is not defined.
        root =
            Root.fromPath(
                directories
                    .getOutputBase()
                    .getRelative(LabelConstants.EXTERNAL_REPOSITORY_LOCATION)
                    .getRelative(repoName().get().getName()));
      }
      return RootedPath.toRootedPath(root, path());
    }
  }

  /**
   * Represents a file input accessed during the repo fetch. Despite being named just "file", this
   * can represent a file or a directory on the filesystem, and it does not need to exist. The value
   * of the input contains whether this is a file or a directory or nonexistent, and if it's a file,
   * the digest of its contents.
   */
  public static final class File extends RepoRecordedInput {
    public static final Parser PARSER =
        new Parser() {
          @Override
          public String getPrefix() {
            return "FILE";
          }

          @Override
          public RepoRecordedInput parse(String s) {
            return new File(RepoCacheFriendlyPath.parse(s));
          }
        };

    private final RepoCacheFriendlyPath path;

    public File(RepoCacheFriendlyPath path) {
      this.path = path;
    }

    @Override
    public Parser getParser() {
      return PARSER;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (!(o instanceof File)) {
        return false;
      }
      File that = (File) o;
      return Objects.equals(path, that.path);
    }

    @Override
    public int hashCode() {
      return path.hashCode();
    }

    @Override
    public String toStringInternal() {
      return path.toString();
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

    @Nullable
    public SkyKey getSkyKey(BlazeDirectories directories) {
      return FileValue.key(path.getRootedPath(directories));
    }

    @Override
    public boolean isUpToDate(
        Environment env, BlazeDirectories directories, @Nullable String oldValue)
        throws InterruptedException {
      try {
        FileValue fileValue =
            (FileValue) env.getValueOrThrow(getSkyKey(directories), IOException.class);
        if (fileValue == null) {
          return false;
        }
        return oldValue.equals(fileValueToMarkerValue(fileValue));
      } catch (IOException e) {
        return false;
      }
    }
  }

  /** Represents the list of entries under a directory accessed during the fetch. */
  public static final class Dirents extends RepoRecordedInput {
    public static final Parser PARSER =
        new Parser() {
          @Override
          public String getPrefix() {
            return "DIRENTS";
          }

          @Override
          public RepoRecordedInput parse(String s) {
            return new Dirents(RepoCacheFriendlyPath.parse(s));
          }
        };

    private final RepoCacheFriendlyPath path;

    public Dirents(RepoCacheFriendlyPath path) {
      this.path = path;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (!(o instanceof Dirents)) {
        return false;
      }
      Dirents that = (Dirents) o;
      return Objects.equals(path, that.path);
    }

    @Override
    public int hashCode() {
      return path.hashCode();
    }

    @Override
    public String toStringInternal() {
      return path.toString();
    }

    @Override
    public Parser getParser() {
      return PARSER;
    }

    @Override
    public SkyKey getSkyKey(BlazeDirectories directories) {
      return DirectoryListingValue.key(path.getRootedPath(directories));
    }

    @Override
    public boolean isUpToDate(
        Environment env, BlazeDirectories directories, @Nullable String oldValue)
        throws InterruptedException {
      SkyKey skyKey = getSkyKey(directories);
      if (env.getValue(skyKey) == null) {
        return false;
      }
      try {
        return oldValue.equals(
            getDirentsMarkerValue(((DirectoryListingValue.Key) skyKey).argument().asPath()));
      } catch (IOException e) {
        return false;
      }
    }

    public static String getDirentsMarkerValue(Path path) throws IOException {
      Fingerprint fp = new Fingerprint();
      fp.addStrings(
          path.getDirectoryEntries().stream()
              .map(Path::getBaseName)
              .sorted()
              .collect(toImmutableList()));
      return fp.hexDigestAndReset();
    }
  }

  /**
   * Represents an entire directory tree accessed during the fetch. Anything under the tree changing
   * (including adding/removing/renaming files or directories and changing file contents) will cause
   * it to go out of date.
   */
  public static final class DirTree extends RepoRecordedInput {
    public static final Parser PARSER =
        new Parser() {
          @Override
          public String getPrefix() {
            return "DIRTREE";
          }

          @Override
          public RepoRecordedInput parse(String s) {
            return new DirTree(RepoCacheFriendlyPath.parse(s));
          }
        };

    private final RepoCacheFriendlyPath path;

    public DirTree(RepoCacheFriendlyPath path) {
      this.path = path;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (!(o instanceof DirTree)) {
        return false;
      }
      DirTree that = (DirTree) o;
      return Objects.equals(path, that.path);
    }

    @Override
    public int hashCode() {
      return path.hashCode();
    }

    @Override
    public String toStringInternal() {
      return path.toString();
    }

    @Override
    public Parser getParser() {
      return PARSER;
    }

    @Override
    public SkyKey getSkyKey(BlazeDirectories directories) {
      return DirectoryTreeDigestValue.key(path.getRootedPath(directories));
    }

    @Override
    public boolean isUpToDate(
        Environment env, BlazeDirectories directories, @Nullable String oldValue)
        throws InterruptedException {
      DirectoryTreeDigestValue value =
          (DirectoryTreeDigestValue) env.getValue(getSkyKey(directories));
      if (value == null) {
        return false;
      }
      return oldValue.equals(value.hexDigest());
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
      // Since we only record repo mapping entries for repos defined in Bzlmod, we can request the
      // WORKSPACE-less version of the main repo mapping (as no repos defined in Bzlmod can see
      // stuff from WORKSPACE).
      return sourceRepo.isMain()
          ? RepositoryMappingValue.KEY_FOR_ROOT_MODULE_WITHOUT_WORKSPACE_REPOS
          : RepositoryMappingValue.key(sourceRepo);
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
