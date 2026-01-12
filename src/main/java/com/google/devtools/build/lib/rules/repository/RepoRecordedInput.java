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
import static com.google.common.collect.ImmutableMap.toImmutableMap;
import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static java.util.Map.Entry.comparingByKey;
import static java.util.Objects.requireNonNull;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.base.Splitter;
import com.google.common.collect.Collections2;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.io.BaseEncoding;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.LabelValidator;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.skyframe.DirectoryListingValue;
import com.google.devtools.build.lib.skyframe.DirectoryTreeDigestValue;
import com.google.devtools.build.lib.skyframe.EnvironmentVariableValue;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.skyframe.RepoEnvironmentFunction;
import com.google.devtools.build.lib.skyframe.RepositoryMappingValue;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyKey;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
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
public abstract sealed class RepoRecordedInput {
  /** Represents a parser for a specific type of recorded inputs. */
  public abstract static class Parser {
    /**
     * The prefix that identifies the type of the recorded inputs: for example, the {@code ENV} part
     * of {@code ENV:MY_ENV_VAR}.
     */
    public abstract String getPrefix();

    /**
     * Parses a recorded input from the post-colon substring that identifies the specific input: for
     * example, the {@code MY_ENV_VAR} part of {@code ENV:MY_ENV_VAR}. Returns null if the parsed
     * part is invalid.
     */
    public abstract RepoRecordedInput parse(String s);
  }

  /**
   * Parses a recorded input from its string representation.
   *
   * @param s the string representation
   * @return The parsed recorded input object, or {@link
   *     NeverUpToDateRepoRecordedInput#PARSE_FAILURE} if the string representation is invalid
   */
  public static RepoRecordedInput parse(String s) {
    List<String> parts = Splitter.on(':').limit(2).splitToList(s);
    if (parts.size() < 2) {
      return NeverUpToDateRepoRecordedInput.PARSE_FAILURE;
    }
    for (Parser parser :
        new Parser[] {
          File.PARSER, Dirents.PARSER, DirTree.PARSER, EnvVar.PARSER, RecordedRepoMapping.PARSER
        }) {
      if (parts.get(0).equals(parser.getPrefix())) {
        return parser.parse(parts.get(1));
      }
    }
    return NeverUpToDateRepoRecordedInput.PARSE_FAILURE;
  }

  /** A recorded input along with its recorded value. */
  @AutoCodec
  public record WithValue(RepoRecordedInput input, @Nullable String value) {
    /** Parses a {@link WithValue} from its string representation. */
    public static Optional<RepoRecordedInput.WithValue> parse(String s) {
      int sChar = s.indexOf(' ');
      if (sChar > 0) {
        var input = RepoRecordedInput.parse(unescape(s.substring(0, sChar)));
        if (!input.equals(NeverUpToDateRepoRecordedInput.PARSE_FAILURE)) {
          return Optional.of(new WithValue(input, unescape(s.substring(sChar + 1))));
        }
      }
      return Optional.empty();
    }

    /**
     * Splits the given list of recorded input values into batches such that within each batch, all
     * recorded inputs's {@link SkyKey}s can be requested together.
     */
    public static ImmutableList<ImmutableList<WithValue>> splitIntoBatches(
        List<WithValue> recordedInputValues) {
      var batches = ImmutableList.<ImmutableList<WithValue>>builder();
      var currentBatch = new ArrayList<WithValue>();
      for (var recordedInputValue : recordedInputValues) {
        if (!recordedInputValue.input().canBeRequestedUnconditionally()
            && !currentBatch.isEmpty()) {
          batches.add(ImmutableList.copyOf(currentBatch));
          currentBatch.clear();
        }
        currentBatch.add(recordedInputValue);
      }
      if (!currentBatch.isEmpty()) {
        batches.add(ImmutableList.copyOf(currentBatch));
      }
      return batches.build();
    }

    /** Converts this {@link WithValue} to a string in a format compatible with {@link #parse}. */
    @Override
    public String toString() {
      return input + " " + escape(value);
    }
  }

  /**
   * Returns whether all values are still up-to-date for each recorded input or a human-readable
   * reason for why that's not the case. If Skyframe values are missing, the return value should be
   * ignored; callers are responsible for checking {@code env.valuesMissing()} and triggering a
   * Skyframe restart if needed.
   */
  public static Optional<String> isAnyValueOutdated(
      Environment env, BlazeDirectories directories, List<WithValue> recordedInputValues)
      throws InterruptedException {
    prefetch(env, directories, Collections2.transform(recordedInputValues, WithValue::input));
    if (env.valuesMissing()) {
      return UNDECIDED;
    }
    for (var recordedInput : recordedInputValues) {
      Optional<String> reason =
          recordedInput.input().isOutdated(env, directories, recordedInput.value());
      if (reason.isPresent()) {
        return reason;
      }
    }
    return Optional.empty();
  }

  /**
   * Requests the information from Skyframe that is required by future calls to {@link
   * #isAnyValueOutdated} for the given set of inputs.
   */
  public static void prefetch(
      Environment env, BlazeDirectories directories, Collection<RepoRecordedInput> recordedInputs)
      throws InterruptedException {
    var keys =
        recordedInputs.stream().map(rri -> rri.getSkyKey(directories)).collect(toImmutableSet());
    if (env.valuesMissing()) {
      return;
    }
    var results = env.getValuesAndExceptions(keys);
    // Per the contract of Environment.getValuesAndExceptions, we need to access the results to
    // actually find all missing values.
    for (SkyKey key : keys) {
      var unused = results.get(key);
    }
  }

  /**
   * Returns a human-readable reason for why the given {@code oldValue} is no longer up-to-date for
   * this recorded input, or an empty Optional if it is still up-to-date.
   *
   * <p>The method can request Skyframe evaluations, and if any values are missing, this method can
   * return any value (doesn't matter what, although {@link #UNDECIDED} is recommended for clarity)
   * and will be reinvoked after a Skyframe restart.
   */
  private Optional<String> isOutdated(
      Environment env, BlazeDirectories directories, @Nullable String oldValue)
      throws InterruptedException {
    MaybeValue wrappedNewValue = getValue(env, directories);
    if (env.valuesMissing()) {
      return UNDECIDED;
    }
    return switch (wrappedNewValue) {
      case MaybeValue.Invalid(String reason) -> Optional.of(reason);
      case MaybeValue.Valid(String newValue) ->
          Objects.equals(oldValue, newValue)
              ? Optional.empty()
              : Optional.of(describeChange(oldValue, newValue));
    };
  }

  @Override
  public abstract boolean equals(Object obj);

  @Override
  public abstract int hashCode();

  @VisibleForTesting
  static String escape(String str) {
    return str == null ? "\\0" : str.replace("\\", "\\\\").replace("\n", "\\n").replace(" ", "\\s");
  }

  @VisibleForTesting
  @Nullable
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

  /**
   * Returns the string representation of this recorded input, in the format suitable for parsing
   * back via {@link #parse}.
   *
   * <p>The returned string never contains spaces or newlines; those characters are escaped.
   */
  @Override
  public final String toString() {
    return getParser().getPrefix() + ":" + escape(toStringInternal());
  }

  /**
   * Represents the possible values returned by {@link #getValue}: either a valid value (which may
   * be null), or an invalid value with a reason (e.g. due to I/O failure).
   */
  public sealed interface MaybeValue {
    MaybeValue VALUES_MISSING = new MaybeValue.Invalid("values missing");

    /** Represents a valid value, which may be null. */
    record Valid(@Nullable String value) implements MaybeValue {}

    /** Represents an invalid value with a reason (e.g. due to I/O failure). */
    record Invalid(String reason) implements MaybeValue {}
  }

  /**
   * Returns the current value of this input, which may be null, wrapped in a {@link
   * MaybeValue.Valid}, or a {@link MaybeValue.Invalid} if the value is known to be invalid.
   *
   * <p>The method can request Skyframe evaluations, and if any values are missing, this method can
   * return any value and will be reinvoked after a Skyframe restart.
   */
  public abstract MaybeValue getValue(Environment env, BlazeDirectories directories)
      throws InterruptedException;

  /**
   * Returns a human-readable description of the change from {@code oldValue} to {@code newValue}.
   */
  protected abstract String describeChange(String oldValue, String newValue);

  /**
   * Returns the post-colon substring that identifies the specific input: for example, the {@code
   * MY_ENV_VAR} part of {@code ENV:MY_ENV_VAR}.
   */
  protected abstract String toStringInternal();

  /** Returns the parser object for this type of recorded inputs. */
  protected abstract Parser getParser();

  /** Returns the {@link SkyKey} that is necessary to determine {@link #isOutdated}. */
  protected abstract SkyKey getSkyKey(BlazeDirectories directories);

  /**
   * Returns true if the {@link #getValue} can be requested even if previous recorded inputs have
   * not been verified to be up to date.
   */
  protected abstract boolean canBeRequestedUnconditionally();

  private static final Optional<String> UNDECIDED = Optional.of("values missing");

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
  @AutoCodec
  public record RepoCacheFriendlyPath(Optional<RepositoryName> repoName, PathFragment path) {
    public RepoCacheFriendlyPath {
      requireNonNull(repoName, "repoName");
      requireNonNull(path, "path");
    }

    public static RepoCacheFriendlyPath createInsideWorkspace(
        RepositoryName repoName, PathFragment path) {
      Preconditions.checkArgument(
          !path.isAbsolute(), "the provided path should be relative to the repo root: %s", path);
      return new RepoCacheFriendlyPath(Optional.of(repoName), path);
    }

    public static RepoCacheFriendlyPath createOutsideWorkspace(PathFragment path) {
      Preconditions.checkArgument(
          path.isAbsolute(), "the provided path should be absolute in the filesystem: %s", path);
      return new RepoCacheFriendlyPath(Optional.empty(), path);
    }

    @Override
    public final String toString() {
      // We store `@@foo//abc/def:ghi.bzl` as just `@@foo//abc/def/ghi.bzl`. See class javadoc for
      // more context.
      return repoName().map(repoName -> repoName + "//" + path()).orElse(path().toString());
    }

    public static RepoCacheFriendlyPath parse(String s) throws LabelSyntaxException {
      if (LabelValidator.isAbsolute(s)) {
        int doubleSlash = s.indexOf("//");
        int skipAts = s.startsWith("@@") ? 2 : s.startsWith("@") ? 1 : 0;
        return createInsideWorkspace(
            RepositoryName.create(s.substring(skipAts, doubleSlash)),
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

    /** Returns true if the path points into an external repository. */
    public boolean inExternalRepo() {
      return repoName().isPresent() && !repoName().get().isMain();
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
            try {
              return new File(RepoCacheFriendlyPath.parse(s));
            } catch (LabelSyntaxException e) {
              // malformed inputs cause refetch
              return NeverUpToDateRepoRecordedInput.PARSE_FAILURE;
            }
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
      if (!(o instanceof File that)) {
        return false;
      }
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
    @VisibleForTesting
    static String fileValueToMarkerValue(RootedPath rootedPath, FileValue fileValue)
        throws IOException {
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
        digest = fileValue.realRootedPath(rootedPath).asPath().getDigest();
      }
      return BaseEncoding.base16().lowerCase().encode(digest);
    }

    @Override
    public SkyKey getSkyKey(BlazeDirectories directories) {
      return FileValue.key(path.getRootedPath(directories));
    }

    @Override
    protected boolean canBeRequestedUnconditionally() {
      // Requesting files in external repositories can result in cycles if the external repo now
      // transitively depends on the requesting repo.
      return !path.inExternalRepo();
    }

    @Override
    public MaybeValue getValue(Environment env, BlazeDirectories directories)
        throws InterruptedException {
      var skyKey = getSkyKey(directories);
      try {
        var fileValue = (FileValue) env.getValueOrThrow(skyKey, IOException.class);
        if (fileValue == null) {
          return MaybeValue.VALUES_MISSING;
        }
        return new MaybeValue.Valid(
            fileValueToMarkerValue((RootedPath) skyKey.argument(), fileValue));
      } catch (IOException e) {
        return new MaybeValue.Invalid("failed to stat %s: %s".formatted(path, e.getMessage()));
      }
    }

    @Override
    public String describeChange(String oldValue, String newValue) {
      return "file info or contents of %s changed".formatted(path);
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
            try {
              return new Dirents(RepoCacheFriendlyPath.parse(s));
            } catch (LabelSyntaxException e) {
              // malformed inputs cause refetch
              return NeverUpToDateRepoRecordedInput.PARSE_FAILURE;
            }
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
      if (!(o instanceof Dirents that)) {
        return false;
      }
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

    private String directoryListingValueToMarkerValue(DirectoryListingValue directoryListingValue) {
      var fp = new Fingerprint();
      fp.addStrings(
          directoryListingValue.getDirents().stream()
              .map(Dirent::getName)
              .sorted()
              .collect(toImmutableList()));
      return fp.hexDigestAndReset();
    }

    @Override
    public SkyKey getSkyKey(BlazeDirectories directories) {
      return DirectoryListingValue.key(path.getRootedPath(directories));
    }

    @Override
    protected boolean canBeRequestedUnconditionally() {
      // Requesting directories in external repositories can result in cycles if the external repo
      // transitively depends on the requesting repo.
      return !path.inExternalRepo();
    }

    @Override
    public MaybeValue getValue(Environment env, BlazeDirectories directories)
        throws InterruptedException {
      var skyKey = getSkyKey(directories);
      var directoryListingValue = (DirectoryListingValue) env.getValue(skyKey);
      if (directoryListingValue == null) {
        return MaybeValue.VALUES_MISSING;
      }
      return new MaybeValue.Valid(directoryListingValueToMarkerValue(directoryListingValue));
    }

    @Override
    public String describeChange(String oldValue, String newValue) {
      return "directory entries of %s changed".formatted(path);
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
            try {
              return new DirTree(RepoCacheFriendlyPath.parse(s));
            } catch (LabelSyntaxException e) {
              // malformed inputs cause refetch
              return NeverUpToDateRepoRecordedInput.PARSE_FAILURE;
            }
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
      if (!(o instanceof DirTree that)) {
        return false;
      }
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
    protected boolean canBeRequestedUnconditionally() {
      // Requesting directory trees in external repositories can result in cycles if the external
      // repo now transitively depends on the requesting repo.
      return !path.inExternalRepo();
    }

    @Override
    public MaybeValue getValue(Environment env, BlazeDirectories directories)
        throws InterruptedException {
      var skyKey = getSkyKey(directories);
      try {
        var directoryTreeDigestValue =
            (DirectoryTreeDigestValue) env.getValueOrThrow(skyKey, IOException.class);
        if (directoryTreeDigestValue == null) {
          return MaybeValue.VALUES_MISSING;
        }
        return new MaybeValue.Valid(directoryTreeDigestValue.hexDigest());
      } catch (IOException e) {
        return new MaybeValue.Invalid(
            "failed to digest directory tree at %s: %s".formatted(path, e.getMessage()));
      }
    }

    @Override
    public String describeChange(String oldValue, String newValue) {
      return "directory tree at %s changed".formatted(path);
    }
  }

  /** Represents an environment variable accessed during the repo fetch. */
  public static final class EnvVar extends RepoRecordedInput {
    public static final Parser PARSER =
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

    public static ImmutableMap<EnvVar, Optional<String>> wrap(
        Map<String, Optional<String>> envVars) {
      return envVars.entrySet().stream()
          .sorted(comparingByKey())
          .collect(toImmutableMap(e -> new EnvVar(e.getKey()), Map.Entry::getValue));
    }

    public EnvVar(String name) {
      this.name = name;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (!(o instanceof EnvVar envVar)) {
        return false;
      }
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
      return RepoEnvironmentFunction.key(name);
    }

    @Override
    protected boolean canBeRequestedUnconditionally() {
      // Environment variables are static data injected into Skyframe, so there is no risk of
      // cycles.
      return true;
    }

    @Override
    public MaybeValue getValue(Environment env, BlazeDirectories directories)
        throws InterruptedException {
      var value = (EnvironmentVariableValue) env.getValue(getSkyKey(directories));
      if (value == null) {
        return MaybeValue.VALUES_MISSING;
      }
      return new MaybeValue.Valid(value.value());
    }

    @Override
    public String describeChange(String oldValue, String newValue) {
      return "environment variable %s changed: %s -> %s"
          .formatted(
              name,
              oldValue == null ? "<unset>" : "'%s'".formatted(oldValue),
              newValue == null ? "<unset>" : "'%s'".formatted(newValue));
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
            try {
              return new RecordedRepoMapping(RepositoryName.create(parts.get(0)), parts.get(1));
            } catch (LabelSyntaxException | IndexOutOfBoundsException e) {
              // malformed inputs cause refetch
              return NeverUpToDateRepoRecordedInput.PARSE_FAILURE;
            }
          }
        };

    final RepositoryName sourceRepo;
    final String apparentName;

    public RecordedRepoMapping(RepositoryName sourceRepo, String apparentName) {
      this.sourceRepo = sourceRepo;
      this.apparentName = apparentName;
    }

    public String apparentName() {
      return apparentName;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (!(o instanceof RecordedRepoMapping that)) {
        return false;
      }
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
    protected boolean canBeRequestedUnconditionally() {
      // Starlark can only request the mapping of the repo it is currently executing from, which
      // means that the repo has already been fetched (either to execute the code or to verify the
      // transitive .bzl hash). Further cycles aren't possible.
      return true;
    }

    @Override
    public MaybeValue getValue(Environment env, BlazeDirectories directories)
        throws InterruptedException {
      var repoMappingValue = (RepositoryMappingValue) env.getValue(getSkyKey(directories));
      if (Objects.equals(repoMappingValue, RepositoryMappingValue.NOT_FOUND_VALUE)) {
        return new MaybeValue.Invalid("source repo %s doesn't exist anymore".formatted(sourceRepo));
      }
      RepositoryName canonicalName = repoMappingValue.repositoryMapping().get(apparentName);
      return new MaybeValue.Valid(canonicalName != null ? canonicalName.getName() : null);
    }

    @Override
    public String describeChange(String oldValue, String newValue) {
      return "canonical name for @%s in %s changed: %s -> %s"
          .formatted(
              apparentName,
              sourceRepo,
              oldValue == null ? "<doesn't exist>" : oldValue,
              newValue == null ? "<doesn't exist>" : newValue);
    }
  }

  /** A sentinel "input" that's always out-of-date for a given reason. */
  public static final class NeverUpToDateRepoRecordedInput extends RepoRecordedInput {
    /** A sentinel "input" that's always out-of-date to signify parse failure. */
    public static final RepoRecordedInput PARSE_FAILURE =
        new NeverUpToDateRepoRecordedInput("malformed marker file entry encountered");

    private final String reason;

    public NeverUpToDateRepoRecordedInput(String reason) {
      this.reason = reason;
    }

    @Override
    public boolean equals(Object obj) {
      return this == obj;
    }

    @Override
    public int hashCode() {
      return 12345678;
    }

    @Override
    public String toStringInternal() {
      throw new UnsupportedOperationException("this sentinel input should never be serialized");
    }

    @Override
    public Parser getParser() {
      throw new UnsupportedOperationException("this sentinel input should never be parsed");
    }

    @Override
    public SkyKey getSkyKey(BlazeDirectories directories) {
      // Return a random SkyKey to satisfy the contract.
      return PrecomputedValue.STARLARK_SEMANTICS.getKey();
    }

    @Override
    protected boolean canBeRequestedUnconditionally() {
      return true;
    }

    @Override
    public MaybeValue getValue(Environment env, BlazeDirectories directories) {
      return new MaybeValue.Invalid(reason);
    }

    @Override
    public String describeChange(String oldValue, String newValue) {
      throw new UnsupportedOperationException(
          "the value for this sentinel input is always invalid");
    }
  }
}
