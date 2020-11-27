// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.includescanning;

import static java.nio.charset.StandardCharsets.ISO_8859_1;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Interner;
import com.google.common.collect.Sets;
import com.google.common.flogger.GoogleLogger;
import com.google.common.io.CharStreams;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactFactory;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.includescanning.IncludeParser.Inclusion.Kind;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.skyframe.ContainingPackageLookupValue;
import com.google.devtools.build.lib.skyframe.GlobDescriptor;
import com.google.devtools.build.lib.skyframe.GlobValue;
import com.google.devtools.build.lib.skyframe.GlobValue.InvalidGlobPatternException;
import com.google.devtools.build.lib.skyframe.PerBuildSyscallCache;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.UnixGlob;
import com.google.devtools.build.lib.vfs.UnixGlob.FilesystemCalls;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.ValueOrException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Executor;
import java.util.concurrent.atomic.AtomicReference;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;
import javax.annotation.Nullable;

/**
 * Scans a source file and extracts the literal inclusions it specifies. Does not store results --
 * repeated requests to the same file will result in repeated scans. Clients should implement a
 * caching layer in order to avoid unnecessary disk access when requesting an already scanned file.
 */
@VisibleForTesting
class IncludeParser {

  /**
   * File types supported by the grep-includes binary. {@link #fileType} must be kept in sync with
   * //tools/cpp:grep-includes.
   */
  public enum GrepIncludesFileType {
    CPP("c++"),
    SWIG("swig");

    private final String fileType;

    GrepIncludesFileType(String fileType) {
      this.fileType = fileType;
    }

    public String getFileType() {
      return fileType;
    }
  }

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  /**
   * Immutable object representation of the four columns making up a single Rule in a Hints set. See
   * {@link Hints} for more details.
   */
  private static class Rule {
    private enum Type {
      PATH,
      FILE,
      INCLUDE_QUOTE,
      INCLUDE_ANGLE
    }

    final Type type;
    final Pattern pattern;
    final String findRoot;
    final String findFilter;

    private Rule(String type, String pattern, String findRoot, String findFilter) {
      this.type = Type.valueOf(type.trim().toUpperCase());
      this.pattern = Pattern.compile("^" + pattern + "$");
      this.findRoot = findRoot.replace('\\', '$');
      this.findFilter = findFilter;
    }

    Rule(String type, String pattern, String findRoot) {
      this(type, pattern, findRoot, null);
      Preconditions.checkArgument(
          (this.type == Type.INCLUDE_QUOTE) || (this.type == Type.INCLUDE_ANGLE), this);
    }

    @Override
    public String toString() {
      return "" + type + " " + pattern + " " + findRoot + " " + findFilter;
    }
  }

  /** {@link SkyValue} encapsulating the source-state-dependent part of {@link Hints}. */
  public static class HintsRules implements SkyValue {
    private final ImmutableList<Rule> rules;

    private HintsRules(ImmutableList<Rule> rules) {
      this.rules = rules;
    }
  }

  /**
   * This class is a representation of the INCLUDE_HINTS file. The hints file contains regexp-based
   * rules to help this simple include scanner cope with computed includes, which would otherwise
   * require a full preprocessor with symbol support. Instead of actually processing symbols to
   * evaluate the computed includes, we instead apply rules to gather inclusions for matching paths.
   *
   * <p>The hints file is read, line by line, into a list of rules each of which encapsulates a line
   * of four columns. Each non-blank, non-comment line has the format:
   *
   * <pre>
   *   &quot;file&quot;|&quot;path&quot;  match-pattern  find-root  find-filter
   * </pre>
   *
   * <p>The first column specifies whether the line is a rule based on matching source
   * <em>files</em> (passed directly to the compiler as inputs, or transitively #included by other
   * inputs) or include <em>paths</em> (passed to the compiler as -I, -iquote, or -isystem flags).
   *
   * <p>The second column is a regexp for files or paths. Whenever a compiler argument of the
   * specified type matches that regexp, the rule is taken. (All matching rules for every path and
   * file on a compiler command line are followed, and the results are combined.)
   *
   * <p>The third column is a point in the local filesystem from which to extract a recursive
   * listing. (This follows symlinks) Backrefs may be used to refer to the regexp or its capturing
   * groups. (This is mostly necessary because --package_path can cause input paths to carry
   * arbitrary prefixes.)
   *
   * <p>The fourth column is a regexp applied to each file found by the recursive listing. All
   * matching files are treated as dependencies.
   */
  public static class Hints {
    private static final Pattern WS_PAT = Pattern.compile("\\s+");
    @VisibleForTesting static final String ALLOWED_PREFIX = "third_party/";
    // Match regular expressions that can only match paths under ALLOWED_PREFIX .
    private static final Pattern ALLOWED_PATTERN = Pattern.compile("^\\(*" + ALLOWED_PREFIX + ".*");

    private static final int HINTS_CACHE_CONCURRENCY = 100;

    private final ImmutableList<Rule> rules;
    private final ArtifactFactory artifactFactory;

    private final AtomicReference<FilesystemCalls> syscallCache = new AtomicReference<>();

    private final LoadingCache<Artifact, Collection<Artifact>> fileLevelHintsCache =
        CacheBuilder.newBuilder()
            .concurrencyLevel(HINTS_CACHE_CONCURRENCY)
            .build(
                new CacheLoader<Artifact, Collection<Artifact>>() {
                  @Override
                  public Collection<Artifact> load(Artifact path) {
                    return getHintedInclusionsLegacy(
                        Rule.Type.FILE, path.getExecPath(), path.getRoot());
                  }
                });

    /**
     * Constructs a hint set for a given INCLUDE_HINTS file to read.
     *
     * @param hintsRules the {@link HintsRules} parsed from INCLUDE_HINTS
     */
    public Hints(HintsRules hintsRules, ArtifactFactory artifactFactory) {
      this.artifactFactory = artifactFactory;
      this.rules = hintsRules.rules;
      clearCachedLegacyHints();
    }

    static HintsRules getRules(Path hintsFile) throws IOException {
      ImmutableList.Builder<Rule> rules = ImmutableList.builder();
      try (InputStream is = hintsFile.getInputStream()) {
        for (String line : CharStreams.readLines(new InputStreamReader(is, "UTF-8"))) {
          line = line.trim();
          if (line.length() == 0 || line.startsWith("#")) {
            continue;
          }
          String[] tokens = WS_PAT.split(line);
          try {
            if (tokens.length == 3) {
              rules.add(new Rule(tokens[0], tokens[1], tokens[2]));
            } else if (tokens.length == 4) {
              if (!ALLOWED_PATTERN.matcher(tokens[1]).matches()) {
                throw new IOException(
                    "Illegal hint regex on: "
                        + line
                        + "\n"
                        + tokens[1]
                        + " does not match only paths in "
                        + ALLOWED_PREFIX);
              }
              rules.add(new Rule(tokens[0], tokens[1], tokens[2], tokens[3]));
            } else {
              throw new IOException("Malformed hint line: " + line);
            }
          } catch (PatternSyntaxException e) {
            throw new IOException("Malformed hint regex on: " + line + "\n  " + e.getMessage());
          } catch (IllegalArgumentException e) {
            throw new IOException("Invalid type on: " + line + "\n  " + e.getMessage());
          }
        }
      }
      return new HintsRules(rules.build());
    }

    /**
     * Clears legacy inclusions cache to maintain inter-build correctness, since filesystem changes
     * are not tracked by cache.
     */
    void clearCachedLegacyHints() {
      fileLevelHintsCache.invalidateAll();
      syscallCache.set(
          PerBuildSyscallCache.newBuilder().setConcurrencyLevel(HINTS_CACHE_CONCURRENCY).build());
    }

    /** Returns the "file" type hinted inclusions for a given path, caching results by path. */
    Collection<Artifact> getFileLevelHintedInclusionsLegacy(Artifact path) {
      if (!path.getExecPathString().startsWith(ALLOWED_PREFIX)) {
        return ImmutableList.of();
      }
      return fileLevelHintsCache.getUnchecked(path);
    }

    /**
     * Returns the "path" type hinted inclusions for the given paths. Callers are responsible for
     * caching.
     */
    Collection<Artifact> getPathLevelHintedInclusions(
        ImmutableList<PathFragment> paths, Environment env) throws InterruptedException {
      return getHintedInclusionsWithSkyframe(Rule.Type.PATH, paths, env);
    }

    /**
     * Performs the work of matching the given paths against the hints and returns the matching
     * files. This is semantically different from {@link #getHintedInclusionsLegacy} in that it will
     * not cross package boundaries.
     */
    private Collection<Artifact> getHintedInclusionsWithSkyframe(
        Rule.Type type, ImmutableList<PathFragment> paths, Environment env)
        throws InterruptedException {
      ImmutableList<String> pathStrings =
          paths.stream()
              .map(PathFragment::getPathString)
              .filter((p) -> p.startsWith(ALLOWED_PREFIX))
              .collect(ImmutableList.toImmutableList());
      if (pathStrings.isEmpty()) {
        return ImmutableList.of();
      }
      // Delay creation until we know we need one. Use a TreeSet to make sure that the results are
      // sorted with a stable order and unique.
      Set<Artifact> hints = null;
      List<ContainingPackageLookupValue.Key> rulePaths = new ArrayList<>(rules.size());
      List<String> findFilters = new ArrayList<>(rules.size());
      for (Rule rule : rules) {
        if (type != rule.type) {
          continue;
        }
        String firstMatchPathString = null;
        Matcher m = null;
        for (String pathString : pathStrings) {
          m = rule.pattern.matcher(pathString);
          if (m.matches()) {
            firstMatchPathString = pathString;
            break;
          }
        }
        if (firstMatchPathString == null) {
          continue;
        }
        if (hints == null) {
          hints = Sets.newTreeSet(Artifact.EXEC_PATH_COMPARATOR);
        }
        PathFragment relativePath = PathFragment.create(m.replaceFirst(rule.findRoot));
        logger.atFine().log(
            "hint for %s %s root: %s", rule.type, firstMatchPathString, relativePath);
        if (!relativePath.getPathString().startsWith(ALLOWED_PREFIX)) {
          logger.atWarning().log(
              "Path %s to search after substitution does not start with %s",
              relativePath.getPathString(), ALLOWED_PREFIX);
          continue;
        }
        rulePaths.add(
            ContainingPackageLookupValue.key(PackageIdentifier.createInMainRepo(relativePath)));
        findFilters.add(rule.findFilter);
      }
      Map<SkyKey, ValueOrException<NoSuchPackageException>> containingPackageLookupValues =
          env.getValuesOrThrow(rulePaths, NoSuchPackageException.class);
      if (env.valuesMissing()) {
        return null;
      }
      List<GlobDescriptor> globKeys = new ArrayList<>(rulePaths.size());
      for (int i = 0; i < rulePaths.size(); i++) {
        ContainingPackageLookupValue containingPackageLookupValue;
        ContainingPackageLookupValue.Key relativePathKey = rulePaths.get(i);
        PathFragment relativePath = relativePathKey.argument().getPackageFragment();
        try {
          containingPackageLookupValue =
              (ContainingPackageLookupValue)
                  containingPackageLookupValues.get(relativePathKey).get();
        } catch (NoSuchPackageException e) {
          logger.atWarning().withCause(e).log(
              "Unexpected exception when looking up containing package for %s"
                  + " (prodaccess expired?)",
              relativePath);
          continue;
        }
        if (!containingPackageLookupValue.hasContainingPackage()) {
          logger.atWarning().log("%s not contained in any package: skipping", relativePath);
          continue;
        }
        PathFragment packageFragment =
            containingPackageLookupValue.getContainingPackageName().getPackageFragment();
        String pattern = findFilters.get(i);
        try {
          globKeys.add(
              GlobValue.key(
                  containingPackageLookupValue.getContainingPackageName(),
                  containingPackageLookupValue.getContainingPackageRoot(),
                  pattern,
                  /* excludeDirs= */ true,
                  relativePath.relativeTo(packageFragment)));
        } catch (InvalidGlobPatternException e) {
          env.getListener()
              .handle(Event.warn("Error parsing pattern " + pattern + " for " + relativePath));
          continue;
        }
      }
      Map<SkyKey, ValueOrException<IOException>> globResults =
          env.getValuesOrThrow(globKeys, IOException.class);
      if (env.valuesMissing()) {
        return null;
      }
      for (Map.Entry<SkyKey, ValueOrException<IOException>> globEntry : globResults.entrySet()) {
        GlobValue globValue;
        GlobDescriptor globKey = (GlobDescriptor) globEntry.getKey();
        PathFragment packageFragment = globKey.getPackageId().getPackageFragment();
        try {
          globValue = (GlobValue) globEntry.getValue().get();
        } catch (IOException e) {
          logger.atWarning().withCause(e).log("Error getting hints for %s", packageFragment);
          continue;
        }
        for (PathFragment file : globValue.getMatches().toList()) {
          hints.add(
              artifactFactory.getSourceArtifact(
                  packageFragment.getRelative(file), globKey.getPackageRoot()));
        }
      }
      return hints == null || hints.isEmpty() ? ImmutableList.<Artifact>of() : hints;
    }

    /**
     * Performs the work of matching a given path against the hints and returns the expanded paths.
     * The above {@link #getHintedInclusionsWithSkyframe} should be used in preference, but if the
     * performance impact of Skyframe restarts is untenable, this can be used as a fallback.
     */
    private Collection<Artifact> getHintedInclusionsLegacy(
        Rule.Type type, PathFragment path, ArtifactRoot sourceRoot) {
      String pathString = path.getPathString();
      // Delay creation until we know we need one. Use a TreeSet to make sure that the results are
      // sorted with a stable order and unique.
      Set<Path> hints = null;
      for (final Rule rule : rules) {
        if (type != rule.type) {
          continue;
        }
        Matcher m = rule.pattern.matcher(pathString);
        if (!m.matches()) {
          continue;
        }
        if (hints == null) {
          hints = Sets.newTreeSet();
        }
        String relativePath = m.replaceFirst(rule.findRoot);
        if (!relativePath.startsWith(ALLOWED_PREFIX)) {
          logger.atWarning().log(
              "Path %s to search after substitution does not start with %s",
              relativePath, ALLOWED_PREFIX);
          continue;
        }
        Path root = sourceRoot.getRoot().getRelative(relativePath);

        logger.atFine().log("hint for %s %s root: %s", rule.type, pathString, root);
        try {
          // The assumption is made here that all files specified by this hint are under the same
          // package path as the original file -- this filesystem tree traversal is completely
          // ignorant of package paths. This could be violated if there were a hint that resolved to
          // foo/**/*.h, there was a package foo/bar, and the packages foo and foo/bar were in
          // different package paths. In that case, this traversal would fail to pick up
          // foo/bar/**/*.h. No examples of this currently exist in the INCLUDE_HINTS
          // file.
          logger.atFine().log("Globbing: %s %s", root, rule.findFilter);
          hints.addAll(
              new UnixGlob.Builder(root)
                  .setFilesystemCalls(syscallCache)
                  .addPattern(rule.findFilter)
                  .glob());
        } catch (UnixGlob.BadPattern | IOException e) {
          logger.atWarning().withCause(e).log("Error in hint expansion");
        }
      }
      if (hints != null && !hints.isEmpty()) {
        // Transform paths into source artifacts (all hints must be to source artifacts).
        List<Artifact> result = new ArrayList<>(hints.size());
        for (Path hint : hints) {
          Root sourcePath = sourceRoot.getRoot();
          result.add(
              Preconditions.checkNotNull(
                  artifactFactory.getSourceArtifact(sourcePath.relativize(hint), sourcePath),
                  "%s %s %s %s",
                  hint,
                  sourcePath,
                  path));
        }
        return result;
      } else {
        return ImmutableList.of();
      }
    }

    private Collection<Inclusion> getHintedInclusions(Artifact path) {
      String pathString = path.getExecPathString();
      // Delay creation until we know we need one. Use a LinkedHashSet to make sure that the results
      // are sorted with a stable order and unique.
      Set<Inclusion> hints = null;
      for (final Rule rule : rules) {
        if ((rule.type != Rule.Type.INCLUDE_ANGLE) && (rule.type != Rule.Type.INCLUDE_QUOTE)) {
          continue;
        }
        Matcher m = rule.pattern.matcher(pathString);
        if (!m.matches()) {
          continue;
        }
        if (hints == null) {
          hints = Sets.newLinkedHashSet();
        }
        Inclusion inclusion =
            Inclusion.create(
                rule.findRoot, rule.type == Rule.Type.INCLUDE_QUOTE ? Kind.QUOTE : Kind.ANGLE);
        hints.add(inclusion);
        logger.atFine().log("hint for %s %s root: %s", rule.type, pathString, inclusion);
      }
      if (hints != null && !hints.isEmpty()) {
        return ImmutableList.copyOf(hints);
      } else {
        return ImmutableList.of();
      }
    }
  }

  Hints getHints() {
    return hints;
  }

  /**
   * An immutable inclusion tuple. This models an {@code #include} or {@code #include_next} line in
   * a file without the context how this file got included.
   */
  public static class Inclusion {
    private static final Interner<Inclusion> INCLUSIONS = BlazeInterners.newWeakInterner();

    /** The format of the #include in the source file -- quoted, angle bracket, etc. */
    enum Kind {
      /** Quote includes: {@code #include "name"}. */
      QUOTE,

      /** Angle bracket includes: {@code #include <name>}. */
      ANGLE,

      /** Quote next includes: {@code #include_next "name"}. */
      NEXT_QUOTE,

      /** Angle next includes: {@code #include_next <name>}. */
      NEXT_ANGLE;

      /** Returns true if this is an {@code #include_next} inclusion, */
      boolean isNext() {
        return this == NEXT_ANGLE || this == NEXT_QUOTE;
      }
    }

    /** The kind of inclusion. */
    final Kind kind;
    /** The relative path of the inclusion. */
    final PathFragment pathFragment;

    private Inclusion(PathFragment pathFragment, Kind kind) {
      this.kind = kind;
      this.pathFragment = Preconditions.checkNotNull(pathFragment);
    }

    static Inclusion create(String includeTarget, Kind kind) {
      return INCLUSIONS.intern(new Inclusion(PathFragment.create(includeTarget), kind));
    }

    static Inclusion create(PathFragment pathFragment, Kind kind) {
      return INCLUSIONS.intern(new Inclusion(Preconditions.checkNotNull(pathFragment), kind));
    }

    String getPathString() {
      return pathFragment.getPathString();
    }

    @Override
    public String toString() {
      return kind + ":" + pathFragment.getPathString();
    }

    @Override
    public boolean equals(Object o) {
      if (o == this) {
        return true;
      }
      if (!(o instanceof Inclusion)) {
        return false;
      }
      Inclusion that = (Inclusion) o;
      return kind == that.kind && pathFragment.equals(that.pathFragment);
    }

    @Override
    public int hashCode() {
      return pathFragment.hashCode() * 37 + kind.hashCode();
    }
  }

  /** The externally-scoped immutable hints helper that is shared by all scanners. */
  private final Hints hints;

  /**
   * Constructs a new FileParser.
   *
   * @param hints regexps for converting computed includes into simple strings
   */
  public IncludeParser(Hints hints) {
    this.hints = hints;
  }

  /**
   * Skips whitespace, \+NL pairs, and block-style / * * / comments. Assumes line comments are
   * handled outside. Does not handle digraphs, trigraphs or decahexagraphs.
   *
   * @param chars characters to scan
   * @param pos the starting position
   * @return the resulting position after skipping whitespace and comments.
   */
  protected static int skipWhitespace(byte[] chars, int pos, int end) {
    while (pos < end) {
      if (Character.isWhitespace(chars[pos] & 0xff)) {
        pos++;
      } else if (chars[pos] == '\\' && pos + 1 < end && chars[pos + 1] == '\n') {
        pos++;
      } else if (chars[pos] == '/' && pos + 1 < end && chars[pos + 1] == '*') {
        pos += 2;
        while (pos < end - 1) {
          if (chars[pos++] == '*') {
            if (chars[pos] == '/') {
              pos++;
              break; // proper comment end
            }
          }
        }
      } else { // not whitespace
        return pos;
      }
    }
    return pos; // pos == len, meaning we fell off the end.
  }

  private static final String HAS_INCLUDE = "__has_include";
  private static final int HAS_INCLUDE_LENGTH = HAS_INCLUDE.length();
  private static final int NECESSARY_HAS_INCLUDE_LENGTH = HAS_INCLUDE_LENGTH + 5;

  /**
   * Returns the index of {@code chars} after the first occurrence of "__has_include" or -1 if no
   * such occurrence exists. Also requires that there be at least 5 characters after the
   * "__has_include", corresponding to a pair of parentheses and angle brackets/quotes and a
   * filename.
   *
   * <p>This code runs on every line that starts with " *# *", so it should be as fast as possible.
   */
  private static int skipThroughHasInclude(byte[] chars, int pos, int end) {
    int lastPos = end - NECESSARY_HAS_INCLUDE_LENGTH;
    while (pos <= lastPos) {
      int curPos = 0;
      while (curPos < HAS_INCLUDE_LENGTH
          && (chars[pos + curPos] & 0xff) == HAS_INCLUDE.charAt(curPos)) {
        curPos++;
      }
      if (curPos == HAS_INCLUDE_LENGTH) {
        return pos + curPos;
      }
      // We're looking for "__has_include" as a preprocessing token, which means that it cannot
      // start in the middle of any characters we've already processed, nor at the mismatching
      // character.
      pos += curPos + 1;
    }
    return -1;
  }

  /**
   * Checks for and skips a given token.
   *
   * @param chars characters to scan
   * @param pos the starting position
   * @param expected the expected token
   * @return the resulting position if found, otherwise -1
   */
  protected static int expect(byte[] chars, int pos, int end, String expected) {
    int si = 0;
    int expectedLen = expected.length();
    while (pos < end) {
      if (si == expectedLen) {
        return pos;
      }
      if ((chars[pos++] & 0xff) != expected.charAt(si++)) {
        return -1;
      }
    }
    return -1;
  }

  /**
   * Finds the index of a given character token from a starting pos.
   *
   * @param chars characters to scan
   * @param pos the starting position
   * @param echar the character to find
   * @return the resulting position of echar if found, otherwise -1
   */
  private static int indexOf(byte[] chars, int pos, int end, char echar) {
    while (pos < end) {
      if (chars[pos] == echar) {
        return pos;
      }
      pos++;
    }
    return -1;
  }

  private static final Pattern BS_NL_PAT = Pattern.compile("\\\\" + "\n");

  // Keep this in sync with the grep-includes binary's scanning output format.
  private static final ImmutableMap<Character, Kind> KIND_MAP =
      ImmutableMap.of(
          '"', Kind.QUOTE,
          '<', Kind.ANGLE,
          'q', Kind.NEXT_QUOTE,
          'a', Kind.NEXT_ANGLE);

  /**
   * Processes the output generated by an auxiliary include-scanning binary.
   *
   * <p>If a source file has the following include statements:
   *
   * <pre>
   *   #include &lt;string&gt;
   *   #include "directory/header.h"
   * </pre>
   *
   * <p>Then the output file has the following contents:
   *
   * <pre>
   *   "directory/header.h
   *   &lt;string
   * </pre>
   *
   * <p>Each line of the output is translated into an Inclusion object.
   */
  private static List<Inclusion> processIncludes(List<String> lines) throws IOException {
    List<Inclusion> inclusions = new ArrayList<>();
    for (String line : lines) {
      if (line.isEmpty()) {
        continue;
      }
      char qchar = line.charAt(0);
      String name = line.substring(1);
      Kind kind = KIND_MAP.get(qchar);
      if (kind == null) {
        throw new IOException("Illegal inclusion kind '" + qchar + "'");
      }
      inclusions.add(Inclusion.create(name, kind));
    }
    return inclusions;
  }

  /** Processes the output generated by an auxiliary include-scanning binary stored in a file. */
  public static List<Inclusion> processIncludes(Path file) throws IOException {
    try {
      byte[] data = FileSystemUtils.readContent(file);
      return IncludeParser.processIncludes(Arrays.asList(new String(data, ISO_8859_1).split("\n")));
    } catch (IOException e) {
      throw new IOException("Error reading include file " + file + ": " + e.getMessage());
    }
  }

  /**
   * Processes the output generated by an auxiliary include-scanning binary read from a stream.
   * Closes the stream upon completion.
   */
  public static List<Inclusion> processIncludes(Object streamName, InputStream is)
      throws IOException {
    try (InputStreamReader reader = new InputStreamReader(is, ISO_8859_1)) {
      return processIncludes(CharStreams.readLines(reader));
    } catch (IOException e) {
      throw new IOException("Error reading include file " + streamName + ": " + e.getMessage());
    }
  }

  @VisibleForTesting
  Inclusion extractInclusion(String line) {
    return extractInclusion(line.getBytes(ISO_8859_1), 0, line.length());
  }

  /**
   * Extracts a new, unresolved an Inclusion from a line of source.
   *
   * @param chars the char array containing the line chars to parse
   * @param lineBegin the position of the first character in the line
   * @param lineEnd the position of the character after the last
   * @return the inclusion object if possible, null if none
   */
  private Inclusion extractInclusion(byte[] chars, int lineBegin, int lineEnd) {
    // expect WS#WS(include|include_next|__has_include\(_next\)?)WS\(?("name"|<name>|<name>)\)?
    IncludesKeywordData data = expectIncludeKeyword(chars, lineBegin, lineEnd);
    int pos = data.pos;
    if (pos == -1 || pos == lineEnd) {
      return null;
    }
    boolean isNext = false;
    if (data.canHaveNext) {
      int npos = expect(chars, pos, lineEnd, "_next");
      if (npos >= 0) {
        isNext = true;
        pos = npos;
      }
    }
    if ((pos = skipWhitespace(chars, pos, lineEnd)) == lineEnd) {
      return null;
    }
    if (data.hasParens) {
      if (chars[pos] != '(') {
        return null;
      }
      pos++;
      if ((pos = skipWhitespace(chars, pos, lineEnd)) == lineEnd) {
        return null;
      }
    }
    if (chars[pos] == '"' || chars[pos] == '<') {
      char qchar = (char) (chars[pos++] & 0xff);
      int spos = pos;
      pos = indexOf(chars, pos + 1, lineEnd, qchar == '<' ? '>' : '"');
      if (pos < 0) {
        return null;
      }
      if (chars[spos] == '/') {
        return null; // disallow absolute paths
      }
      String name = new String(chars, spos, pos - spos);
      if (name.contains("\n")) { // strip any \+NL pairs within name
        name = BS_NL_PAT.matcher(name).replaceAll("");
      }
      if (isNext) {
        return Inclusion.create(name, qchar == '"' ? Kind.NEXT_QUOTE : Kind.NEXT_ANGLE);
      } else {
        return Inclusion.create(name, qchar == '"' ? Kind.QUOTE : Kind.ANGLE);
      }
    } else {
      return createOtherInclusion(new String(chars, pos, lineEnd - pos));
    }
  }

  /**
   * Extracts all inclusions from characters of a file.
   *
   * @param chars the file contents to parse & extract inclusions from
   * @return a new set of inclusions, normalized to the cache
   */
  @VisibleForTesting
  List<Inclusion> extractInclusions(byte[] chars) {
    List<Inclusion> inclusions = new ArrayList<>();
    int lineBegin = 0; // the first char of each line
    int end = chars.length; // the file end
    while (lineBegin < end) {
      int lineEnd = lineBegin; // the char after the last non-\n in each line
      // skip to the next \n or after end of buffer, ignoring continuations
      while (lineEnd < end) {
        if (chars[lineEnd] == '\n') {
          break;
        } else if (chars[lineEnd] == '\\') {
          lineEnd++;
          if (chars[lineEnd] == '\n') {
            lineEnd++;
          }
        } else {
          lineEnd++;
        }
      }

      // TODO(bazel-team) handle multiline block comments /* */ for the cases:
      //   /* blah blah blah
      //    lalala  */ #include "foo.h"
      // and:
      //   /* blah
      //   #include "foo.h"
      //   */

      // extract the inclusion, and save only the kind we care about.
      Inclusion inclusion = extractInclusion(chars, lineBegin, lineEnd);
      if (inclusion != null) {
        if (isValidInclusionKind(inclusion.kind)) {
          inclusions.add(inclusion);
        }
      }
      lineBegin = lineEnd + 1; // next line starts after the previous line
    }
    return inclusions;
  }

  /**
   * Extracts all inclusions from a given source file.
   *
   * @param file the file to parse & extract inclusions from
   * @param actionExecutionContext Services in the scope of the action, like the stream to which
   *     scanning messages are printed
   * @return a new set of inclusions, normalized to the cache
   */
  Collection<Inclusion> extractInclusions(
      Artifact file,
      ActionExecutionMetadata actionExecutionMetadata,
      ActionExecutionContext actionExecutionContext,
      Artifact grepIncludes,
      @Nullable SpawnIncludeScanner remoteIncludeScanner,
      boolean isOutputFile)
      throws IOException, ExecException, InterruptedException {
    Collection<Inclusion> inclusions;

    // TODO(ulfjack): grepIncludes may be null if the corresponding attribute on the rule is missing
    //  (see CppHelper.getGrepIncludes) or misspelled. It would be better to disallow this case.
    if (remoteIncludeScanner != null
        && grepIncludes != null
        && remoteIncludeScanner.shouldParseRemotely(file, actionExecutionContext)) {
      inclusions =
          remoteIncludeScanner.extractInclusions(
              file,
              actionExecutionMetadata,
              actionExecutionContext,
              grepIncludes,
              getFileType(),
              isOutputFile);
    } else {
      try (SilentCloseable c =
          Profiler.instance().profile(ProfilerTask.SCANNER, file.getExecPathString())) {
        inclusions =
            extractInclusions(
                FileSystemUtils.readContent(actionExecutionContext.getInputPath(file)));
      } catch (IOException e) {
        if (remoteIncludeScanner != null && grepIncludes != null) {
          logger.atWarning().withCause(e).log(
              "Falling back on remote parsing of %s", actionExecutionContext.getInputPath(file));
          inclusions =
              remoteIncludeScanner.extractInclusions(
                  file,
                  actionExecutionMetadata,
                  actionExecutionContext,
                  grepIncludes,
                  getFileType(),
                  isOutputFile);
        } else {
          throw e;
        }
      }
    }
    if (hints != null) {
      inclusions.addAll(hints.getHintedInclusions(file));
    }
    return ImmutableList.copyOf(inclusions);
  }

  /**
   * Extracts all inclusions from a given source file.
   *
   * @param file the file to parse & extract inclusions from
   * @param actionExecutionContext Services in the scope of the action, like the stream to which
   *     scanning messages are printed
   * @return a new set of inclusions, normalized to the cache
   */
  ListenableFuture<Collection<Inclusion>> extractInclusionsAsync(
      Executor executor,
      Artifact file,
      ActionExecutionMetadata actionExecutionMetadata,
      ActionExecutionContext actionExecutionContext,
      Artifact grepIncludes,
      @Nullable SpawnIncludeScanner remoteIncludeScanner,
      boolean isOutputFile)
      throws IOException {
    ListenableFuture<Collection<Inclusion>> inclusions;
    if (remoteIncludeScanner != null
        && remoteIncludeScanner.shouldParseRemotely(file, actionExecutionContext)) {
      inclusions =
          remoteIncludeScanner.extractInclusionsAsync(
              executor,
              file,
              actionExecutionMetadata,
              actionExecutionContext,
              grepIncludes,
              getFileType(),
              isOutputFile);
    } else {
      try (SilentCloseable c =
          Profiler.instance().profile(ProfilerTask.SCANNER, file.getExecPathString())) {
        inclusions =
            Futures.immediateFuture(
                extractInclusions(
                    FileSystemUtils.readContent(actionExecutionContext.getInputPath(file))));
      } catch (IOException e) {
        if (remoteIncludeScanner != null) {
          logger.atWarning().withCause(e).log(
              "Falling back on remote parsing of %s", actionExecutionContext.getInputPath(file));
          inclusions =
              remoteIncludeScanner.extractInclusionsAsync(
                  executor,
                  file,
                  actionExecutionMetadata,
                  actionExecutionContext,
                  grepIncludes,
                  getFileType(),
                  isOutputFile);
        } else {
          throw e;
        }
      }
    }
    if (hints != null) {
      return Futures.transform(
          inclusions,
          (c) -> {
            // Ugly, but saves doing another copy.
            c.addAll(hints.getHintedInclusions(file));
            return c;
          },
          MoreExecutors.directExecutor());
    }
    return inclusions;
  }

  /**
   * Returns type of the scanned file.
   *
   * <p>Supported values are "c++" for standard c/c++ headers and sources, and "swig" for .swig
   * files. Changes to this method must be synchronized with change to //tools/cpp:grep-includes.
   */
  protected GrepIncludesFileType getFileType() {
    return GrepIncludesFileType.CPP;
  }

  /**
   * Position of found include together with information about how to process the remaining include
   * line further.
   */
  protected static class IncludesKeywordData {
    protected static final IncludesKeywordData NONE = new IncludesKeywordData(-1, false, false);
    private final int pos;
    private final boolean canHaveNext;
    private final boolean hasParens;

    private IncludesKeywordData(int pos, boolean canHaveNext, boolean hasParens) {
      this.pos = pos;
      this.canHaveNext = canHaveNext;
      this.hasParens = hasParens;
    }

    protected static IncludesKeywordData normal(int pos) {
      return new IncludesKeywordData(pos, true, false);
    }

    protected static IncludesKeywordData importOrSwig(int pos) {
      return new IncludesKeywordData(pos, false, false);
    }

    protected static IncludesKeywordData hasInclude(int pos) {
      return new IncludesKeywordData(pos, true, true);
    }
  }

  /**
   * Parses include keyword in the provided char array and returns position immediately after
   * include keyword or -1 if keyword was not found, along with information to aid future parsing.
   * Can be overridden by subclasses.
   */
  protected IncludesKeywordData expectIncludeKeyword(byte[] chars, int position, int end) {
    int pos = expect(chars, skipWhitespace(chars, position, end), end, "#");
    if (pos > 0) {
      int npos = skipWhitespace(chars, pos, end);
      if ((pos = expect(chars, npos, end, "include")) > 0) {
        return IncludesKeywordData.normal(pos);
      } else if ((pos = expect(chars, npos, end, "import")) > 0) {
        return IncludesKeywordData.importOrSwig(pos);
      } else if ((pos = skipThroughHasInclude(chars, npos, end)) > 0) {
        return IncludesKeywordData.hasInclude(pos);
      }
    }
    return IncludesKeywordData.NONE;
  }

  /**
   * Returns true if we interested in the given inclusion kind. Can be overridden by the subclass.
   *
   * @param kind
   */
  protected boolean isValidInclusionKind(Kind kind) {
    return true;
  }

  /**
   * Returns inclusion object for non-standard inclusion cases or null if inclusion should be
   * ignored.
   *
   * @param inclusionContent
   */
  @Nullable
  protected Inclusion createOtherInclusion(String inclusionContent) {
    return null;
  }
}
