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

import static com.google.common.collect.ImmutableList.toImmutableList;
import static java.nio.charset.StandardCharsets.ISO_8859_1;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.github.benmanes.caffeine.cache.Caffeine;
import com.github.benmanes.caffeine.cache.LoadingCache;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Interner;
import com.google.common.collect.Sets;
import com.google.common.flogger.GoogleLogger;
import com.google.common.io.CharStreams;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactFactory;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.includescanning.IncludeParser.Inclusion.Kind;
import com.google.devtools.build.lib.packages.BuildFileNotFoundException;
import com.google.devtools.build.lib.packages.Globber;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.skyframe.ContainingPackageLookupValue;
import com.google.devtools.build.lib.skyframe.GlobDescriptor;
import com.google.devtools.build.lib.skyframe.GlobValue;
import com.google.devtools.build.lib.skyframe.InvalidGlobPatternException;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.lib.vfs.UnixGlob;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.SkyframeLookupResult;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;
import javax.annotation.Nullable;

/**
 * Scans a source file and extracts the literal inclusions it specifies. Does not store results --
 * repeated requests to the same file will result in repeated scans. Clients should implement a
 * caching layer in order to avoid unnecessary disk access when requesting an already scanned file.
 *
 * <p>Both this class and the static inner class {@link Hints} have lifetime of a single build (or a
 * single include scanning operation in the case of the {@link SwigIncludeParser}).
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
      return type + " " + pattern + " " + findRoot + " " + findFilter;
    }
  }

  /** {@link SkyValue} encapsulating the source-state-dependent part of {@link Hints}. */
  public static final class HintsRules implements SkyValue {
    static final HintsRules EMPTY = new HintsRules(ImmutableList.of());
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

    private final SyscallCache syscallCache;

    private final LoadingCache<Artifact, ImmutableList<Artifact>> fileLevelHintsCache =
        Caffeine.newBuilder()
            .initialCapacity(HINTS_CACHE_CONCURRENCY)
            .build(this::getHintedInclusionsLegacy);

    /**
     * Constructs a hint set for a given INCLUDE_HINTS file to read.
     *
     * @param hintsRules the {@link HintsRules} parsed from INCLUDE_HINTS
     */
    Hints(HintsRules hintsRules, SyscallCache syscallCache, ArtifactFactory artifactFactory) {
      this.syscallCache = syscallCache;
      this.artifactFactory = artifactFactory;
      this.rules = hintsRules.rules;
    }

    static HintsRules getRules(Path hintsFile) throws IOException {
      ImmutableList.Builder<Rule> rules = ImmutableList.builder();
      try (InputStream is = hintsFile.getInputStream()) {
        for (String line : CharStreams.readLines(new InputStreamReader(is, UTF_8))) {
          line = line.trim();
          if (line.isEmpty() || line.startsWith("#")) {
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

    /** Returns the "file" type hinted inclusions for a given path, caching results by path. */
    ImmutableList<Artifact> getFileLevelHintedInclusionsLegacy(Artifact path) {
      if (!path.getExecPathString().startsWith(ALLOWED_PREFIX)) {
        return ImmutableList.of();
      }
      return fileLevelHintsCache.get(path);
    }

    /**
     * Returns the "path" type hinted inclusions for the given paths. Callers are responsible for
     * caching.
     *
     * <p>Returns {@code null} when a skyframe restart is necessary.
     */
    @Nullable
    ImmutableSet<Artifact> getPathLevelHintedInclusions(
        ImmutableList<PathFragment> paths, Environment env)
        throws InterruptedException, IOException, NoSuchPackageException {
      ImmutableList<String> pathStrings =
          paths.stream()
              .map(PathFragment::getPathString)
              .filter(p -> p.startsWith(ALLOWED_PREFIX))
              .collect(toImmutableList());
      if (pathStrings.isEmpty()) {
        return ImmutableSet.of();
      }
      // Delay creation until we know we need one. Use a sorted set to make sure that the results
      // have a stable order and are unique.
      ImmutableSortedSet.Builder<Artifact> hints = null;
      List<ContainingPackageLookupValue.Key> rulePaths = new ArrayList<>(rules.size());
      List<String> findFilters = new ArrayList<>(rules.size());
      for (Rule rule : rules) {
        if (rule.type != Rule.Type.PATH) {
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
          hints = ImmutableSortedSet.orderedBy(Artifact.EXEC_PATH_COMPARATOR);
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
      SkyframeLookupResult containingPackageLookupValues = env.getValuesAndExceptions(rulePaths);
      if (env.valuesMissing() && !env.inErrorBubbling()) {
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
                  containingPackageLookupValues.getOrThrow(
                      relativePathKey, NoSuchPackageException.class);
        } catch (NoSuchPackageException e) {
          if (env.inErrorBubbling()) {
            throw e;
          }
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
          // TODO: b/290998109#comment60 - Convert to create GLOBS node in IncludeParser.
          globKeys.add(
              GlobValue.key(
                  containingPackageLookupValue.getContainingPackageName(),
                  containingPackageLookupValue.getContainingPackageRoot(),
                  pattern,
                  Globber.Operation.FILES,
                  relativePath.relativeTo(packageFragment)));
        } catch (InvalidGlobPatternException e) {
          env.getListener()
              .handle(Event.warn("Error parsing pattern " + pattern + " for " + relativePath));
        }
      }
      if (env.valuesMissing()) {
        return null;
      }
      SkyframeLookupResult globResults = env.getValuesAndExceptions(globKeys);
      if (env.valuesMissing() && !env.inErrorBubbling()) {
        return null;
      }
      for (GlobDescriptor globKey : globKeys) {
        PathFragment packageFragment = globKey.getPackageId().getPackageFragment();
        GlobValue globValue;
        try {
          globValue =
              (GlobValue)
                  globResults.getOrThrow(
                      globKey, IOException.class, BuildFileNotFoundException.class);
        } catch (IOException | BuildFileNotFoundException e) {
          if (env.inErrorBubbling()) {
            throw e;
          }
          logger.atWarning().withCause(e).log(
              "Unexpected exception when computing glob for %s" + " (prodaccess expired?)",
              globKey);
          continue;
        }
        for (PathFragment file : globValue.getMatches()) {
          hints.add(
              artifactFactory.getSourceArtifact(
                  packageFragment.getRelative(file), globKey.getPackageRoot()));
        }
      }
      if (env.valuesMissing()) {
        return null;
      }
      return hints == null ? ImmutableSet.of() : hints.build();
    }

    /**
     * Performs the work of matching a given path against the hints and returns the expanded paths.
     * The above {@link #getHintedInclusions} should be used in preference, but if the performance
     * impact of Skyframe restarts is untenable, this can be used as a fallback.
     */
    private ImmutableList<Artifact> getHintedInclusionsLegacy(Artifact artifact) {
      String pathString = artifact.getExecPath().getPathString();
      Root sourceRoot = artifact.getRoot().getRoot();
      // Delay creation until we know we need one. Use a TreeSet to make sure that the results are
      // sorted with a stable order and unique.
      Set<Path> hints = null;
      for (Rule rule : rules) {
        if (rule.type != Rule.Type.FILE) {
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
        Path root = sourceRoot.getRelative(relativePath);

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
          hints.addAll(new UnixGlob.Builder(root, syscallCache).addPattern(rule.findFilter).glob());
        } catch (UnixGlob.BadPattern | IOException e) {
          logger.atWarning().withCause(e).log("Error in hint expansion");
        }
      }
      if (hints == null || hints.isEmpty()) {
        return ImmutableList.of();
      }
      // Transform paths into source artifacts (all hints must be to source artifacts).
      ImmutableList.Builder<Artifact> result = ImmutableList.builderWithExpectedSize(hints.size());
      for (Path hint : hints) {
        result.add(
            Preconditions.checkNotNull(
                artifactFactory.getSourceArtifact(sourceRoot.relativize(hint), sourceRoot),
                "Missing source artifact, hint=%s, sourceRoot=%s, pathString=%s",
                hint,
                sourceRoot,
                pathString));
      }
      return result.build();
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
      if (!(o instanceof Inclusion that)) {
        return false;
      }
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
  static int skipWhitespace(byte[] chars, int pos, int end) {
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
  static List<Inclusion> processIncludes(Path file) throws IOException {
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
  static List<Inclusion> processIncludes(Object streamName, InputStream is) throws IOException {
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
  @Nullable
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
      @Nullable PlatformInfo grepIncludesExecutionPlatform,
      @Nullable SpawnIncludeScanner remoteIncludeScanner,
      boolean isOutputFile)
      throws IOException, ExecException, InterruptedException {
    Collection<Inclusion> inclusions;
    if (remoteIncludeScanner != null
        && grepIncludes != null
        && remoteIncludeScanner.shouldParseRemotely(file)) {
      inclusions =
          remoteIncludeScanner.extractInclusions(
              file,
              actionExecutionMetadata,
              actionExecutionContext,
              grepIncludes,
              grepIncludesExecutionPlatform,
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
          logger.atWarning().atMostEvery(1, TimeUnit.SECONDS).log(
              "Falling back on remote parsing of %s (cause %s)",
              actionExecutionContext.getInputPath(file), e.getMessage());
          inclusions =
              remoteIncludeScanner.extractInclusions(
                  file,
                  actionExecutionMetadata,
                  actionExecutionContext,
                  grepIncludes,
                  grepIncludesExecutionPlatform,
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

    static IncludesKeywordData importOrSwig(int pos) {
      return new IncludesKeywordData(pos, false, false);
    }

    static IncludesKeywordData hasInclude(int pos) {
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
   */
  protected boolean isValidInclusionKind(Kind kind) {
    return true;
  }

  /**
   * Returns inclusion object for non-standard inclusion cases or null if inclusion should be
   * ignored.
   */
  @Nullable
  protected Inclusion createOtherInclusion(String inclusionContent) {
    return null;
  }
}
