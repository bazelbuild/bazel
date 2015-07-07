// Copyright 2014 Google Inc. All rights reserved.
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

package com.google.devtools.build.lib.cmdline;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.LabelValidator.BadLabelException;
import com.google.devtools.build.lib.cmdline.LabelValidator.PackageAndTarget;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Objects;

import javax.annotation.concurrent.Immutable;

/**
 * Represents a target pattern. Target patterns are a generalization of labels to include
 * wildcards for finding all packages recursively beneath some root, and for finding all targets
 * within a package.
 *
 * <p>Note that this class does not handle negative patterns ("-//foo/bar"); these must be handled
 * one level up. In particular, the query language comes with built-in support for negative
 * patterns.
 *
 * <p>In order to resolve target patterns, you need an implementation of {@link
 * TargetPatternResolver}. This class is thread-safe if the corresponding instance is thread-safe.
 *
 * <p>See lib/blaze/commands/target-syntax.txt for details.
 */
public abstract class TargetPattern implements Serializable {

  private static final Splitter SLASH_SPLITTER = Splitter.on('/');
  private static final Joiner SLASH_JOINER = Joiner.on('/');

  private static final Parser DEFAULT_PARSER = new Parser("");

  private final Type type;
  private final String originalPattern;

  /**
   * Returns a parser with no offset. Note that the Parser class is immutable, so this method may
   * return the same instance on subsequent calls.
   */
  public static Parser defaultParser() {
    return DEFAULT_PARSER;
  }

  private static String removeSuffix(String s, String suffix) {
    if (s.endsWith(suffix)) {
      return s.substring(0, s.length() - suffix.length());
    } else {
      throw new IllegalArgumentException(s + ", " + suffix);
    }
  }

  /**
   * Normalizes the given relative path by resolving {@code //}, {@code /./} and {@code x/../}
   * pieces. Note that leading {@code ".."} segments are not removed, so the returned string can
   * have leading {@code ".."} segments.
   *
   * @throws IllegalArgumentException if the path is absolute, i.e. starts with a @{code '/'}
   */
  @VisibleForTesting
  static String normalize(String path) {
    Preconditions.checkArgument(!path.startsWith("/"));
    Preconditions.checkArgument(!path.startsWith("@"));
    Iterator<String> it = SLASH_SPLITTER.split(path).iterator();
    List<String> pieces = new ArrayList<>();
    while (it.hasNext()) {
      String piece = it.next();
      if (".".equals(piece) || piece.isEmpty()) {
        continue;
      }
      if ("..".equals(piece)) {
        if (pieces.isEmpty()) {
          pieces.add(piece);
          continue;
        }
        String predecessor = pieces.remove(pieces.size() - 1);
        if ("..".equals(predecessor)) {
          pieces.add(piece);
          pieces.add(piece);
        }
        continue;
      }
      pieces.add(piece);
    }
    return SLASH_JOINER.join(pieces);
  }

  private TargetPattern(Type type, String originalPattern) {
    // Don't allow inheritance outside this class.
    this.type = type;
    this.originalPattern = Preconditions.checkNotNull(originalPattern);
  }

  /**
   * Return the type of the pattern. Examples include "below directory" like "foo/..." and "single
   * target" like "//x:y".
   */
  public Type getType() {
    return type;
  }

  /**
   * Return the string that was parsed into this pattern.
   */
  public String getOriginalPattern() {
    return originalPattern;
  }

  /**
   * Evaluates the current target pattern and returns the result.
   */
  public <T> ResolvedTargets<T> eval(TargetPatternResolver<T> resolver)
      throws TargetParsingException, InterruptedException {
    return eval(resolver, ImmutableSet.<String>of());
  }

  /**
   * Evaluates the current target pattern, excluding targets under directories in
   * {@code excludedSubdirectories}, and returns the result.
   *
   * @throws IllegalArgumentException if {@code excludedSubdirectories} is nonempty and this
   *      pattern does not have type {@code Type.TARGETS_BELOW_DIRECTORY}.
   */
  public abstract <T> ResolvedTargets<T> eval(TargetPatternResolver<T> resolver,
      ImmutableSet<String> excludedSubdirectories)
      throws TargetParsingException, InterruptedException;

  /**
   * Returns {@code true} iff this pattern has type {@code Type.TARGETS_BELOW_DIRECTORY} and
   * {@param directory} is contained by or equals this pattern's directory. For example,
   * returns {@code true} for {@code this = TargetPattern ("//...")} and {@code directory
   * = "foo")}.
   */
  public abstract boolean containsBelowDirectory(String directory);

  /**
   * Shorthand for {@code containsBelowDirectory(containedPattern.getDirectory())}.
   */
  public boolean containsBelowDirectory(TargetPattern containedPattern) {
    return containsBelowDirectory(containedPattern.getDirectory());
  }

  /**
   * Returns the most specific containing directory of the patterns that could be matched by this
   * pattern.
   *
   * <p>For patterns of type {@code Type.TARGETS_BELOW_DIRECTORY}, this returns the referred-to
   * directory. For example, for "//foo/bar/...", this returns "foo/bar".
   *
   * <p>The returned value always has no leading "//" and no trailing "/".
   */
  public abstract String getDirectory();

  /**
   * Returns {@code true} iff this pattern has type {@code Type.TARGETS_BELOW_DIRECTORY} or
   * {@code Type.TARGETS_IN_PACKAGE} and the target pattern suffix specified it should match
   * rules only.
   */
  public abstract boolean getRulesOnly();

  private static final class SingleTarget extends TargetPattern {

    private final String targetName;
    private final String directory;

    private SingleTarget(String targetName, String directory, String originalPattern) {
      super(Type.SINGLE_TARGET, originalPattern);
      this.targetName = Preconditions.checkNotNull(targetName);
      this.directory = Preconditions.checkNotNull(directory);
    }

    @Override
    public <T> ResolvedTargets<T> eval(TargetPatternResolver<T> resolver,
        ImmutableSet<String> excludedSubdirectories)
        throws TargetParsingException, InterruptedException {
      Preconditions.checkArgument(excludedSubdirectories.isEmpty(),
          "Target pattern \"%s\" of type %s cannot be evaluated with excluded subdirectories: %s.",
          getOriginalPattern(), getType(), excludedSubdirectories);
      return resolver.getExplicitTarget(targetName);
    }

    @Override
    public boolean containsBelowDirectory(String directory) {
      return false;
    }

    @Override
    public String getDirectory() {
      return directory;
    }

    @Override
    public boolean getRulesOnly() {
      return false;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (!(o instanceof SingleTarget)) {
        return false;
      }
      SingleTarget that = (SingleTarget) o;
      return targetName.equals(that.targetName) && directory.equals(that.directory);
    }

    @Override
    public int hashCode() {
      return Objects.hash(getType(), targetName, directory);
    }
  }

  private static final class InterpretPathAsTarget extends TargetPattern {

    private final String path;

    private InterpretPathAsTarget(String path, String originalPattern) {
      super(Type.PATH_AS_TARGET, originalPattern);
      this.path = normalize(Preconditions.checkNotNull(path));
    }

    @Override
    public <T> ResolvedTargets<T> eval(TargetPatternResolver<T> resolver,
        ImmutableSet<String> excludedSubdirectories)
        throws TargetParsingException, InterruptedException {
      Preconditions.checkArgument(excludedSubdirectories.isEmpty(),
          "Target pattern \"%s\" of type %s cannot be evaluated with excluded subdirectories: %s.",
          getOriginalPattern(), getType(), excludedSubdirectories);
      if (resolver.isPackage(path)) {
        // User has specified a package name. lookout for default target.
        return resolver.getExplicitTarget("//" + path);
      }

      List<String> pieces = SLASH_SPLITTER.splitToList(path);

      // Interprets the label as a file target.  This loop stops as soon as the
      // first BUILD file is found (i.e. longest prefix match).
      for (int i = pieces.size() - 1; i > 0; i--) {
        String packageName = SLASH_JOINER.join(pieces.subList(0, i));
        if (resolver.isPackage(packageName)) {
          String targetName = SLASH_JOINER.join(pieces.subList(i, pieces.size()));
          return resolver.getExplicitTarget("//" + packageName + ":" + targetName);
        }
      }

      throw new TargetParsingException("couldn't determine target from filename '" + path + "'");
    }

    @Override
    public boolean containsBelowDirectory(String directory) {
      return false;
    }

    @Override
    public String getDirectory() {
      int lastSlashIndex = path.lastIndexOf('/');
      return lastSlashIndex < 0 ? "" : path.substring(0, lastSlashIndex);
    }

    @Override
    public boolean getRulesOnly() {
      return false;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (!(o instanceof InterpretPathAsTarget)) {
        return false;
      }
      InterpretPathAsTarget that = (InterpretPathAsTarget) o;
      return path.equals(that.path);
    }

    @Override
    public int hashCode() {
      return Objects.hash(getType(), path);
    }
  }

  private static final class TargetsInPackage extends TargetPattern {

    private final String pattern;
    private final String suffix;
    private final boolean isAbsolute;
    private final boolean rulesOnly;
    private final boolean checkWildcardConflict;

    private TargetsInPackage(String originalPattern, String pattern, String suffix,
        boolean isAbsolute, boolean rulesOnly, boolean checkWildcardConflict) {
      super(Type.TARGETS_IN_PACKAGE, originalPattern);
      this.pattern = Preconditions.checkNotNull(pattern);
      this.suffix = Preconditions.checkNotNull(suffix);
      this.isAbsolute = isAbsolute;
      this.rulesOnly = rulesOnly;
      this.checkWildcardConflict = checkWildcardConflict;
    }

    @Override
    public <T> ResolvedTargets<T> eval(TargetPatternResolver<T> resolver,
        ImmutableSet<String> excludedSubdirectories)
        throws TargetParsingException, InterruptedException {
      Preconditions.checkArgument(excludedSubdirectories.isEmpty(),
          "Target pattern \"%s\" of type %s cannot be evaluated with excluded subdirectories: %s.",
          getOriginalPattern(), getType(), excludedSubdirectories);
      if (checkWildcardConflict) {
        ResolvedTargets<T> targets = getWildcardConflict(resolver);
        if (targets != null) {
          return targets;
        }
      }
      return resolver.getTargetsInPackage(getOriginalPattern(), removeSuffix(pattern, suffix),
          rulesOnly);
    }

    @Override
    public boolean containsBelowDirectory(String directory) {
      return false;
    }

    @Override
    public String getDirectory() {
      return removeSuffix(pattern, suffix);
    }

    @Override
    public boolean getRulesOnly() {
      return rulesOnly;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (!(o instanceof TargetsInPackage)) {
        return false;
      }
      TargetsInPackage that = (TargetsInPackage) o;
      return isAbsolute == that.isAbsolute && rulesOnly == that.rulesOnly
          && checkWildcardConflict == that.checkWildcardConflict
          && getOriginalPattern().equals(that.getOriginalPattern())
          && pattern.equals(that.pattern) && suffix.equals(that.suffix);
    }

    @Override
    public int hashCode() {
      return Objects.hash(getType(), getOriginalPattern(), pattern, suffix, isAbsolute, rulesOnly,
          checkWildcardConflict);
    }

    /**
     * There's a potential ambiguity if '//foo/bar:all' refers to an actual target. In this case, we
     * use the the target but print a warning.
     *
     * @return the Target corresponding to the given pattern, if the pattern is absolute and there
     *         is such a target. Otherwise, return null.
     */
    private <T> ResolvedTargets<T> getWildcardConflict(TargetPatternResolver<T> resolver)
        throws InterruptedException {
      if (!isAbsolute) {
        return null;
      }

      T target = resolver.getTargetOrNull("//" + pattern);
      if (target != null) {
        String name = pattern.lastIndexOf(':') != -1
            ? pattern.substring(pattern.lastIndexOf(':') + 1)
            : pattern.substring(pattern.lastIndexOf('/') + 1);
        resolver.warn(String.format("The Blaze target pattern '%s' is ambiguous: '%s' is " +
                                    "both a wildcard, and the name of an existing %s; " +
                                    "using the latter interpretation",
                                    "//" + pattern, ":" + name,
                                    resolver.getTargetKind(target)));
        try {
          return resolver.getExplicitTarget("//" + pattern);
        } catch (TargetParsingException e) {
          throw new IllegalStateException(
              "getTargetOrNull() returned non-null, so target should exist", e);
        }
      }
      return null;
    }
  }

  private static final class TargetsBelowDirectory extends TargetPattern {

    private final String directory;
    private final boolean rulesOnly;

    private TargetsBelowDirectory(String originalPattern, String directory, boolean rulesOnly) {
      super(Type.TARGETS_BELOW_DIRECTORY, originalPattern);
      this.directory = Preconditions.checkNotNull(directory);
      this.rulesOnly = rulesOnly;
    }

    @Override
    public <T> ResolvedTargets<T> eval(TargetPatternResolver<T> resolver,
        ImmutableSet<String> excludedSubdirectories)
        throws TargetParsingException, InterruptedException {
      return resolver.findTargetsBeneathDirectory(getOriginalPattern(), directory, rulesOnly,
          excludedSubdirectories);
    }

    @Override
    public boolean containsBelowDirectory(String containedDirectory) {
      // Note that merely checking to see if the directory startsWith the TargetsBelowDirectory's
      // directory is insufficient. "food" begins with "foo", but "//foo/..." does not contain
      // "//food/...".
      return directory.isEmpty() || (containedDirectory + "/").startsWith(directory + "/");
    }

    @Override
    public String getDirectory() {
      return directory;
    }

    @Override
    public boolean getRulesOnly() {
      return rulesOnly;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (!(o instanceof TargetsBelowDirectory)) {
        return false;
      }
      TargetsBelowDirectory that = (TargetsBelowDirectory) o;
      return rulesOnly == that.rulesOnly && getOriginalPattern().equals(that.getOriginalPattern())
          && directory.equals(that.directory);
    }

    @Override
    public int hashCode() {
      return Objects.hash(getType(), getOriginalPattern(), directory, rulesOnly);
    }
  }

  @Immutable
  public static final class Parser {
    // TODO(bazel-team): Merge the Label functionality that requires similar constants into this
    // class.
    /**
     * The set of target-pattern suffixes which indicate wildcards over all <em>rules</em> in a
     * single package.
     */
    private static final List<String> ALL_RULES_IN_SUFFIXES = ImmutableList.of(
        "all");

    /**
     * The set of target-pattern suffixes which indicate wildcards over all <em>targets</em> in a
     * single package.
     */
    private static final List<String> ALL_TARGETS_IN_SUFFIXES = ImmutableList.of(
        "*",
        "all-targets");

    private static final List<String> SUFFIXES;

    static {
      SUFFIXES = ImmutableList.<String>builder()
          .addAll(ALL_RULES_IN_SUFFIXES)
          .addAll(ALL_TARGETS_IN_SUFFIXES)
          .add("/...")
          .build();
    }

    /**
     * Returns whether the given pattern is simple, i.e., not starting with '-' and using none of
     * the target matching suffixes.
     */
    public static boolean isSimpleTargetPattern(String pattern) {
      if (pattern.startsWith("-")) {
        return false;
      }

      for (String suffix : SUFFIXES) {
        if (pattern.endsWith(":" + suffix)) {
          return false;
        }
      }
      return true;
    }

    /**
     * Directory prefix to use when resolving relative labels (rather than absolute ones). For
     * example, if the working directory is "<workspace root>/foo", then this should be "foo",
     * which will make patterns such as "bar:bar" be resolved as "//foo/bar:bar". This makes the
     * command line a bit more convenient to use.
     */
    private final String relativeDirectory;

    /**
     * Creates a new parser with the given offset for relative patterns.
     */
    public Parser(String relativeDirectory) {
      this.relativeDirectory = relativeDirectory;
    }

    /**
     * Parses the given pattern, and throws an exception if the pattern is invalid.
     *
     * @return a target pattern corresponding to the pattern parsed
     * @throws TargetParsingException if the pattern is invalid
     */
    public TargetPattern parse(String pattern) throws TargetParsingException {
      // The structure of this method is by cases, according to the usage string
      // constant (see lib/blaze/commands/target-syntax.txt).

      String originalPattern = pattern;
      final boolean includesRepo = pattern.startsWith("@");
      String repoName = "";
      if (includesRepo) {
        int pkgStart = pattern.indexOf("//");
        if (pkgStart < 0) {
          throw new TargetParsingException("Couldn't find package in target " + pattern);
        }
        repoName = pattern.substring(0, pkgStart);
        pattern = pattern.substring(pkgStart);
      }
      final boolean isAbsolute = pattern.startsWith("//");

      // We now absolutize non-absolute target patterns.
      pattern = isAbsolute ? pattern.substring(2) : absolutize(pattern);
      // Check for common errors.
      if (pattern.startsWith("/")) {
        throw new TargetParsingException("not a relative path or label: '" + pattern + "'");
      }
      if (pattern.isEmpty()) {
        throw new TargetParsingException("the empty string is not a valid target");
      }

      // Transform "/BUILD" suffix into ":BUILD" to accept //foo/bar/BUILD
      // syntax as a synonym to //foo/bar:BUILD.
      if (pattern.endsWith("/BUILD")) {
        pattern = pattern.substring(0, pattern.length() - 6) + ":BUILD";
      }

      int colonIndex = pattern.lastIndexOf(':');
      String packagePart = colonIndex < 0 ? pattern : pattern.substring(0, colonIndex);
      String targetPart = colonIndex < 0 ? "" : pattern.substring(colonIndex + 1);

      if (packagePart.equals("...")) {
        packagePart = "/...";  // special case this for easier parsing
      }

      if (packagePart.endsWith("/")) {
        throw new TargetParsingException("The package part of '" + originalPattern
            + "' should not end in a slash");
      }

      if (packagePart.endsWith("/...")) {
        String realPackagePart = removeSuffix(packagePart, "/...");
        if (targetPart.isEmpty() || ALL_RULES_IN_SUFFIXES.contains(targetPart)) {
          return new TargetsBelowDirectory(originalPattern, realPackagePart, true);
        } else if (ALL_TARGETS_IN_SUFFIXES.contains(targetPart)) {
          return new TargetsBelowDirectory(originalPattern, realPackagePart, false);
        }
      }

      if (ALL_RULES_IN_SUFFIXES.contains(targetPart)) {
        return new TargetsInPackage(
            originalPattern, pattern, ":" + targetPart, isAbsolute, true, true);
      }

      if (ALL_TARGETS_IN_SUFFIXES.contains(targetPart)) {
        return new TargetsInPackage(
            originalPattern, pattern, ":" + targetPart, isAbsolute, false, true);
      }


      if (includesRepo || isAbsolute || pattern.contains(":")) {
        PackageAndTarget packageAndTarget;
        String fullLabel = repoName + "//" + pattern;
        try {
          packageAndTarget = LabelValidator.validateAbsoluteLabel(fullLabel);
        } catch (BadLabelException e) {
          String error = "invalid target format '" + originalPattern + "': " + e.getMessage();
          throw new TargetParsingException(error);
        }
        return new SingleTarget(fullLabel, packageAndTarget.getPackageName(), originalPattern);
      }

      // This is a stripped-down version of interpretPathAsTarget that does no I/O.  We have a basic
      // relative path. e.g. "foo/bar/Wiz.java". The strictest correct check we can do here (without
      // I/O) is just to ensure that there is *some* prefix that is a valid package-name. It's
      // sufficient to test the first segment. This is really a rather weak check; perhaps we should
      // just eliminate it.
      int slashIndex = pattern.indexOf('/');
      String packageName = pattern;
      if (slashIndex > 0) {
        packageName = pattern.substring(0, slashIndex);
      }
      String errorMessage = LabelValidator.validatePackageName(packageName);
      if (errorMessage != null) {
        throw new TargetParsingException("Bad target pattern '" + originalPattern + "': " +
            errorMessage);
      }
      return new InterpretPathAsTarget(pattern, originalPattern);
    }

    /**
     * Absolutizes the target pattern to the offset.
     * Patterns starting with "/" are absolute and not modified.
     *
     * If the offset is "foo":
     *   absolutize(":bar") --> "foo:bar"
     *   absolutize("bar") --> "foo/bar"
     *   absolutize("/biz/bar") --> "biz/bar" (absolute)
     *   absolutize("biz:bar") --> "foo/biz:bar"
     *
     * @param pattern The target pattern to parse.
     * @return the pattern, absolutized to the offset if approprate.
     */
    private String absolutize(String pattern) {
      if (relativeDirectory.isEmpty() || pattern.startsWith("/")) {
        return pattern;
      }

      // It seems natural to use {@link PathFragment#getRelative()} here,
      // but it doesn't work when the pattern starts with ":".
      // "foo".getRelative(":all") would return "foo/:all", where we
      // really want "foo:all".
      return pattern.startsWith(":")
          ? relativeDirectory + pattern
          : relativeDirectory + "/" + pattern;
    }
  }

  /**
   * The target pattern type (targets below package, in package, explicit target, etc.)
   */
  public enum Type {
    /** A path interpreted as a target, eg "foo/bar/baz" */
    PATH_AS_TARGET,
    /** An explicit target, eg "//foo:bar." */
    SINGLE_TARGET,
    /** Targets below a directory, eg "foo/...". */
    TARGETS_BELOW_DIRECTORY,
    /** Target in a package, eg "foo:all". */
    TARGETS_IN_PACKAGE;
  }
}
