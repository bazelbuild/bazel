// Copyright 2014 The Bazel Authors. All rights reserved.
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
import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.LabelValidator.BadLabelException;
import com.google.devtools.build.lib.cmdline.LabelValidator.PackageAndTarget;
import com.google.devtools.build.lib.cmdline.PackageIdentifier.RepositoryName;
import com.google.devtools.build.lib.util.BatchCallback;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.util.StringUtilities;
import com.google.devtools.build.lib.vfs.PathFragment;

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
  private final String offset;

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

  private TargetPattern(Type type, String originalPattern, String offset) {
    // Don't allow inheritance outside this class.
    this.type = type;
    this.originalPattern = Preconditions.checkNotNull(originalPattern);
    this.offset = Preconditions.checkNotNull(offset);
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
   * Return the offset this target pattern was parsed with.
   */
  public String getOffset() {
    return offset;
  }

  /**
   * Evaluates the current target pattern and returns the result.
   */
  public <T, E extends Exception> void eval(
      TargetPatternResolver<T> resolver, BatchCallback<T, E> callback)
      throws TargetParsingException, E, InterruptedException {
    eval(resolver, ImmutableSet.<PathFragment>of(), callback);
  }

  /**
   * Evaluates the current target pattern, excluding targets under directories in
   * {@code excludedSubdirectories}, and returns the result.
   *
   * @throws IllegalArgumentException if {@code excludedSubdirectories} is nonempty and this
   *      pattern does not have type {@code Type.TARGETS_BELOW_DIRECTORY}.
   */
  public abstract <T, E extends Exception> void eval(
      TargetPatternResolver<T> resolver,
      ImmutableSet<PathFragment> excludedSubdirectories,
      BatchCallback<T, E> callback)
      throws TargetParsingException, E, InterruptedException;

  /**
   * Returns {@code true} iff this pattern has type {@code Type.TARGETS_BELOW_DIRECTORY} and
   * {@code directory} is contained by or equals this pattern's directory. For example,
   * returns {@code true} for {@code this = TargetPattern ("//...")} and {@code directory
   * = "foo")}.
   */
  public abstract boolean containsBelowDirectory(PackageIdentifier directory);

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
  public abstract PackageIdentifier getDirectory();

  /**
   * Returns {@code true} iff this pattern has type {@code Type.TARGETS_BELOW_DIRECTORY} or
   * {@code Type.TARGETS_IN_PACKAGE} and the target pattern suffix specified it should match
   * rules only.
   */
  public abstract boolean getRulesOnly();

  private static final class SingleTarget extends TargetPattern {

    private final String targetName;
    private final PackageIdentifier directory;

    private SingleTarget(
        String targetName, PackageIdentifier directory, String originalPattern, String offset) {
      super(Type.SINGLE_TARGET, originalPattern, offset);
      this.targetName = Preconditions.checkNotNull(targetName);
      this.directory = Preconditions.checkNotNull(directory);
    }

    @Override
    public <T, E extends Exception> void eval(
        TargetPatternResolver<T> resolver,
        ImmutableSet<PathFragment> excludedSubdirectories,
        BatchCallback<T, E> callback)
        throws TargetParsingException, E, InterruptedException {
      Preconditions.checkArgument(excludedSubdirectories.isEmpty(),
          "Target pattern \"%s\" of type %s cannot be evaluated with excluded subdirectories: %s.",
          getOriginalPattern(), getType(), excludedSubdirectories);
      callback.process(resolver.getExplicitTarget(label(targetName)).getTargets());
    }

    @Override
    public boolean containsBelowDirectory(PackageIdentifier directory) {
      return false;
    }

    @Override
    public PackageIdentifier getDirectory() {
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

    private InterpretPathAsTarget(String path, String originalPattern, String offset) {
      super(Type.PATH_AS_TARGET, originalPattern, offset);
      this.path = normalize(Preconditions.checkNotNull(path));
    }

    @Override
    public <T, E extends Exception> void eval(
        TargetPatternResolver<T> resolver,
        ImmutableSet<PathFragment> excludedSubdirectories,
        BatchCallback<T, E> callback)
        throws TargetParsingException, E, InterruptedException {
      Preconditions.checkArgument(excludedSubdirectories.isEmpty(),
          "Target pattern \"%s\" of type %s cannot be evaluated with excluded subdirectories: %s.",
          getOriginalPattern(), getType(), excludedSubdirectories);
      if (resolver.isPackage(PackageIdentifier.createInDefaultRepo(path))) {
        // User has specified a package name. lookout for default target.
        callback.process(resolver.getExplicitTarget(label("//" + path)).getTargets());
      } else {

        List<String> pieces = SLASH_SPLITTER.splitToList(path);

        // Interprets the label as a file target.  This loop stops as soon as the
        // first BUILD file is found (i.e. longest prefix match).
        for (int i = pieces.size() - 1; i > 0; i--) {
          String packageName = SLASH_JOINER.join(pieces.subList(0, i));
          if (resolver.isPackage(PackageIdentifier.createInDefaultRepo(packageName))) {
            String targetName = SLASH_JOINER.join(pieces.subList(i, pieces.size()));
            callback.process(
                resolver
                    .getExplicitTarget(label("//" + packageName + ":" + targetName))
                    .getTargets());
            return;
          }
        }

        throw new TargetParsingException("couldn't determine target from filename '" + path + "'");
      }
    }

    @Override
    public boolean containsBelowDirectory(PackageIdentifier directory) {
      return false;
    }

    @Override
    public PackageIdentifier getDirectory() {
      int lastSlashIndex = path.lastIndexOf('/');
      // The package name cannot be illegal because we verified it during target parsing
      return PackageIdentifier.createInDefaultRepo(
          lastSlashIndex < 0 ? "" : path.substring(0, lastSlashIndex));
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

    private final PackageIdentifier packageIdentifier;
    private final String suffix;
    private final boolean isAbsolute;
    private final boolean rulesOnly;
    private final boolean checkWildcardConflict;

    private TargetsInPackage(String originalPattern, String offset,
        PackageIdentifier packageIdentifier, String suffix, boolean isAbsolute, boolean rulesOnly,
        boolean checkWildcardConflict) {
      super(Type.TARGETS_IN_PACKAGE, originalPattern, offset);
      this.packageIdentifier = packageIdentifier;
      this.suffix = Preconditions.checkNotNull(suffix);
      this.isAbsolute = isAbsolute;
      this.rulesOnly = rulesOnly;
      this.checkWildcardConflict = checkWildcardConflict;
    }

    @Override
    public <T, E extends Exception> void eval(
        TargetPatternResolver<T> resolver,
        ImmutableSet<PathFragment> excludedSubdirectories,
        BatchCallback<T, E> callback)
        throws TargetParsingException, E, InterruptedException {
      Preconditions.checkArgument(excludedSubdirectories.isEmpty(),
          "Target pattern \"%s\" of type %s cannot be evaluated with excluded subdirectories: %s.",
          getOriginalPattern(), getType(), excludedSubdirectories);
      if (checkWildcardConflict) {
        ResolvedTargets<T> targets = getWildcardConflict(resolver);
        if (targets != null) {
          callback.process(targets.getTargets());
          return;
        }
      }

      callback.process(
          resolver
              .getTargetsInPackage(getOriginalPattern(), packageIdentifier, rulesOnly)
              .getTargets());
    }

    @Override
    public boolean containsBelowDirectory(PackageIdentifier directory) {
      return false;
    }

    @Override
    public PackageIdentifier getDirectory() {
      return packageIdentifier;
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
          && packageIdentifier.equals(that.packageIdentifier) && suffix.equals(that.suffix);
    }

    @Override
    public int hashCode() {
      return Objects.hash(getType(), getOriginalPattern(), packageIdentifier, suffix, isAbsolute,
          rulesOnly, checkWildcardConflict);
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

      T target;
      Label label;
      try {
        label = Label.create(packageIdentifier, suffix);
        target = resolver.getTargetOrNull(label);
      } catch (LabelSyntaxException e) {
        return null;
      }

      if (target != null) {
        resolver.warn(String.format("The target pattern '%s' is ambiguous: '%s' is " +
                                    "both a wildcard, and the name of an existing %s; " +
                                    "using the latter interpretation",
                                    getOriginalPattern(), ":" + suffix,
                                    resolver.getTargetKind(target)));
        try {
          return resolver.getExplicitTarget(label);
        } catch (TargetParsingException e) {
          throw new IllegalStateException(
              "getTargetOrNull() returned non-null, so target should exist", e);
        }
      }
      return null;
    }
  }

  private static final class TargetsBelowDirectory extends TargetPattern {

    private final PackageIdentifier directory;
    private final boolean rulesOnly;

    private TargetsBelowDirectory(
        String originalPattern, String offset, PackageIdentifier directory, boolean rulesOnly) {
      super(Type.TARGETS_BELOW_DIRECTORY, originalPattern, offset);
      this.directory = Preconditions.checkNotNull(directory);
      this.rulesOnly = rulesOnly;
    }

    @Override
    public <T, E extends Exception> void eval(
        TargetPatternResolver<T> resolver,
        ImmutableSet<PathFragment> excludedSubdirectories,
        BatchCallback<T, E> callback)
        throws TargetParsingException, E, InterruptedException {
      resolver.findTargetsBeneathDirectory(
          directory.getRepository(),
          getOriginalPattern(),
          directory.getPackageFragment().getPathString(),
          rulesOnly,
          excludedSubdirectories,
          callback);
    }

    @Override
    public boolean containsBelowDirectory(PackageIdentifier containedDirectory) {
      // Note that merely checking to see if the directory startsWith the TargetsBelowDirectory's
      // directory is insufficient. "food" begins with "foo", but "//foo/..." does not contain
      // "//food/...".
      return containedDirectory.getRepository().equals(directory.getRepository())
          && containedDirectory.getPackageFragment().startsWith(directory.getPackageFragment());
    }

    @Override
    public PackageIdentifier getDirectory() {
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
      RepositoryName repository = PackageIdentifier.DEFAULT_REPOSITORY_NAME;
      if (includesRepo) {
        int pkgStart = pattern.indexOf("//");
        if (pkgStart < 0) {
          throw new TargetParsingException("Couldn't find package in target " + pattern);
        }
        try {
          repository = RepositoryName.create(pattern.substring(0, pkgStart));
        } catch (LabelSyntaxException e) {
          throw new TargetParsingException(e.getMessage());
        }

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
        PackageIdentifier packageIdentifier;
        try {
          packageIdentifier = PackageIdentifier.parse(
              repository.getName() + "//" + realPackagePart);
        } catch (LabelSyntaxException e) {
          throw new TargetParsingException(
              "Invalid package name '" + realPackagePart + "': " + e.getMessage());
        }
        if (targetPart.isEmpty() || ALL_RULES_IN_SUFFIXES.contains(targetPart)) {
          return new TargetsBelowDirectory(
              originalPattern, relativeDirectory, packageIdentifier, true);
        } else if (ALL_TARGETS_IN_SUFFIXES.contains(targetPart)) {
          return new TargetsBelowDirectory(
              originalPattern, relativeDirectory, packageIdentifier, false);
        }
      }

      if (ALL_RULES_IN_SUFFIXES.contains(targetPart)) {
        PackageIdentifier packageIdentifier;
        try {
          packageIdentifier = PackageIdentifier.parse(repository.getName() + "//" + packagePart);
        } catch (LabelSyntaxException e) {
          throw new TargetParsingException(
              "Invalid package name '" + packagePart + "': " + e.getMessage());
        }
        return new TargetsInPackage(originalPattern, relativeDirectory, packageIdentifier,
            targetPart, isAbsolute, true, true);
      }

      if (ALL_TARGETS_IN_SUFFIXES.contains(targetPart)) {
        PackageIdentifier packageIdentifier;
        try {
          packageIdentifier = PackageIdentifier.parse(repository.getName() + "//" + packagePart);
        } catch (LabelSyntaxException e) {
          throw new TargetParsingException(
              "Invalid package name '" + packagePart + "': " + e.getMessage());
        }
        return new TargetsInPackage(originalPattern, relativeDirectory, packageIdentifier,
            targetPart, isAbsolute, false, true);
      }

      if (includesRepo || isAbsolute || pattern.contains(":")) {
        PackageIdentifier packageIdentifier;
        String fullLabel = repository.getName() + "//" + pattern;
        try {
          PackageAndTarget packageAndTarget = LabelValidator.validateAbsoluteLabel(fullLabel);
          packageIdentifier = PackageIdentifier.create(repository,
              new PathFragment(packageAndTarget.getPackageName()));
        } catch (BadLabelException e) {
          String error = "invalid target format '" + originalPattern + "': " + e.getMessage();
          throw new TargetParsingException(error);
        }
        return new SingleTarget(
            fullLabel, packageIdentifier, originalPattern, relativeDirectory);
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
      try {
        PackageIdentifier.parse("//" + packageName);
      } catch (LabelSyntaxException e) {
        throw new TargetParsingException(
            "Bad target pattern '" + originalPattern + "': " + e.getMessage());
      }
      return new InterpretPathAsTarget(pattern, originalPattern, relativeDirectory);
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

  // Parse 'label' as a Label, mapping LabelSyntaxException into
  // TargetParsingException.
  private static Label label(String label) throws TargetParsingException {
    try {
      return Label.parseAbsolute(label);
    } catch (LabelSyntaxException e) {
      throw new TargetParsingException("invalid target format: '"
          + StringUtilities.sanitizeControlChars(label) + "'; "
          + StringUtilities.sanitizeControlChars(e.getMessage()));
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
