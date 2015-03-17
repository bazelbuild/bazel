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
import com.google.devtools.build.lib.cmdline.LabelValidator.BadLabelException;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

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
public abstract class TargetPattern {

  private static final Splitter SLASH_SPLITTER = Splitter.on('/');
  private static final Joiner SLASH_JOINER = Joiner.on('/');

  private static final Parser DEFAULT_PARSER = new Parser("");

  private final Type type;

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

  private TargetPattern(Type type) {
    // Don't allow inheritance outside this class.
    this.type = type;
  }

  /**
   * Return the type of the pattern. Examples include "below package" like "foo/..." and "single
   * target" like "//x:y".
   */
  public Type getType() {
    return type;
  }

  /**
   * Evaluates the current target pattern and returns the result.
   */
  public abstract <T> ResolvedTargets<T> eval(TargetPatternResolver<T> resolver)
      throws TargetParsingException, InterruptedException;

  private static final class SingleTarget extends TargetPattern {

    private final String targetName;

    private SingleTarget(String targetName) {
      super(Type.SINGLE_TARGET);
      this.targetName = targetName;
    }

    @Override
    public <T> ResolvedTargets<T> eval(TargetPatternResolver<T> resolver)
        throws TargetParsingException, InterruptedException {
      return resolver.getExplicitTarget(targetName);
    }
  }

  private static final class InterpretPathAsTarget extends TargetPattern {

    private final String path;

    private InterpretPathAsTarget(String path) {
      super(Type.PATH_AS_TARGET);
      this.path = normalize(path);
    }

    @Override
    public <T> ResolvedTargets<T> eval(TargetPatternResolver<T> resolver)
        throws TargetParsingException, InterruptedException {
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

      throw new TargetParsingException(
          "couldn't determine target from filename '" + path + "'");
    }
  }

  private static final class TargetsInPackage extends TargetPattern {

    private final String originalPattern;
    private final String pattern;
    private final String suffix;
    private final boolean isAbsolute;
    private final boolean rulesOnly;
    private final boolean checkWildcardConflict;

    private TargetsInPackage(String originalPattern, String pattern, String suffix,
        boolean isAbsolute, boolean rulesOnly, boolean checkWildcardConflict) {
      super(Type.TARGETS_IN_PACKAGE);
      this.originalPattern = originalPattern;
      this.pattern = pattern;
      this.suffix = suffix;
      this.isAbsolute = isAbsolute;
      this.rulesOnly = rulesOnly;
      this.checkWildcardConflict = checkWildcardConflict;
    }

    @Override
    public <T> ResolvedTargets<T> eval(TargetPatternResolver<T> resolver)
        throws TargetParsingException, InterruptedException {
      if (checkWildcardConflict) {
        ResolvedTargets<T> targets = getWildcardConflict(resolver);
        if (targets != null) {
          return targets;
        }
      }
      return resolver.getTargetsInPackage(originalPattern, removeSuffix(pattern, suffix),
          rulesOnly);
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

  private static final class TargetsBelowPackage extends TargetPattern {

    private final String originalPattern;
    private final String pathPrefix;
    private final boolean rulesOnly;

    private TargetsBelowPackage(String originalPattern, String pathPrefix, boolean rulesOnly) {
      super(Type.TARGETS_BELOW_PACKAGE);
      this.originalPattern = originalPattern;
      this.pathPrefix = pathPrefix;
      this.rulesOnly = rulesOnly;
    }

    @Override
    public <T> ResolvedTargets<T> eval(TargetPatternResolver<T> resolver)
        throws TargetParsingException, InterruptedException {
      return resolver.findTargetsBeneathDirectory(originalPattern, pathPrefix, rulesOnly);
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
          return new TargetsBelowPackage(originalPattern, realPackagePart, true);
        } else if (ALL_TARGETS_IN_SUFFIXES.contains(targetPart)) {
          return new TargetsBelowPackage(originalPattern, realPackagePart, false);
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


      if (isAbsolute || pattern.contains(":")) {
        String fullLabel = "//" + pattern;
        try {
          LabelValidator.validateAbsoluteLabel(fullLabel);
        } catch (BadLabelException e) {
          String error = "invalid target format '" + originalPattern + "': " + e.getMessage();
          throw new TargetParsingException(error);
        }
        return new SingleTarget(fullLabel);
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
      return new InterpretPathAsTarget(pattern);
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
    /** Targets below a package, eg "foo/...". */
    TARGETS_BELOW_PACKAGE,
    /** Target in a package, eg "foo:all". */
    TARGETS_IN_PACKAGE;
  }
}
