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

import static com.google.common.util.concurrent.Futures.immediateCancelledFuture;
import static com.google.common.util.concurrent.Futures.immediateVoidFuture;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.ListeningExecutorService;
import com.google.devtools.build.lib.cmdline.LabelValidator.BadLabelException;
import com.google.devtools.build.lib.cmdline.LabelValidator.PackageAndTarget;
import com.google.devtools.build.lib.concurrent.BatchCallback;
import com.google.devtools.build.lib.server.FailureDetails.TargetPatterns;
import com.google.devtools.build.lib.supplier.InterruptibleSupplier;
import com.google.devtools.build.lib.util.StringUtilities;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.errorprone.annotations.CheckReturnValue;
import com.google.errorprone.annotations.CompileTimeConstant;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Pattern;
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

  private static final Parser DEFAULT_PARSER = new Parser(PathFragment.EMPTY_FRAGMENT);

  private final String originalPattern;
  private final PathFragment offset;

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

  private TargetPattern(String originalPattern, PathFragment offset) {
    // Don't allow inheritance outside this class.
    this.originalPattern = Preconditions.checkNotNull(originalPattern);
    this.offset = Preconditions.checkNotNull(offset);
  }

  /**
   * Return the type of the pattern. Examples include "below directory" like "foo/..." and "single
   * target" like "//x:y".
   */
  public abstract Type getType();

  /**
   * Return the string that was parsed into this pattern.
   */
  public String getOriginalPattern() {
    return originalPattern;
  }

  /** Returns the offset this target pattern was parsed with. */
  public PathFragment getOffset() {
    return offset;
  }

  /**
   * Evaluates the current target pattern, excluding targets under directories in both {@code
   * ignoredSubdirectories} and {@code excludedSubdirectories}, and returns the result.
   *
   * @throws IllegalArgumentException if either {@code ignoredSubdirectories} or {@code
   *     excludedSubdirectories} is nonempty and this pattern does not have type {@code
   *     Type.TARGETS_BELOW_DIRECTORY}.
   */
  public abstract <T, E extends Exception> void eval(
      TargetPatternResolver<T> resolver,
      InterruptibleSupplier<ImmutableSet<PathFragment>> ignoredSubdirectories,
      ImmutableSet<PathFragment> excludedSubdirectories,
      BatchCallback<T, E> callback,
      Class<E> exceptionClass)
      throws TargetParsingException, E, InterruptedException;

  /**
   * Evaluates this {@link TargetPattern} synchronously, feeding the result to the given {@code
   * callback}, and then returns an appropriate immediate {@link ListenableFuture}.
   *
   * <p>If the returned {@link ListenableFuture}'s {@link ListenableFuture#get} throws an {@link
   * ExecutionException}, the cause will be an instance of either {@link TargetParsingException} or
   * the given {@code exceptionClass}.
   */
  public final <T, E extends Exception> ListenableFuture<Void> evalAdaptedForAsync(
      TargetPatternResolver<T> resolver,
      InterruptibleSupplier<ImmutableSet<PathFragment>> ignoredSubdirectories,
      ImmutableSet<PathFragment> excludedSubdirectories,
      BatchCallback<T, E> callback,
      Class<E> exceptionClass) {
    try {
      eval(resolver, ignoredSubdirectories, excludedSubdirectories, callback, exceptionClass);
      return Futures.immediateFuture(null);
    } catch (TargetParsingException e) {
      return Futures.immediateFailedFuture(e);
    } catch (InterruptedException e) {
      return immediateCancelledFuture();
    } catch (Exception e) {
      if (exceptionClass.isInstance(e)) {
        return Futures.immediateFailedFuture(exceptionClass.cast(e));
      }
      throw new IllegalStateException(e);
    }
  }

  /**
   * Returns a {@link ListenableFuture} representing the asynchronous evaluation of this {@link
   * TargetPattern} that feeds the results to the given {@code callback}.
   *
   * <p>If the returned {@link ListenableFuture}'s {@link ListenableFuture#get} throws an {@link
   * ExecutionException}, the cause will be an instance of either {@link TargetParsingException} or
   * the given {@code exceptionClass}.
   */
  public <T, E extends Exception> ListenableFuture<Void> evalAsync(
      TargetPatternResolver<T> resolver,
      InterruptibleSupplier<ImmutableSet<PathFragment>> ignoredSubdirectories,
      ImmutableSet<PathFragment> excludedSubdirectories,
      BatchCallback<T, E> callback,
      Class<E> exceptionClass,
      ListeningExecutorService executor) {
    return evalAdaptedForAsync(
        resolver, ignoredSubdirectories, excludedSubdirectories, callback, exceptionClass);
  }

  /**
   * For patterns of type {@link Type#PATH_AS_TARGET}, returns the path in question.
   *
   * <p>The interpretation of this path, of course, depends on the existence of packages.
   * See {@link InterpretPathAsTarget#eval}.
   */
  public String getPathForPathAsTarget() {
    throw new IllegalStateException();
  }

  /** For patterns of type {@link Type#SINGLE_TARGET}, returns the target path. */
  public String getSingleTargetPath() {
    throw new IllegalStateException();
  }

  /**
   * For patterns of type {@link Type#SINGLE_TARGET}, {@link Type#TARGETS_IN_PACKAGE}, and {@link
   * Type#TARGETS_BELOW_DIRECTORY}, returns the {@link PackageIdentifier} of the pattern.
   *
   * <p>Note that we are using the {@link PackageIdentifier} type as a convenience; there may not
   * actually be a package corresponding to this directory!
   *
   * <p>Examples:
   *
   * <ul>
   *   <li>For pattern {@code //foo:bar}, returns package identifier {@code //foo}.
   *   <li>For pattern {@code //foo:all}, returns package identifier {@code //foo}.
   *   <li>For pattern {@code //foo/...}, returns package identifier {@code //foo}.
   * </ul>
   */
  public PackageIdentifier getDirectory() {
    throw new IllegalStateException();
  }

  /** Returns the repository name of the target pattern. */
  public abstract RepositoryName getRepository();

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
        String targetName,
        PackageIdentifier directory,
        String originalPattern,
        PathFragment offset) {
      super(originalPattern, offset);
      this.targetName = Preconditions.checkNotNull(targetName);
      this.directory = Preconditions.checkNotNull(directory);
    }

    @Override
    public <T, E extends Exception> void eval(
        TargetPatternResolver<T> resolver,
        InterruptibleSupplier<ImmutableSet<PathFragment>> ignoredSubdirectories,
        ImmutableSet<PathFragment> excludedSubdirectories,
        BatchCallback<T, E> callback,
        Class<E> exceptionClass)
        throws TargetParsingException, E, InterruptedException {
      callback.process(resolver.getExplicitTarget(label(targetName)).getTargets());
    }

    @Override
    public PackageIdentifier getDirectory() {
      return directory;
    }

    @Override
    public RepositoryName getRepository() {
      return directory.getRepository();
    }

    @Override
    public boolean getRulesOnly() {
      return false;
    }

    @Override
    public String getSingleTargetPath() {
      return targetName;
    }

    @Override
    public Type getType() {
      return Type.SINGLE_TARGET;
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

    private InterpretPathAsTarget(String path, String originalPattern, PathFragment offset) {
      super(originalPattern, offset);
      this.path = normalize(Preconditions.checkNotNull(path));
    }

    @Override
    public <T, E extends Exception> void eval(
        TargetPatternResolver<T> resolver,
        InterruptibleSupplier<ImmutableSet<PathFragment>> ignoredSubdirectories,
        ImmutableSet<PathFragment> excludedSubdirectories,
        BatchCallback<T, E> callback,
        Class<E> exceptionClass)
        throws TargetParsingException, E, InterruptedException {
      if (resolver.isPackage(PackageIdentifier.createInMainRepo(path))) {
        // User has specified a package name. lookout for default target.
        callback.process(resolver.getExplicitTarget(label("//" + path)).getTargets());
      } else {

        List<String> pieces = SLASH_SPLITTER.splitToList(path);

        // Interprets the label as a file target.  This loop stops as soon as the
        // first BUILD file is found (i.e. longest prefix match).
        for (int i = pieces.size() - 1; i >= 0; i--) {
          String packageName = SLASH_JOINER.join(pieces.subList(0, i));
          if (resolver.isPackage(PackageIdentifier.createInMainRepo(packageName))) {
            String targetName = SLASH_JOINER.join(pieces.subList(i, pieces.size()));
            callback.process(
                resolver
                    .getExplicitTarget(label("//" + packageName + ":" + targetName))
                    .getTargets());
            return;
          }
        }

        throw new TargetParsingException(
            "couldn't determine target from filename '" + path + "'",
            TargetPatterns.Code.CANNOT_DETERMINE_TARGET_FROM_FILENAME);
      }
    }

    @Override
    public String getPathForPathAsTarget() {
      return path;
    }

    @Override
    public RepositoryName getRepository() {
      // InterpretPathAsTarget is validated by PackageIdentifier.createInMainRepo,
      // therefore it must belong to the main repository.
      return RepositoryName.MAIN;
    }

    @Override
    public boolean getRulesOnly() {
      return false;
    }

    @Override
    public Type getType() {
      return Type.PATH_AS_TARGET;
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
    private final boolean wasOriginallyAbsolute;
    private final boolean rulesOnly;
    private final boolean checkWildcardConflict;

    private TargetsInPackage(
        String originalPattern,
        PathFragment offset,
        PackageIdentifier packageIdentifier,
        String suffix,
        boolean wasOriginallyAbsolute,
        boolean rulesOnly,
        boolean checkWildcardConflict) {
      super(originalPattern, offset);
      Preconditions.checkArgument(!packageIdentifier.getRepository().isDefault());
      this.packageIdentifier = packageIdentifier;
      this.suffix = Preconditions.checkNotNull(suffix);
      this.wasOriginallyAbsolute = wasOriginallyAbsolute;
      this.rulesOnly = rulesOnly;
      this.checkWildcardConflict = checkWildcardConflict;
    }

    @Override
    public <T, E extends Exception> void eval(
        TargetPatternResolver<T> resolver,
        InterruptibleSupplier<ImmutableSet<PathFragment>> ignoredSubdirectories,
        ImmutableSet<PathFragment> excludedSubdirectories,
        BatchCallback<T, E> callback,
        Class<E> exceptionClass)
        throws TargetParsingException, E, InterruptedException {
      if (checkWildcardConflict) {
        ResolvedTargets<T> targets = getWildcardConflict(resolver);
        if (targets != null) {
          callback.process(targets.getTargets());
          return;
        }
      }

      callback.process(
          resolver.getTargetsInPackage(getOriginalPattern(), packageIdentifier, rulesOnly));
    }

    @Override
    public PackageIdentifier getDirectory() {
      return packageIdentifier;
    }

    @Override
    public RepositoryName getRepository() {
      return packageIdentifier.getRepository();
    }

    @Override
    public boolean getRulesOnly() {
      return rulesOnly;
    }

    @Override
    public Type getType() {
      return Type.TARGETS_IN_PACKAGE;
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
      return wasOriginallyAbsolute == that.wasOriginallyAbsolute && rulesOnly == that.rulesOnly
          && checkWildcardConflict == that.checkWildcardConflict
          && getOriginalPattern().equals(that.getOriginalPattern())
          && packageIdentifier.equals(that.packageIdentifier) && suffix.equals(that.suffix);
    }

    @Override
    public int hashCode() {
      return Objects.hash(getType(), getOriginalPattern(), packageIdentifier, suffix,
          wasOriginallyAbsolute, rulesOnly, checkWildcardConflict);
    }

    /**
     * There's a potential ambiguity if '//foo/bar:all' refers to an actual target. In this case, we
     * use the target but print a warning.
     *
     * @return the Target corresponding to the given pattern, if the pattern is absolute and there
     *     is such a target. Otherwise, return null.
     */
    private <T> ResolvedTargets<T> getWildcardConflict(TargetPatternResolver<T> resolver)
        throws InterruptedException {
      if (!wasOriginallyAbsolute) {
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

  /**
   * Specialization of {@link TargetPattern} for {@link Type#TARGETS_BELOW_DIRECTORY}. Exposed
   * because it has a considerable number of specific methods. If {@link TargetPattern#getType}
   * returns {@link Type#TARGETS_BELOW_DIRECTORY} the instance can safely be cast to {@code
   * TargetsBelowDirectory}.
   */
  public static final class TargetsBelowDirectory extends TargetPattern {
    private final PackageIdentifier directory;
    private final boolean rulesOnly;

    private TargetsBelowDirectory(
        String originalPattern,
        PathFragment offset,
        PackageIdentifier directory,
        boolean rulesOnly) {
      super(originalPattern, offset);
      Preconditions.checkArgument(!directory.getRepository().isDefault());
      this.directory = Preconditions.checkNotNull(directory);
      this.rulesOnly = rulesOnly;
    }

    @Override
    public <T, E extends Exception> void eval(
        TargetPatternResolver<T> resolver,
        InterruptibleSupplier<ImmutableSet<PathFragment>> ignoredSubdirectories,
        ImmutableSet<PathFragment> excludedSubdirectories,
        BatchCallback<T, E> callback,
        Class<E> exceptionClass)
        throws TargetParsingException, E, InterruptedException {
      Preconditions.checkState(
          !excludedSubdirectories.contains(directory.getPackageFragment()),
          "Fully excluded target pattern %s should have already been filtered out (%s)",
          this,
          excludedSubdirectories);
      IgnoredPathFragmentsInScopeOrFilteringIgnorer ignoredIntersection =
          getAllIgnoredSubdirectoriesToExclude(ignoredSubdirectories);
      if (warnIfFiltered(ignoredIntersection, resolver)) {
        return;
      }
      resolver.findTargetsBeneathDirectory(
          directory.getRepository(),
          getOriginalPattern(),
          directory.getPackageFragment().getPathString(),
          rulesOnly,
          ignoredIntersection.ignoredPathFragments(),
          excludedSubdirectories,
          callback,
          exceptionClass);
    }

    @Override
    public <T, E extends Exception> ListenableFuture<Void> evalAsync(
        TargetPatternResolver<T> resolver,
        InterruptibleSupplier<ImmutableSet<PathFragment>> ignoredSubdirectories,
        ImmutableSet<PathFragment> excludedSubdirectories,
        BatchCallback<T, E> callback,
        Class<E> exceptionClass,
        ListeningExecutorService executor) {
      Preconditions.checkState(
          !excludedSubdirectories.contains(directory.getPackageFragment()),
          "Fully excluded target pattern %s should have already been filtered out (%s)",
          this,
          excludedSubdirectories);
      IgnoredPathFragmentsInScopeOrFilteringIgnorer ignoredIntersection;
      try {
        ignoredIntersection = getAllIgnoredSubdirectoriesToExclude(ignoredSubdirectories);
      } catch (InterruptedException e) {
        return immediateCancelledFuture();
      }
      if (warnIfFiltered(ignoredIntersection, resolver)) {
        return immediateVoidFuture();
      }
      return resolver.findTargetsBeneathDirectoryAsync(
          directory.getRepository(),
          getOriginalPattern(),
          directory.getPackageFragment().getPathString(),
          rulesOnly,
          ignoredIntersection.ignoredPathFragments(),
          excludedSubdirectories,
          callback,
          exceptionClass,
          executor);
    }

    private boolean warnIfFiltered(
        IgnoredPathFragmentsInScopeOrFilteringIgnorer ignoredIntersection,
        TargetPatternResolver<?> resolver) {
      if (ignoredIntersection.wasFiltered()) {
        resolver.warn(
            "Pattern '"
                + getOriginalPattern()
                + "' was filtered out by ignored directory '"
                + ignoredIntersection.filteringIgnorer().getPathString()
                + "'");
        return true;
      }
      return false;
    }

    public IgnoredPathFragmentsInScopeOrFilteringIgnorer getAllIgnoredSubdirectoriesToExclude(
        InterruptibleSupplier<ImmutableSet<PathFragment>> ignoredPackagePrefixes)
        throws InterruptedException {
      ImmutableSet.Builder<PathFragment> ignoredPathsBuilder =
          ImmutableSet.builderWithExpectedSize(0);
      for (PathFragment ignoredPackagePrefix : ignoredPackagePrefixes.get()) {
        if (this.containedIn(ignoredPackagePrefix)) {
          return new IgnoredPathFragmentsInScopeOrFilteringIgnorer.FilteringIgnorer(
              ignoredPackagePrefix);
        }
        PackageIdentifier pkgIdForIgnoredDirectorPrefix =
            PackageIdentifier.create(this.getDirectory().getRepository(), ignoredPackagePrefix);
        if (this.containsAllTransitiveSubdirectories(pkgIdForIgnoredDirectorPrefix)) {
          ignoredPathsBuilder.add(ignoredPackagePrefix);
        }
      }
      return IgnoredPathFragmentsInScopeOrFilteringIgnorer.IgnoredPathFragments.of(
          ignoredPathsBuilder.build());
    }

    /**
     * Morally an {@code Either<ImmutableSet<PathFragment>, PathFragment>}, saying whether the given
     * set of ignored directories intersected a directory (in which case the directories that were
     * in the intersection are returned) or completely contained it (in which case a containing
     * directory is returned).
     */
    public abstract static class IgnoredPathFragmentsInScopeOrFilteringIgnorer {
      public abstract boolean wasFiltered();

      public abstract ImmutableSet<PathFragment> ignoredPathFragments();

      public abstract PathFragment filteringIgnorer();

      private static class IgnoredPathFragments
          extends IgnoredPathFragmentsInScopeOrFilteringIgnorer {
        private static final IgnoredPathFragments EMPTYSET_IGNORED =
            new IgnoredPathFragments(ImmutableSet.of());

        private final ImmutableSet<PathFragment> ignoredPathFragments;

        private IgnoredPathFragments(ImmutableSet<PathFragment> ignoredPathFragments) {
          this.ignoredPathFragments = ignoredPathFragments;
        }

        static IgnoredPathFragments of(ImmutableSet<PathFragment> ignoredPathFragments) {
          if (ignoredPathFragments.isEmpty()) {
            return EMPTYSET_IGNORED;
          }
          return new IgnoredPathFragments(ignoredPathFragments);
        }

        @Override
        public boolean wasFiltered() {
          return false;
        }

        @Override
        public ImmutableSet<PathFragment> ignoredPathFragments() {
          return ignoredPathFragments;
        }

        @Override
        public PathFragment filteringIgnorer() {
          throw new UnsupportedOperationException("No filter: " + ignoredPathFragments);
        }
      }

      private static class FilteringIgnorer extends IgnoredPathFragmentsInScopeOrFilteringIgnorer {
        private final PathFragment filteringIgnorer;

        FilteringIgnorer(PathFragment filteringIgnorer) {
          this.filteringIgnorer = filteringIgnorer;
        }

        @Override
        public boolean wasFiltered() {
          return true;
        }

        @Override
        public ImmutableSet<PathFragment> ignoredPathFragments() {
          throw new UnsupportedOperationException("was filtered: " + filteringIgnorer);
        }

        @Override
        public PathFragment filteringIgnorer() {
          return filteringIgnorer;
        }
      }
    }

    /** Is {@code containingDirectory} an ancestor of or equal to this {@link #directory}? */
    public boolean containedIn(PathFragment containingDirectory) {
      return directory.getPackageFragment().startsWith(containingDirectory);
    }

    /**
     * Returns true if {@code containedDirectory} is contained by or equals this pattern's
     * directory.
     *
     * <p>For example, returns {@code true} for {@code this = TargetPattern ("//...")} and {@code
     * directory = "foo")}.
     */
    public boolean containsAllTransitiveSubdirectories(PackageIdentifier containedDirectory) {
      // Note that merely checking to see if the directory startsWith the TargetsBelowDirectory's
      // directory is insufficient. "food" begins with "foo", but "//foo/..." does not contain
      // "//food/...".
      return containedDirectory.getRepository().equals(directory.getRepository())
          && containedDirectory.getPackageFragment().startsWith(directory.getPackageFragment());
    }

    /**
     * Determines how, if it all, the evaluation of this pattern with a directory exclusion of the
     * given {@code containedPattern}'s directory relates to the evaluation of the subtraction of
     * the given {@code containedPattern} from this one.
     */
    public ContainsResult contains(TargetsBelowDirectory containedPattern) {
      if (containsAllTransitiveSubdirectories(containedPattern.getDirectory())) {
        return !getRulesOnly() && containedPattern.getRulesOnly()
            ? ContainsResult.DIRECTORY_EXCLUSION_WOULD_BE_TOO_BROAD
            : ContainsResult.DIRECTORY_EXCLUSION_WOULD_BE_EXACT;
      } else {
        return ContainsResult.NOT_CONTAINED;
      }
    }

    /** A tristate return value for {@link #contains}. */
    public enum ContainsResult {
      /**
       * Evaluating this pattern with a directory exclusion of the other pattern's directory would
       * result in exactly the same set of targets as evaluating the subtraction of the other
       * pattern from this one.
       */
      DIRECTORY_EXCLUSION_WOULD_BE_EXACT,
      /**
       * A directory exclusion of the other pattern's directory would be too broad because this
       * pattern is not "rules only" and the other one is, meaning that this pattern potentially
       * matches more targets underneath the directory in question than the other one does. Thus, a
       * directory exclusion would incorrectly exclude non-rule targets.
       */
      DIRECTORY_EXCLUSION_WOULD_BE_TOO_BROAD,
      /** None of the above. The other pattern isn't contained by this pattern. */
      NOT_CONTAINED,
    }

    @Override
    public PackageIdentifier getDirectory() {
      return directory;
    }

    @Override
    public RepositoryName getRepository() {
      return directory.getRepository();
    }

    @Override
    public boolean getRulesOnly() {
      return rulesOnly;
    }

    @Override
    public Type getType() {
      return Type.TARGETS_BELOW_DIRECTORY;
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

  /**
   * Apply a renaming to the repository part of a pattern string, returning the renamed pattern
   * string. This function only looks at the repository part of the pattern string, not the rest; so
   * any syntactic errors will not be handled here, but simply remain. Similarly, if the repository
   * part of the pattern is not syntactically valid, the renaming simply does not match and the
   * string is returned unchanged.
   */
  public static String renameRepository(
      String pattern, Map<RepositoryName, RepositoryName> renaming) {
    if (!pattern.startsWith("@")) {
      return pattern;
    }
    int pkgStart = pattern.indexOf("//");
    if (pkgStart < 0) {
      return pattern;
    }
    RepositoryName repository;
    try {
      repository = RepositoryName.create(pattern.substring(0, pkgStart));
    } catch (LabelSyntaxException e) {
      return pattern;
    }
    RepositoryName newRepository = renaming.get(repository);
    if (newRepository == null) {
      // No renaming required
      return pattern;
    }
    return newRepository.getName() + pattern.substring(pkgStart);
  }

  @Immutable
  public static final class Parser {
    // A valid pattern either starts with exactly 0 slashes (relative pattern) or exactly two
    // slashes (absolute pattern).
    private static final Pattern VALID_SLASH_PREFIX = Pattern.compile("(//)?([^/]|$)");

    // TODO(bazel-team): Merge the Label functionality that requires similar constants into this
    // class.
    /**
     * The set of target-pattern suffixes which indicate wildcards over all <em>rules</em> in a
     * single package.
     */
    private static final ImmutableList<String> ALL_RULES_IN_SUFFIXES = ImmutableList.of("all");

    /**
     * The set of target-pattern suffixes which indicate wildcards over all <em>targets</em> in a
     * single package.
     */
    private static final ImmutableList<String> ALL_TARGETS_IN_SUFFIXES =
        ImmutableList.of("*", "all-targets");

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
     * example, if the working directory is "<workspace root>/foo", then this should be "foo", which
     * will make patterns such as "bar:bar" be resolved as "//foo/bar:bar". This makes the command
     * line a bit more convenient to use.
     */
    private final PathFragment relativeDirectory;

    /** Creates a new parser with the given offset for relative patterns. */
    public Parser(PathFragment relativeDirectory) {
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
      RepositoryName repository = null;
      if (includesRepo) {
        int pkgStart = pattern.indexOf("//");
        if (pkgStart < 0) {
          throw new TargetParsingException(
              "Couldn't find package in target " + pattern, TargetPatterns.Code.PACKAGE_NOT_FOUND);
        }
        try {
          repository = RepositoryName.create(pattern.substring(0, pkgStart));
        } catch (LabelSyntaxException e) {
          throw new TargetParsingException(e.getMessage(), TargetPatterns.Code.LABEL_SYNTAX_ERROR);
        }

        pattern = pattern.substring(pkgStart);
      }

      if (!VALID_SLASH_PREFIX.matcher(pattern).lookingAt()) {
        throw new TargetParsingException(
            "not a valid absolute pattern (absolute target patterns "
                + "must start with exactly two slashes): '"
                + pattern
                + "'",
            TargetPatterns.Code.ABSOLUTE_TARGET_PATTERN_INVALID);
      }

      final boolean wasOriginallyAbsolute = pattern.startsWith("//");
      // We now ensure the relativeDirectory is applied to relative patterns.
      pattern = absolutize(pattern).substring(2);

      if (pattern.isEmpty()) {
        throw new TargetParsingException(
            "the empty string is not a valid target",
            TargetPatterns.Code.TARGET_CANNOT_BE_EMPTY_STRING);
      }

      int colonIndex = pattern.lastIndexOf(':');
      String packagePart = colonIndex < 0 ? pattern : pattern.substring(0, colonIndex);
      String targetPart = colonIndex < 0 ? "" : pattern.substring(colonIndex + 1);

      if (packagePart.equals("...")) {
        packagePart = "/...";  // special case this for easier parsing
      }

      if (packagePart.endsWith("/")) {
        throw new TargetParsingException(
            "The package part of '" + originalPattern + "' should not end in a slash",
            TargetPatterns.Code.PACKAGE_PART_CANNOT_END_IN_SLASH);
      }

      if (repository == null) {
        repository = RepositoryName.MAIN;
      }

      if (packagePart.endsWith("/...")) {
        String realPackagePart = removeSuffix(packagePart, "/...");
        PackageIdentifier packageIdentifier;
        try {
          packageIdentifier = PackageIdentifier.parse(
              repository.getName() + "//" + realPackagePart);
        } catch (LabelSyntaxException e) {
          throw new TargetParsingException(
              "Invalid package name '" + realPackagePart + "': " + e.getMessage(),
              TargetPatterns.Code.LABEL_SYNTAX_ERROR);
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
              "Invalid package name '" + packagePart + "': " + e.getMessage(),
              TargetPatterns.Code.LABEL_SYNTAX_ERROR);
        }
        return new TargetsInPackage(originalPattern, relativeDirectory, packageIdentifier,
            targetPart, wasOriginallyAbsolute, true, true);
      }

      if (ALL_TARGETS_IN_SUFFIXES.contains(targetPart)) {
        PackageIdentifier packageIdentifier;
        try {
          packageIdentifier = PackageIdentifier.parse(repository.getName() + "//" + packagePart);
        } catch (LabelSyntaxException e) {
          throw new TargetParsingException(
              "Invalid package name '" + packagePart + "': " + e.getMessage(),
              TargetPatterns.Code.LABEL_SYNTAX_ERROR);
        }
        return new TargetsInPackage(originalPattern, relativeDirectory, packageIdentifier,
            targetPart, wasOriginallyAbsolute, false, true);
      }

      if (includesRepo || wasOriginallyAbsolute || pattern.contains(":")) {
        PackageIdentifier packageIdentifier;
        String fullLabel = repository.getName() + "//" + pattern;
        try {
          PackageAndTarget packageAndTarget = LabelValidator.validateAbsoluteLabel(fullLabel);
          packageIdentifier =
              PackageIdentifier.create(
                  repository, PathFragment.create(packageAndTarget.getPackageName()));
        } catch (BadLabelException e) {
          String error = "invalid target format '" + originalPattern + "': " + e.getMessage();
          throw new TargetParsingException(error, TargetPatterns.Code.TARGET_FORMAT_INVALID);
        }
        return new SingleTarget(fullLabel, packageIdentifier, originalPattern, relativeDirectory);
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
            "Bad target pattern '" + originalPattern + "': " + e.getMessage(),
            TargetPatterns.Code.LABEL_SYNTAX_ERROR);
      }
      return new InterpretPathAsTarget(pattern, originalPattern, relativeDirectory);
    }

    /**
     * Parses a constant string TargetPattern, throwing IllegalStateException on invalid pattern.
     */
    @CheckReturnValue
    public TargetPattern parseConstantUnchecked(@CompileTimeConstant String pattern) {
      try {
        return parse(pattern);
      } catch (TargetParsingException e) {
        throw new IllegalStateException(e);
      }
    }

    /**
     * Absolutizes the target pattern to the offset. Patterns starting with "//" are absolute and
     * not modified. Assumes the given pattern is not invalid wrt leading "/"s.
     *
     * <p>If the offset is "foo": absolutize(":bar") --> "//foo:bar" absolutize("bar") -->
     * "//foo/bar" absolutize("//biz/bar") --> "//biz/bar" (absolute) absolutize("biz:bar") -->
     * "//foo/biz:bar"
     *
     * @param pattern The target pattern to parse.
     * @return the pattern, absolutized to the offset if approprate.
     */
    public String absolutize(String pattern) {
      if (pattern.startsWith("//")) {
        return pattern;
      }

      // PathFragment#getRelative doesn't work when the pattern starts with ":".
      // "foo".getRelative(":all") would return "foo/:all", where we really want "foo:all".
      return pattern.startsWith(":") || relativeDirectory.isEmpty()
          ? "//" + relativeDirectory.getPathString() + pattern
          : "//" + relativeDirectory.getPathString() + "/" + pattern;
    }
  }

  // Parse 'label' as a Label, mapping LabelSyntaxException into
  // TargetParsingException.
  private static Label label(String label) throws TargetParsingException {
    try {
      return Label.parseAbsolute(label, ImmutableMap.of());
    } catch (LabelSyntaxException e) {
      throw new TargetParsingException(
          "invalid target format: '"
              + StringUtilities.sanitizeControlChars(label)
              + "'; "
              + StringUtilities.sanitizeControlChars(e.getMessage()),
          TargetPatterns.Code.TARGET_FORMAT_INVALID);
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
