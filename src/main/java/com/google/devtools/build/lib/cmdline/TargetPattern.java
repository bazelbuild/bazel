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
import com.google.common.base.MoreObjects;
import com.google.common.base.MoreObjects.ToStringHelper;
import com.google.common.base.Preconditions;
import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.ListeningExecutorService;
import com.google.devtools.build.lib.io.InconsistentFilesystemException;
import com.google.devtools.build.lib.io.ProcessPackageDirectoryException;
import com.google.devtools.build.lib.server.FailureDetails.TargetPatterns;
import com.google.devtools.build.lib.server.FailureDetails.TargetPatterns.Code;
import com.google.devtools.build.lib.supplier.InterruptibleSupplier;
import com.google.devtools.build.lib.util.StringUtilities;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.errorprone.annotations.CheckReturnValue;
import com.google.errorprone.annotations.CompileTimeConstant;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Objects;
import javax.annotation.Nullable;
import javax.annotation.concurrent.Immutable;

/**
 * Represents a target pattern. Target patterns are a generalization of labels to include wildcards
 * for finding all packages recursively beneath some root, and for finding all targets within a
 * package.
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

  private static final Parser DEFAULT_PARSER = mainRepoParser(PathFragment.EMPTY_FRAGMENT);

  private final String originalPattern;

  /**
   * Returns a parser defaulting to the main repo, with no offset or repo mapping. Note that the
   * Parser class is immutable, so this method may return the same instance on subsequent calls.
   */
  public static Parser defaultParser() {
    return DEFAULT_PARSER;
  }

  /**
   * Returns a parser defaulting to the main repo, with repo mapping, but using the given offset.
   */
  // NOTE(wyv): This is only strictly correct within a monorepo. If external repos exist, there
  // should always be a proper repo mapping. We should audit calls to this function and add a repo
  // mapping wherever appropriate.
  public static Parser mainRepoParser(PathFragment offset) {
    return new Parser(offset, RepositoryName.MAIN, RepositoryMapping.ALWAYS_FALLBACK);
  }

  /**
   * Normalizes the given relative path by resolving {@code //}, {@code /./} and {@code x/../}
   * pieces. Note that leading {@code ".."} segments are not removed, so the returned string can
   * have leading {@code ".."} segments.
   *
   * @throws IllegalArgumentException if the path is absolute, i.e. starts with {@code /}
   */
  @VisibleForTesting
  static String normalize(String path) {
    Preconditions.checkArgument(!path.startsWith("/"), path);
    Preconditions.checkArgument(!path.startsWith("@"), path);
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

  private TargetPattern(String originalPattern) {
    // Don't allow inheritance outside this class.
    this.originalPattern = Preconditions.checkNotNull(originalPattern);
  }

  /**
   * Return the type of the pattern. Examples include "below directory" like "foo/..." and "single
   * target" like "//x:y".
   */
  public abstract Type getType();

  /** Return the string that was parsed into this pattern. */
  public String getOriginalPattern() {
    return originalPattern;
  }

  /**
   * Evaluates the current target pattern, excluding targets under directories in both {@code
   * ignoredSubdirectories} and {@code excludedSubdirectories}, and returns the result.
   *
   * @throws InconsistentFilesystemException if {@code resolver} makes Skyframe calls and discovers
   *     a filesystem inconsistency as observed by Skyframe, and this pattern does not have type
   *     {@code Type.TARGETS_BELOW_DIRECTORY}
   * @throws ProcessPackageDirectoryException if {@code resolver} makes Skyframe calls and discovers
   *     a filesystem inconsistency as observed by Skyframe, and this pattern has type {@code
   *     Type.TARGETS_BELOW_DIRECTORY}
   * @throws IllegalArgumentException if either {@code ignoredSubdirectories} or {@code
   *     excludedSubdirectories} is nonempty and this pattern does not have type {@code
   *     Type.TARGETS_BELOW_DIRECTORY}.
   */
  public abstract <T, E extends Exception & QueryExceptionMarkerInterface> void eval(
      TargetPatternResolver<T> resolver,
      InterruptibleSupplier<ImmutableSet<PathFragment>> ignoredSubdirectories,
      ImmutableSet<PathFragment> excludedSubdirectories,
      BatchCallback<T, E> callback,
      Class<E> exceptionClass)
      throws TargetParsingException, E, InterruptedException, ProcessPackageDirectoryException,
          InconsistentFilesystemException;

  /**
   * Evaluates this {@link TargetPattern} synchronously, feeding the result to the given {@code
   * callback}, and then returns an appropriate immediate {@link ListenableFuture}.
   *
   * <p>If the returned {@link ListenableFuture}'s {@link ListenableFuture#get} throws an {@code
   * ExecutionException}, the cause will be an instance of either {@link TargetParsingException} or
   * the given {@code exceptionClass}.
   *
   * <p>This method must not be called from within Skyframe evaluation. Use {@link
   * com.google.devtools.build.lib.skyframe.TargetPatternFunction} and friends for that.
   */
  public final <T, E extends Exception & QueryExceptionMarkerInterface>
      ListenableFuture<Void> evalAdaptedForAsync(
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
    } catch (ProcessPackageDirectoryException | InconsistentFilesystemException e) {
      throw new IllegalStateException(
          "Cannot throw filesystem-related exceptions outside of Skyframe evaluation for " + this,
          e);
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
   * <p>If the returned {@link ListenableFuture}'s {@link ListenableFuture#get} throws an {@code
   * ExecutionException}, the cause will be an instance of either {@link TargetParsingException} or
   * the given {@code exceptionClass}.
   */
  public <T, E extends Exception & QueryExceptionMarkerInterface> ListenableFuture<Void> evalAsync(
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
   * <p>The interpretation of this path, of course, depends on the existence of packages. See {@link
   * InterpretPathAsTarget#eval}.
   */
  public String getPathForPathAsTarget() {
    throw new IllegalStateException();
  }

  /** For patterns of type {@link Type#SINGLE_TARGET}, returns the label to the target. */
  public Label getSingleTargetLabel() {
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
   * Returns {@code true} iff this pattern has type {@code Type.TARGETS_BELOW_DIRECTORY} or {@code
   * Type.TARGETS_IN_PACKAGE} and the target pattern suffix specified it should match rules only.
   */
  public abstract boolean getRulesOnly();

  protected final ToStringHelper toStringHelper() {
    return MoreObjects.toStringHelper(this).add("originalPattern", originalPattern);
  }

  @VisibleForTesting
  static final class SingleTarget extends TargetPattern {

    private final Label target;

    @VisibleForTesting
    SingleTarget(String originalPattern, Label target) {
      super(originalPattern);
      this.target = Preconditions.checkNotNull(target);
    }

    @Override
    public <T, E extends Exception & QueryExceptionMarkerInterface> void eval(
        TargetPatternResolver<T> resolver,
        InterruptibleSupplier<ImmutableSet<PathFragment>> ignoredSubdirectories,
        ImmutableSet<PathFragment> excludedSubdirectories,
        BatchCallback<T, E> callback,
        Class<E> exceptionClass)
        throws TargetParsingException, E, InterruptedException {
      callback.process(resolver.getExplicitTarget(target).getTargets());
    }

    @Override
    public PackageIdentifier getDirectory() {
      return target.getPackageIdentifier();
    }

    @Override
    public RepositoryName getRepository() {
      return target.getRepository();
    }

    @Override
    public boolean getRulesOnly() {
      return false;
    }

    @Override
    public Label getSingleTargetLabel() {
      return target;
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
      return target.equals(that.target);
    }

    @Override
    public int hashCode() {
      return Objects.hash(getType(), target);
    }

    @Override
    public String toString() {
      return toStringHelper().add("target", target).toString();
    }
  }

  @VisibleForTesting
  static final class InterpretPathAsTarget extends TargetPattern {
    private final String path;

    @VisibleForTesting
    InterpretPathAsTarget(String originalPattern, String path) {
      super(originalPattern);
      this.path = normalize(Preconditions.checkNotNull(path));
    }

    @Override
    public <T, E extends Exception & QueryExceptionMarkerInterface> void eval(
        TargetPatternResolver<T> resolver,
        InterruptibleSupplier<ImmutableSet<PathFragment>> ignoredSubdirectories,
        ImmutableSet<PathFragment> excludedSubdirectories,
        BatchCallback<T, E> callback,
        Class<E> exceptionClass)
        throws TargetParsingException, E, InterruptedException, InconsistentFilesystemException {
      PackageIdentifier pathAsPackage = PackageIdentifier.createInMainRepo(path);
      if (resolver.isPackage(pathAsPackage)) {
        // User has specified a package name. lookout for default target.
        callback.process(
            resolver
                .getExplicitTarget(
                    label(pathAsPackage, pathAsPackage.getPackageFragment().getBaseName()))
                .getTargets());
      } else {
        List<String> pieces = SLASH_SPLITTER.splitToList(path);

        // Interprets the label as a file target.  This loop stops as soon as the
        // first BUILD file is found (i.e. longest prefix match).
        for (int i = pieces.size() - 1; i >= 0; i--) {
          PackageIdentifier pkg =
              PackageIdentifier.createInMainRepo(SLASH_JOINER.join(pieces.subList(0, i)));
          if (resolver.isPackage(pkg)) {
            String targetName = SLASH_JOINER.join(pieces.subList(i, pieces.size()));
            callback.process(resolver.getExplicitTarget(label(pkg, targetName)).getTargets());
            return;
          }
        }

        throw new TargetParsingException(
            "couldn't determine target from filename '" + path + "'",
            Code.CANNOT_DETERMINE_TARGET_FROM_FILENAME);
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

    @Override
    public String toString() {
      return toStringHelper().add("path", path).toString();
    }
  }

  @VisibleForTesting
  static final class TargetsInPackage extends TargetPattern {
    private final PackageIdentifier packageIdentifier;
    private final String suffix;
    private final boolean wasOriginallyAbsolute;
    private final boolean rulesOnly;

    @VisibleForTesting
    TargetsInPackage(
        String originalPattern,
        PackageIdentifier packageIdentifier,
        String suffix,
        boolean wasOriginallyAbsolute,
        boolean rulesOnly) {
      super(originalPattern);
      this.packageIdentifier = packageIdentifier;
      this.suffix = Preconditions.checkNotNull(suffix);
      this.wasOriginallyAbsolute = wasOriginallyAbsolute;
      this.rulesOnly = rulesOnly;
    }

    @Override
    public <T, E extends Exception & QueryExceptionMarkerInterface> void eval(
        TargetPatternResolver<T> resolver,
        InterruptibleSupplier<ImmutableSet<PathFragment>> ignoredSubdirectories,
        ImmutableSet<PathFragment> excludedSubdirectories,
        BatchCallback<T, E> callback,
        Class<E> exceptionClass)
        throws TargetParsingException, E, InterruptedException, InconsistentFilesystemException {
      ResolvedTargets<T> targets = getWildcardConflict(resolver);
      if (targets != null) {
        callback.process(targets.getTargets());
        return;
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
      return wasOriginallyAbsolute == that.wasOriginallyAbsolute
          && rulesOnly == that.rulesOnly
          && getOriginalPattern().equals(that.getOriginalPattern())
          && packageIdentifier.equals(that.packageIdentifier)
          && suffix.equals(that.suffix);
    }

    @Override
    public int hashCode() {
      return Objects.hash(
          getType(),
          getOriginalPattern(),
          packageIdentifier,
          suffix,
          wasOriginallyAbsolute,
          rulesOnly);
    }

    @Override
    public String toString() {
      return toStringHelper()
          .add("packageIdentifier", packageIdentifier)
          .add("suffix", suffix)
          .add("wasOriginallyAbsolute", wasOriginallyAbsolute)
          .add("rulesOnly", rulesOnly)
          .toString();
    }

    /**
     * There's a potential ambiguity if '//foo/bar:all' refers to an actual target. In this case, we
     * use the target but print a warning.
     *
     * @return the Target corresponding to the given pattern, if the pattern is absolute and there
     *     is such a target. Otherwise, return null.
     */
    @Nullable
    private <T> ResolvedTargets<T> getWildcardConflict(TargetPatternResolver<T> resolver)
        throws InconsistentFilesystemException, InterruptedException {
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
        resolver.warn(
            String.format(
                "The target pattern '%s' is ambiguous: '%s' is "
                    + "both a wildcard, and the name of an existing %s; "
                    + "using the latter interpretation",
                getOriginalPattern(), ":" + suffix, resolver.getTargetKind(target)));
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

    @VisibleForTesting
    TargetsBelowDirectory(String originalPattern, PackageIdentifier directory, boolean rulesOnly) {
      super(originalPattern);
      this.directory = Preconditions.checkNotNull(directory);
      this.rulesOnly = rulesOnly;
    }

    @Override
    public <T, E extends Exception & QueryExceptionMarkerInterface> void eval(
        TargetPatternResolver<T> resolver,
        InterruptibleSupplier<ImmutableSet<PathFragment>> ignoredSubdirectories,
        ImmutableSet<PathFragment> excludedSubdirectories,
        BatchCallback<T, E> callback,
        Class<E> exceptionClass)
        throws TargetParsingException, E, InterruptedException, ProcessPackageDirectoryException {
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
    public <T, E extends Exception & QueryExceptionMarkerInterface>
        ListenableFuture<Void> evalAsync(
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
            PackageIdentifier.create(directory.getRepository(), ignoredPackagePrefix);
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
      if (containsAllTransitiveSubdirectories(containedPattern.directory)) {
        return !rulesOnly && containedPattern.rulesOnly
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
      return rulesOnly == that.rulesOnly
          && getOriginalPattern().equals(that.getOriginalPattern())
          && directory.equals(that.directory);
    }

    @Override
    public int hashCode() {
      return Objects.hash(getType(), getOriginalPattern(), directory, rulesOnly);
    }

    @Override
    public String toString() {
      return toStringHelper().add("directory", directory).add("rulesOnly", rulesOnly).toString();
    }
  }

  @Immutable
  public static final class Parser {
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

    /**
     * Directory prefix to use when resolving relative labels (rather than absolute ones). For
     * example, if the working directory is "<workspace root>/foo", then this should be "foo", which
     * will make patterns such as "bar:bar" be resolved as "//foo/bar:bar". This makes the command
     * line a bit more convenient to use.
     */
    private final PathFragment relativeDirectory;

    // The repo to use for any repo-relative target patterns (so "//foo" becomes
    // "@currentRepo//foo").
    private final RepositoryName currentRepo;

    // The repo mapping to use for the @repo part of target patterns.
    private final RepositoryMapping repoMapping;

    /** Creates a new parser with the given offset for relative patterns. */
    public Parser(
        PathFragment relativeDirectory, RepositoryName currentRepo, RepositoryMapping repoMapping) {
      Preconditions.checkArgument(
          currentRepo.isMain() || relativeDirectory.isEmpty(),
          "parsing target patterns in a non-main repo with a relative directory is unsupported");
      this.relativeDirectory = relativeDirectory;
      this.currentRepo = currentRepo;
      this.repoMapping = repoMapping;
    }

    /**
     * Parses the given pattern, and throws an exception if the pattern is invalid.
     *
     * @return a target pattern corresponding to the pattern parsed
     * @throws TargetParsingException if the pattern is invalid
     */
    public TargetPattern parse(String pattern) throws TargetParsingException {
      LabelParser.Parts parts;
      try {
        parts = LabelParser.Parts.parse(pattern);
      } catch (LabelSyntaxException e) {
        throw new TargetParsingException(e.getMessage(), TargetPatterns.Code.LABEL_SYNTAX_ERROR);
      }

      // Special case: For a target pattern that just looks like `foo/bar/baz`, we treat this as a
      // file path. LabelParser parses it as `:foo/bar/baz`, so we need to distinguish this case by
      // checking if the original pattern contains a colon.
      if (!parts.pkgIsAbsolute()
          && currentRepo.isMain()
          && parts.pkg().isEmpty()
          && !parts.pkgEndsWithTripleDots()
          && !pattern.contains(":")) {
        return new InterpretPathAsTarget(
            pattern, relativeDirectory.getRelative(parts.target()).getPathString());
      }

      PackageIdentifier packageIdentifier = createPackageIdentifierFromParts(parts);
      if (parts.pkgEndsWithTripleDots()) {
        if (parts.target().isEmpty() || ALL_RULES_IN_SUFFIXES.contains(parts.target())) {
          return new TargetsBelowDirectory(pattern, packageIdentifier, true);
        } else if (ALL_TARGETS_IN_SUFFIXES.contains(parts.target())) {
          return new TargetsBelowDirectory(pattern, packageIdentifier, false);
        }
        throw new TargetParsingException(
            "Invalid target pattern " + pattern + ": '...' can only be used with wildcard targets",
            Code.LABEL_SYNTAX_ERROR);
      }

      if (pattern.contains(":") && ALL_RULES_IN_SUFFIXES.contains(parts.target())) {
        return new TargetsInPackage(
            pattern, packageIdentifier, parts.target(), parts.pkgIsAbsolute(), true);
      }

      if (pattern.contains(":") && ALL_TARGETS_IN_SUFFIXES.contains(parts.target())) {
        return new TargetsInPackage(
            pattern, packageIdentifier, parts.target(), parts.pkgIsAbsolute(), false);
      }

      return new SingleTarget(pattern, Label.createUnvalidated(packageIdentifier, parts.target()));
    }

    private PackageIdentifier createPackageIdentifierFromParts(LabelParser.Parts parts)
        throws TargetParsingException {
      RepositoryName repo;
      if (parts.repo() == null) {
        repo = currentRepo;
      } else if (parts.repoIsCanonical()) {
        repo = RepositoryName.createUnvalidated(parts.repo());
      } else {
        repo = repoMapping.get(parts.repo());
        if (!repo.isVisible()) {
          throw new TargetParsingException(
              String.format(
                  "No repository visible as '@%s' from %s",
                  repo.getName(), repo.getOwnerRepoDisplayString()),
              Code.PACKAGE_NOT_FOUND);
        }
      }

      PathFragment packagePathFragment =
          parts.pkgIsAbsolute()
              ? PathFragment.create(parts.pkg())
              : relativeDirectory.getRelative(parts.pkg());
      return PackageIdentifier.create(repo, packagePathFragment);
    }

    public RepositoryMapping getRepoMapping() {
      return repoMapping;
    }

    public RepositoryName getCurrentRepo() {
      return currentRepo;
    }

    public PathFragment getRelativeDirectory() {
      return relativeDirectory;
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

  // Creates a label from parts, mapping LabelSyntaxException into TargetParsingException.
  private static Label label(PackageIdentifier pkg, String targetName)
      throws TargetParsingException {
    try {
      return Label.create(pkg, targetName);
    } catch (LabelSyntaxException e) {
      throw new TargetParsingException(
          "invalid target name: '"
              + StringUtilities.sanitizeControlChars(targetName)
              + "'; "
              + StringUtilities.sanitizeControlChars(e.getMessage()),
          TargetPatterns.Code.TARGET_FORMAT_INVALID);
    }
  }

  /** The target pattern type (targets below package, in package, explicit target, etc.) */
  public enum Type {
    /** A path interpreted as a target, eg "foo/bar/baz" */
    PATH_AS_TARGET,
    /** An explicit target, eg "//foo:bar." */
    SINGLE_TARGET,
    /** Targets below a directory, eg "foo/...". */
    TARGETS_BELOW_DIRECTORY,
    /** Target in a package, eg "foo:all". */
    TARGETS_IN_PACKAGE
  }
}
