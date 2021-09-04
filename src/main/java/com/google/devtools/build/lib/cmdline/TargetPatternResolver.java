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

import com.google.common.collect.ImmutableSet;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.ListeningExecutorService;
import com.google.devtools.build.lib.concurrent.BatchCallback;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Collection;

/**
 * A callback that is used during the process of converting target patterns (such as
 * <code>//foo:all</code>) into one or more lists of targets (such as <code>//foo:foo,
 * //foo:bar</code>). During a call to {@link TargetPattern#eval}, the {@link TargetPattern} makes
 * calls to this interface to implement the target pattern semantics. The generic type {@code T} is
 * only for compile-time type safety; there are no requirements to the actual type.
 */
public abstract class TargetPatternResolver<T> {

  /**
   * Reports the given warning.
   */
  public abstract void warn(String msg);

  /**
   * Returns a single target corresponding to the given label, or null. This method may only throw
   * an exception if the current thread was interrupted.
   */
  public abstract T getTargetOrNull(Label label) throws InterruptedException;

  /**
   * Returns a single target corresponding to the given label, or an empty or failed result.
   */
  public abstract ResolvedTargets<T> getExplicitTarget(Label label)
      throws TargetParsingException, InterruptedException;

  /**
   * Returns the set containing the targets found in the given package. The specified directory is
   * not necessarily a valid package name. If {@code rulesOnly} is true, then this method should
   * only return rules in the given package.
   *
   * @param originalPattern the original target pattern for error reporting purposes
   * @param packageIdentifier the identifier of the package
   * @param rulesOnly whether to return rules only
   */
  public abstract Collection<T> getTargetsInPackage(
      String originalPattern, PackageIdentifier packageIdentifier, boolean rulesOnly)
      throws TargetParsingException, InterruptedException;

  /**
   * Computes the set containing the targets found below the given {@code directory}, passing it in
   * batches to {@code callback}. Conceptually, this method should look for all packages that start
   * with the {@code directory} (as a proper prefix directory, i.e., "foo/ba" is not a proper prefix
   * of "foo/bar/"), and then collect all targets in each such package (subject to {@code
   * rulesOnly}) as if calling {@link #getTargetsInPackage}. The specified directory is not
   * necessarily a valid package name.
   *
   * <p>Note that the {@code directory} can be empty, which corresponds to the "//..." pattern.
   * Implementations may choose not to support this case and throw an {@link
   * IllegalArgumentException} exception instead, or may restrict the set of directories that are
   * considered by default.
   *
   * <p>If the {@code directory} points to a package, then that package should also be part of the
   * result.
   *
   * @param originalPattern the original target pattern for error reporting purposes
   * @param directory the directory in which to look for packages
   * @param rulesOnly whether to return rules only
   * @param forbiddenSubdirectories a set of transitive subdirectories beneath {@code directory} to
   *     ignore
   * @param excludedSubdirectories another set of transitive subdirectories beneath {@code
   *     directory} to ignore
   * @param callback the callback to receive the result, possibly in multiple batches.
   * @param exceptionClass The class type of the parameterized exception.
   * @throws TargetParsingException under implementation-specific failure conditions
   */
  public abstract <E extends Exception> void findTargetsBeneathDirectory(
      RepositoryName repository,
      String originalPattern,
      String directory,
      boolean rulesOnly,
      ImmutableSet<PathFragment> forbiddenSubdirectories,
      ImmutableSet<PathFragment> excludedSubdirectories,
      BatchCallback<T, E> callback,
      Class<E> exceptionClass)
      throws TargetParsingException, E, InterruptedException;

  /**
   * Async version of {@link #findTargetsBeneathDirectory}
   *
   * <p>Default implementation is synchronous.
   */
  public <E extends Exception> ListenableFuture<Void> findTargetsBeneathDirectoryAsync(
      RepositoryName repository,
      String originalPattern,
      String directory,
      boolean rulesOnly,
      ImmutableSet<PathFragment> forbiddenSubdirectories,
      ImmutableSet<PathFragment> excludedSubdirectories,
      BatchCallback<T, E> callback,
      Class<E> exceptionClass,
      ListeningExecutorService executor) {
      try {
      findTargetsBeneathDirectory(
          repository,
          originalPattern,
          directory,
          rulesOnly,
          forbiddenSubdirectories,
          excludedSubdirectories,
          callback,
          exceptionClass);
        return Futures.immediateFuture(null);
      } catch (TargetParsingException e) {
        return Futures.immediateFailedFuture(e);
      } catch (InterruptedException e) {
        return Futures.immediateCancelledFuture();
      } catch (Exception e) {
        if (exceptionClass.isInstance(e)) {
          return Futures.immediateFailedFuture(e);
        }
        throw new IllegalStateException(e);
      }
  }

  /**
   * Returns true, if and only if the given package identifier corresponds to a package, i.e., a
   * file with the name {@code packageName/BUILD} exists in the appropriate repository.
   */
  public abstract boolean isPackage(PackageIdentifier packageIdentifier)
      throws InterruptedException;

  /**
   * Returns the target kind of the given target, for example {@code cc_library rule}.
   */
  public abstract String getTargetKind(T target);
}
