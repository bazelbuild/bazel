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

package com.google.devtools.build.lib.pkgcache;

import static com.google.common.base.Preconditions.checkState;

import com.google.common.collect.ImmutableCollection;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.io.InconsistentFilesystemException;
import com.google.devtools.build.lib.packages.InputFile;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchPackagePieceException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.PackagePiece;
import com.google.devtools.build.lib.packages.Packageoid;
import com.google.devtools.build.lib.packages.Target;

/**
 * API for retrieving packages. Implementations generally load packages to fulfill requests.
 *
 * <p><b>Concurrency</b>: Implementations should be thread safe for {@link #getPackage}.
 */
public interface PackageProvider extends TargetProvider {

  /**
   * Returns the {@link Package} named "packageName". If there is no such package (e.g. {@code
   * isPackage(packageName)} returns false), throws a {@link NoSuchPackageException}.
   *
   * <p>The returned package may contain lexical/grammatical errors, in which case <code>
   * pkg.containsErrors() == true</code>. Such packages may be missing some rules. Any rules that
   * are present may soundly be used for builds, though.
   *
   * @param eventHandler the eventHandler on which to report warnings and errors associated with
   *     loading the package, but only if the package has not already been loaded
   * @param packageName a legal package name.
   * @throws NoSuchPackageException if the package could not be found.
   * @throws InterruptedException if the package loading was interrupted.
   */
  Package getPackage(ExtendedEventHandler eventHandler, PackageIdentifier packageName)
      throws NoSuchPackageException, InterruptedException;

  /**
   * If a {@link Target} is owned by a monolithic {@link Package}, returns it; otherwise, loads and
   * returns the full package encompassing the target's package piece.
   *
   * @throws NoSuchPackageException if target is owned by a {@link PackagePiece}, and the full
   *     package could not be loaded due to an error while loading a different package piece.
   * @throws InterruptedException if the package loading was interrupted.
   */
  default Package getPackage(ExtendedEventHandler eventHandler, Target target)
      throws NoSuchPackageException, InterruptedException {
    Packageoid packageoid = target.getPackageoid();
    if (packageoid instanceof Package pkg) {
      // Monolithic package.
      return pkg;
    }
    return getPackage(eventHandler, packageoid.getPackageIdentifier());
  }

  /**
   * Returns whether a package with the given name exists. That is, returns whether all the
   * following hold
   *
   * <ol>
   *   <li>{@code packageName} is a valid package name
   *   <li>there is a BUILD file for the package
   *   <li>the package is not considered deleted via --deleted_packages
   * </ol>
   *
   * <p>If these don't hold, then attempting to read the package with {@link #getPackage} may fail
   * or may return a package containing errors.
   *
   * @param eventHandler if {@code packageName} specifies a package that could not be looked up
   *     because of a symlink cycle or IO error, the error is reported here
   * @param packageName the name of the package.
   */
  boolean isPackage(ExtendedEventHandler eventHandler, PackageIdentifier packageName)
      throws InconsistentFilesystemException, InterruptedException;

  /**
   * Returns the BUILD file target of the given package, loading, parsing and evaluating either the
   * full package (if lazy macro expansion is disabled) or just the package piece owning the BUILD
   * file (if lazy macro expansion is enabled) if it is not already loaded.
   *
   * @throws NoSuchPackageException if the package could not be found
   * @throws NoSuchPackagePieceException if lazy macro expansion is enabled, and the package piece
   *     owning the BUILD file failed validation
   * @throws InterruptedException if the package loading was interrupted
   */
  InputFile getBuildFile(ExtendedEventHandler eventHandler, PackageIdentifier packageName)
      throws NoSuchPackageException, NoSuchPackagePieceException, InterruptedException;

  @Override
  default InputFile getBuildFile(Target target) throws InterruptedException {
    Packageoid packageoid = target.getPackageoid();
    if (packageoid instanceof Package pkg) {
      // Monolithic package.
      return pkg.getBuildFile();
    } else if (packageoid instanceof PackagePiece.ForBuildFile forBuildFile) {
      // Lazy macro expansion, target is top-level.
      return forBuildFile.getBuildFile();
    } else {
      // Lazy macro expansion, target is in a PackagePiece.ForMacro, we need to retrieve the
      // BUILD file from the (already loaded) PackagePiece.ForBuildFile.
      StoredEventHandler localEventHandler = new StoredEventHandler();
      InputFile buildFile;
      try {
        buildFile =
            getBuildFile(localEventHandler, target.getPackageMetadata().packageIdentifier());
      } catch (NoSuchPackageException | NoSuchPackagePieceException e) {
        // If a PackagePiece.ForMacro exists, its corresponding PackagePiece.ForBuildFile must also
        // exist (and already be loaded).
        throw new IllegalStateException(
            String.format(
                "Bug in package loading machinery: failed to load package piece for BUILD file of"
                    + " already-loaded target %s",
                target),
            e);
      }
      // If PackagePiece.ForMacro was loaded, its corresponding PackagePiece.ForBuildFile could not
      // be in error.
      checkState(
          !localEventHandler.hasErrors(),
          "Bug in package loading machinery: unexpected error while retrieving package piece for"
              + " BUILD file of already-loaded target %s: %s",
          target,
          localEventHandler.getEvents());
      return buildFile;
    }
  }

  @Override
  default ImmutableCollection<Target> getSiblingTargetsInPackage(
      ExtendedEventHandler eventHandler, Target target)
      throws NoSuchPackageException, InterruptedException {
    return getPackage(eventHandler, target).getTargets().values();
  }
}
