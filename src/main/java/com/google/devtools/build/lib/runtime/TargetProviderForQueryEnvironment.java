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
package com.google.devtools.build.lib.runtime;

import static com.google.common.base.Throwables.throwIfInstanceOf;
import static com.google.common.base.Throwables.throwIfUnchecked;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableCollection;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.BuildFileContainsErrorsException;
import com.google.devtools.build.lib.packages.InputFile;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.PackagePiece;
import com.google.devtools.build.lib.packages.PackagePieceIdentifier;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.PackageProvider;
import com.google.devtools.build.lib.pkgcache.TargetProvider;
import com.google.devtools.build.lib.skyframe.PackagePieceValue;
import com.google.devtools.build.lib.skyframe.PackageValue;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.WalkableGraph;
import java.util.Optional;

/**
 * Serves as TargetProvider using WalkableGraph as direct access to graph. Refers to delegate in
 * case if WalkableGraph has not value for specific key.
 */
public class TargetProviderForQueryEnvironment implements TargetProvider {

  private final WalkableGraph walkableGraph;

  /** If WalkableGraph has not node requested, then delegate used as fall back strategy. */
  private final PackageProvider delegate;

  public TargetProviderForQueryEnvironment(WalkableGraph walkableGraph, PackageProvider delegate) {
    this.walkableGraph = Preconditions.checkNotNull(walkableGraph);
    this.delegate = Preconditions.checkNotNull(delegate);
  }

  @Override
  public Target getTarget(ExtendedEventHandler eventHandler, Label label)
      throws NoSuchPackageException, NoSuchTargetException, InterruptedException {

    Optional<Package> optional = getPackageFromWalkableGraph(label.getPackageIdentifier());
    if (optional.isPresent()) {
      return optional.get().getTarget(label.getName());
    }

    return delegate.getTarget(eventHandler, label);
  }

  @Override
  public InputFile getBuildFile(Target target) throws InterruptedException {
    if (target.getPackageoid() instanceof Package pkg) {
      // Monolithic package.
      return pkg.getBuildFile();
    } else if (target.getPackageoid() instanceof PackagePiece.ForBuildFile forBuildFile) {
      // Lazy macro expansion, target is top-level.
      return forBuildFile.getBuildFile();
    }
    // Lazy macro expansion mode, target is in a PackagePiece.ForMacro.
    PackagePieceIdentifier.ForBuildFile skyKey =
        new PackagePieceIdentifier.ForBuildFile(target.getPackageMetadata().packageIdentifier());
    SkyValue skyValue = walkableGraph.getValue(skyKey);
    if (skyValue != null) {
      PackagePieceValue.ForBuildFile packageValue = (PackagePieceValue.ForBuildFile) skyValue;
      return packageValue.getPackagePiece().getBuildFile();
    }
    if (walkableGraph.getException(skyKey) != null) {
      throw illegalErrorInPackagePieceForBuildFile(target, walkableGraph.getException(skyKey));
    }
    try {
      checkCycle(skyKey);
    } catch (NoSuchPackageException e) {
      throw illegalErrorInPackagePieceForBuildFile(target, e);
    }

    return delegate.getBuildFile(target);
  }

  private static IllegalStateException illegalErrorInPackagePieceForBuildFile(
      Target target, Exception cause) {
    return new IllegalStateException(
        String.format(
            "Bug in package loading machinery: failed to load package piece for BUILD file of"
                + " already-loaded target %s",
            target),
        cause);
  }

  @Override
  public ImmutableCollection<Target> getSiblingTargetsInPackage(
      ExtendedEventHandler eventHandler, Target target)
      throws NoSuchPackageException, InterruptedException {
    Optional<Package> optional =
        getPackageFromWalkableGraph(target.getPackageMetadata().packageIdentifier());
    if (optional.isPresent()) {
      return optional.get().getTargets().values();
    }
    return delegate.getSiblingTargetsInPackage(eventHandler, target);
  }

  private Optional<Package> getPackageFromWalkableGraph(PackageIdentifier pkgId)
      throws InterruptedException, NoSuchPackageException {
    SkyValue skyValue = walkableGraph.getValue(pkgId);

    if (skyValue != null) {
      PackageValue packageValue = (PackageValue) skyValue;
      return Optional.of(packageValue.getPackage());
    }

    Exception exception = walkableGraph.getException(pkgId);
    if (exception != null) {
      // PackageFunction should be catching, swallowing, and rethrowing all transitive
      // errors as NoSuchPackageExceptions or constructing packages with errors.
      throwIfInstanceOf(exception, NoSuchPackageException.class);
      throwIfUnchecked(exception);
      throw new IllegalStateException(
          String.format("Unexpected Exception type from PackageValue for %s", pkgId));
    }
    checkCycle(pkgId);
    return Optional.empty();
  }

  private void checkCycle(SkyKey key) throws InterruptedException, NoSuchPackageException {
    if (walkableGraph.isCycle(key)) {
      PackageIdentifier pkgId =
          key instanceof PackagePieceIdentifier.ForBuildFile keyForBuildFile
              ? keyForBuildFile.getPackageIdentifier()
              : (PackageIdentifier) key;
      throw new BuildFileContainsErrorsException(
          pkgId, "Cycle encountered while loading package " + pkgId);
    }
  }
}
