// Copyright 2022 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.genquery;

import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.CachingPackageLocator;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.PackageProvider;
import com.google.devtools.build.lib.vfs.Path;
import javax.annotation.Nullable;

/** Provide packages and targets to the query operations using precomputed transitive closure. */
final class GenQueryPackageProvider implements PackageProvider, CachingPackageLocator {

  private final ImmutableMap<PackageIdentifier, Package> pkgMap;
  private final ImmutableMap<Label, Target> labelToTarget;

  GenQueryPackageProvider(
      ImmutableMap<PackageIdentifier, Package> pkgMap, ImmutableMap<Label, Target> labelToTarget) {
    this.pkgMap = pkgMap;
    this.labelToTarget = labelToTarget;
  }

  Predicate<Label> getValidTargetPredicate() {
    return Predicates.in(labelToTarget.keySet());
  }

  @Override
  public Package getPackage(ExtendedEventHandler eventHandler, PackageIdentifier packageId)
      throws NoSuchPackageException {
    Package pkg = pkgMap.get(packageId);
    if (pkg != null) {
      return pkg;
    }
    // Prefer to throw a checked exception on error; malformed genquery should not crash.
    throw new NoSuchPackageException(packageId, "is not within the scope of the query");
  }

  @Override
  public Target getTarget(ExtendedEventHandler eventHandler, Label label)
      throws NoSuchPackageException, NoSuchTargetException {
    // Try to perform only one map lookup in the common case.
    Target target = labelToTarget.get(label);
    if (target != null) {
      return target;
    }
    // Prefer to throw a checked exception on error; malformed genquery should not crash.
    // Because it'd be more valuable, see if NoSuchPackageException should be thrown:
    Package unused = getPackage(eventHandler, label.getPackageIdentifier());
    throw new NoSuchTargetException(label, "is not within the scope of the query");
  }

  @Override
  public boolean isPackage(ExtendedEventHandler eventHandler, PackageIdentifier packageName) {
    throw new UnsupportedOperationException();
  }

  @Nullable
  @Override
  public Path getBuildFileForPackage(PackageIdentifier packageId) {
    Package pkg = pkgMap.get(packageId);
    if (pkg == null) {
      return null;
    }
    return pkg.getBuildFile().getPath();
  }

  @Nullable
  @Override
  public String getBaseNameForLoadedPackage(PackageIdentifier packageName) {
    // TODO(b/123795023): we should have the data here but we don't have all packages for Starlark
    //  loads present here.
    Package pkg = pkgMap.get(packageName);
    return pkg == null ? null : pkg.getBuildFileLabel().getName();
  }
}
