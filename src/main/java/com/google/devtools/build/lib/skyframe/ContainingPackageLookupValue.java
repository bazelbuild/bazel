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
package com.google.devtools.build.lib.skyframe;

import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

/**
 * A value that represents the result of looking for the existence of a package that owns a
 * specific directory path. Compare with {@link PackageLookupValue}, which deals with existence of
 * a specific package.
 */
public abstract class ContainingPackageLookupValue implements SkyValue {

  public static final NoContainingPackage NONE = new NoContainingPackage();

  /** Returns whether there is a containing package. */
  public abstract boolean hasContainingPackage();

  /** If there is a containing package, returns its name. */
  public abstract PackageIdentifier getContainingPackageName();

  /** If there is a containing package, returns its package root */
  public abstract Path getContainingPackageRoot();

  public static SkyKey key(PackageIdentifier id) {
    Preconditions.checkArgument(!id.getPackageFragment().isAbsolute(), id);
    return SkyKey.create(SkyFunctions.CONTAINING_PACKAGE_LOOKUP, id);
  }

  public static ContainingPackage withContainingPackage(PackageIdentifier pkgId, Path root) {
    return new ContainingPackage(pkgId, root);
  }

  /** Value indicating there is no containing package. */
  public static class NoContainingPackage extends ContainingPackageLookupValue {

    private NoContainingPackage() {}

    @Override
    public boolean hasContainingPackage() {
      return false;
    }

    @Override
    public PackageIdentifier getContainingPackageName() {
      throw new IllegalStateException();
    }

    @Override
    public Path getContainingPackageRoot() {
      throw new IllegalStateException();
    }
  }

  /** A successful lookup value. */
  public static class ContainingPackage extends ContainingPackageLookupValue {
    private final PackageIdentifier containingPackage;
    private final Path containingPackageRoot;

    private ContainingPackage(PackageIdentifier pkgId, Path containingPackageRoot) {
      this.containingPackage = pkgId;
      this.containingPackageRoot = containingPackageRoot;
    }

    @Override
    public boolean hasContainingPackage() {
      return true;
    }

    @Override
    public PackageIdentifier getContainingPackageName() {
      return containingPackage;
    }

    @Override
    public Path getContainingPackageRoot() {
      return containingPackageRoot;
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      }
      if (!(obj instanceof ContainingPackage)) {
        return false;
      }
      ContainingPackage other = (ContainingPackage) obj;
      return containingPackage.equals(other.containingPackage)
          && containingPackageRoot.equals(other.containingPackageRoot);
    }

    @Override
    public int hashCode() {
      return containingPackage.hashCode();
    }
  }
}
