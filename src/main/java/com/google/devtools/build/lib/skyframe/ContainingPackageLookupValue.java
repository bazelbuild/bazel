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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.skyframe.serialization.VisibleForSerialization;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationConstant;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.skyframe.AbstractSkyKey;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import javax.annotation.Nonnull;

/**
 * A value that represents the result of looking for the existence of a package that owns a
 * specific directory path. Compare with {@link PackageLookupValue}, which deals with existence of
 * a specific package.
 */
public abstract class ContainingPackageLookupValue implements SkyValue {

  @SerializationConstant public static final NoContainingPackage NONE = new NoContainingPackage();

  /** Returns whether there is a containing package. */
  public abstract boolean hasContainingPackage();

  public abstract boolean hasProjectFile();

  /** If there is a containing package, returns its name. */
  public abstract PackageIdentifier getContainingPackageName();

  /** If there is a containing package, returns its package root */
  public abstract Root getContainingPackageRoot();

  /**
   * If there is not a containing package, returns a reason why (this is usually the reason the
   * outer-most directory isn't a package).
   */
  public String getReasonForNoContainingPackage() {
    throw new IllegalStateException();
  }

  public static Key key(PackageIdentifier id) {
    Preconditions.checkArgument(!id.getPackageFragment().isAbsolute(), id);
    return Key.create(id);
  }

  static String getErrorMessageForLabelCrossingPackageBoundary(
      Root pkgRoot,
      Label label,
      ContainingPackageLookupValue containingPkgLookupValue) {
    PackageIdentifier containingPkg = containingPkgLookupValue.getContainingPackageName();
    boolean crossesPackageBoundaryBelow =
        containingPkg.getSourceRoot().startsWith(label.getPackageIdentifier().getSourceRoot());
    PathFragment labelNameFragment = PathFragment.create(label.getName());
    String message;
    if (crossesPackageBoundaryBelow) {
      message =
          String.format("Label '%s' is invalid because '%s' is a subpackage", label, containingPkg);
    } else {
      message =
          String.format(
              "Label '%s' is invalid because '%s' is not a package", label, label.getPackageName());
    }

    Root containingRoot = containingPkgLookupValue.getContainingPackageRoot();
    if (pkgRoot.equals(containingRoot)) {
      PathFragment containingPkgFragment = containingPkg.getPackageFragment();
      PathFragment labelNameInContainingPackage =
          crossesPackageBoundaryBelow
              ? labelNameFragment.subFragment(
                  containingPkgFragment.segmentCount()
                      - label.getPackageFragment().segmentCount(),
                  labelNameFragment.segmentCount())
              : label.toPathFragment().relativeTo(containingPkgFragment);
      message += "; perhaps you meant to put the colon here: '";
      if (containingPkg.getRepository().isMain()) {
        message += "//";
      }
      message += containingPkg + ":" + labelNameInContainingPackage + "'?";
    } else {
      message +=
          "; have you deleted "
              + containingPkg
              + "/BUILD? "
              + "If so, use the --deleted_packages="
              + containingPkg
              + " option";
    }
    return message;
  }

  /** {@link com.google.devtools.build.skyframe.SkyKey} for {@code ContainingPackageLookupValue}. */
  @AutoCodec
  public static class Key extends AbstractSkyKey<PackageIdentifier> {
    private static final SkyKeyInterner<Key> interner = SkyKey.newInterner();

    private Key(PackageIdentifier arg) {
      super(arg);
    }

    private static Key create(PackageIdentifier arg) {
      return interner.intern(new Key(arg));
    }

    @VisibleForSerialization
    @AutoCodec.Interner
    static Key intern(Key key) {
      return interner.intern(key);
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.CONTAINING_PACKAGE_LOOKUP;
    }

    @Override
    public SkyKeyInterner<Key> getSkyKeyInterner() {
      return interner;
    }
  }

  public static ContainingPackage withContainingPackage(
      PackageIdentifier pkgId, Root root, boolean hasProjectFile) {
    return new ContainingPackage(pkgId, root, hasProjectFile);
  }

  static ContainingPackageLookupValue noContainingPackage(String reason) {
    return new NoContainingPackage(reason);
  }

  /** Value indicating there is no containing package. */
  public static class NoContainingPackage extends ContainingPackageLookupValue {
    private final String reason;

    private NoContainingPackage() {
      this.reason = null;
    }

    private NoContainingPackage(@Nonnull String reason) {
      this.reason = reason;
    }

    @Override
    public boolean hasContainingPackage() {
      return false;
    }

    @Override
    public boolean hasProjectFile() {
      return false;
    }

    @Override
    public PackageIdentifier getContainingPackageName() {
      throw new IllegalStateException();
    }

    @Override
    public Root getContainingPackageRoot() {
      throw new IllegalStateException();
    }

    @Override
    public String toString() {
      return getClass().getName();
    }

    @Override
    public String getReasonForNoContainingPackage() {
      return reason;
    }
  }

  /** A successful lookup value. */
  @VisibleForTesting
  public static class ContainingPackage extends ContainingPackageLookupValue {
    private final PackageIdentifier containingPackage;
    private final Root containingPackageRoot;

    private final boolean hasProjectFile;

    private ContainingPackage(
        PackageIdentifier containingPackage, Root containingPackageRoot, boolean hasProjectFile) {
      this.containingPackage = containingPackage;
      this.containingPackageRoot = containingPackageRoot;
      this.hasProjectFile = hasProjectFile;
    }

    @Override
    public boolean hasContainingPackage() {
      return true;
    }

    @Override
    public boolean hasProjectFile() {
      return hasProjectFile;
    }

    @Override
    public PackageIdentifier getContainingPackageName() {
      return containingPackage;
    }

    @Override
    public Root getContainingPackageRoot() {
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
          && containingPackageRoot.equals(other.containingPackageRoot)
          && hasProjectFile == other.hasProjectFile;
    }

    @Override
    public int hashCode() {
      return containingPackage.hashCode();
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(this)
          .add("containingPackage", containingPackage)
          .add("containingPackageRoot", containingPackageRoot)
          .add("hasProjectFile", hasProjectFile)
          .toString();
    }
  }
}
