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
package com.google.devtools.build.lib.skyframe;

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.io.InconsistentFilesystemException;
import com.google.devtools.build.lib.packages.BuildFileNotFoundException;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyKey;
import javax.annotation.Nullable;

/** Holds the utility method {@link #loadTarget}. */
public class TargetLoadingUtil {

  private TargetLoadingUtil() {}

  /**
   * Loads the {@link Target} specified by the {@link Label} by loading the label's {@link Package},
   * in the context of a Skyframe evaluation.
   *
   * <p>Establishes all Skyframe dependencies needed for incremental correctness.
   *
   * <p>Returns {@link TargetAndErrorIfAny} if no dep was mising; otherwise, returns the {@link
   * SkyKey} specifying the missing dep.
   */
  public static Object loadTarget(Environment env, Label label)
      throws NoSuchTargetException, NoSuchPackageException, InterruptedException {
    if (label.getName().contains("/")) {
      // This target is in a subdirectory, therefore it could potentially be invalidated by
      // a new BUILD file appearing in the hierarchy.
      PathFragment containingDirectory = getContainingDirectory(label);
      PackageIdentifier newPkgId =
          PackageIdentifier.create(label.getRepository(), containingDirectory);
      ContainingPackageLookupValue containingPackageLookupValue;
      SkyKey containingPackageKey = ContainingPackageLookupValue.key(newPkgId);
      try {
        containingPackageLookupValue =
            (ContainingPackageLookupValue)
                env.getValueOrThrow(
                    containingPackageKey,
                    BuildFileNotFoundException.class,
                    InconsistentFilesystemException.class);
      } catch (InconsistentFilesystemException e) {
        throw new NoSuchTargetException(label, e.getMessage());
      }
      if (containingPackageLookupValue == null) {
        return containingPackageKey;
      }

      if (!containingPackageLookupValue.hasContainingPackage()) {
        // This means the label's package doesn't exist. E.g. there is no package 'a' and we are
        // trying to build the target for label 'a:b/foo'.
        throw new BuildFileNotFoundException(
            label.getPackageIdentifier(),
            "BUILD file not found on package path for '"
                + label.getPackageFragment().getPathString()
                + "'");
      }
      if (!containingPackageLookupValue
          .getContainingPackageName()
          .equals(label.getPackageIdentifier())) {
        throw new NoSuchTargetException(
            label,
            String.format(
                "Label '%s' crosses boundary of subpackage '%s'",
                label, containingPackageLookupValue.getContainingPackageName()));
      }
    }

    SkyKey packageKey = label.getPackageIdentifier();
    PackageValue packageValue =
        (PackageValue) env.getValueOrThrow(packageKey, NoSuchPackageException.class);
    if (packageValue == null) {
      return packageKey;
    }

    Package pkg = packageValue.getPackage();
    Target target = pkg.getTarget(label.getName());
    NoSuchTargetException error = pkg.containsErrors() ? new NoSuchTargetException(target) : null;
    return new TargetAndErrorIfAny(
        /* packageLoadedSuccessfully= */ !pkg.containsErrors(), error, target);
  }

  private static PathFragment getContainingDirectory(Label label) {
    PathFragment pkg = label.getPackageFragment();
    String name = label.getName();
    return name.equals(".") ? pkg : pkg.getRelative(name).getParentDirectory();
  }

  /**
   * Returned by {@link #loadTarget}. Contains the loaded {@link Target} and also specifies whether
   * there were errors during the target's package load.
   */
  public static class TargetAndErrorIfAny {

    private final boolean packageLoadedSuccessfully;
    @Nullable private final NoSuchTargetException errorLoadingTarget;
    private final Target target;

    @VisibleForTesting
    TargetAndErrorIfAny(
        boolean packageLoadedSuccessfully,
        @Nullable NoSuchTargetException errorLoadingTarget,
        Target target) {
      this.packageLoadedSuccessfully = packageLoadedSuccessfully;
      this.errorLoadingTarget = errorLoadingTarget;
      this.target = target;
    }

    public boolean isPackageLoadedSuccessfully() {
      return packageLoadedSuccessfully;
    }

    @Nullable
    public NoSuchTargetException getErrorLoadingTarget() {
      return errorLoadingTarget;
    }

    public Target getTarget() {
      return target;
    }
  }
}
