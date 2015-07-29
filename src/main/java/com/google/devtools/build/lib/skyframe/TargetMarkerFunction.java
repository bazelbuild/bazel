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
package com.google.devtools.build.lib.skyframe;

import com.google.devtools.build.lib.packages.BuildFileNotFoundException;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.PackageIdentifier;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

/**
 * A SkyFunction for {@link TargetMarkerValue}s. Returns a {@link
 * TargetMarkerValue#TARGET_MARKER_INSTANCE} if the {@link Label} in the {@link SkyKey}
 * specifies a {@link Package} that exists and a {@link Target} that exists in that package. The
 * package may have errors.
 */
public final class TargetMarkerFunction implements SkyFunction {

  public TargetMarkerFunction() {
  }

  @Override
  public SkyValue compute(SkyKey key, Environment env) throws TargetMarkerFunctionException {
    Label label = (Label) key.argument();
    PathFragment pkgForLabel = label.getPackageFragment();

    if (label.getName().contains("/")) {
      // This target is in a subdirectory, therefore it could potentially be invalidated by
      // a new BUILD file appearing in the hierarchy.
      PathFragment containingDirectory = label.toPathFragment().getParentDirectory();
      ContainingPackageLookupValue containingPackageLookupValue = null;
      try {
        PackageIdentifier newPkgId = new PackageIdentifier(
            label.getPackageIdentifier().getRepository(), containingDirectory);
        containingPackageLookupValue = (ContainingPackageLookupValue) env.getValueOrThrow(
            ContainingPackageLookupValue.key(newPkgId),
            BuildFileNotFoundException.class, InconsistentFilesystemException.class);
      } catch (BuildFileNotFoundException e) {
        // Thrown when there are IO errors looking for BUILD files.
        throw new TargetMarkerFunctionException(e);
      } catch (InconsistentFilesystemException e) {
        throw new TargetMarkerFunctionException(new NoSuchTargetException(label,
            e.getMessage()));
      }
      if (containingPackageLookupValue == null) {
        return null;
      }
      if (!containingPackageLookupValue.hasContainingPackage()) {
        // This means the label's package doesn't exist. E.g. there is no package 'a' and we are
        // trying to build the target for label 'a:b/foo'.
        throw new TargetMarkerFunctionException(new BuildFileNotFoundException(
            label.getPackageIdentifier(), "BUILD file not found on package path for '"
                + pkgForLabel.getPathString() + "'"));
      }
      if (!containingPackageLookupValue.getContainingPackageName().equals(
              label.getPackageIdentifier())) {
        throw new TargetMarkerFunctionException(new NoSuchTargetException(label,
            String.format("Label '%s' crosses boundary of subpackage '%s'", label,
                containingPackageLookupValue.getContainingPackageName())));
      }
    }

    SkyKey pkgSkyKey = PackageValue.key(label.getPackageIdentifier());
    NoSuchPackageException nspe = null;
    Package pkg;
    try {
      PackageValue value = (PackageValue)
          env.getValueOrThrow(pkgSkyKey, NoSuchPackageException.class);
      if (value == null) {
        return null;
      }
      pkg = value.getPackage();
    } catch (NoSuchPackageException e) {
      // For consistency with pre-Skyframe Blaze, we can return a valid Target from a Package
      // containing errors.
      pkg = e.getPackage();
      if (pkg == null) {
        // Re-throw this exception with our key because root causes should be targets, not packages.
        throw new TargetMarkerFunctionException(e);
      }
      nspe = e;
    }

    Target target;
    try {
      target = pkg.getTarget(label.getName());
    } catch (NoSuchTargetException e) {
      throw new TargetMarkerFunctionException(e);
    }

    if (nspe != null) {
      // There is a target, but its package is in error. We rethrow so that the root cause is the
      // target, not the package. Note that targets are only in error when their package is
      // "in error" (because a package is in error if there was an error evaluating the package, or
      // if one of its targets was in error).
      throw new TargetMarkerFunctionException(new NoSuchTargetException(target, nspe));
    }
    return TargetMarkerValue.TARGET_MARKER_INSTANCE;
  }

  @Override
  public String extractTag(SkyKey skyKey) {
    return Label.print((Label) skyKey.argument());
  }

  /**
   * Used to declare all the exception types that can be wrapped in the exception thrown by
   * {@link TargetMarkerFunction#compute}.
   */
  private static final class TargetMarkerFunctionException extends SkyFunctionException {
    public TargetMarkerFunctionException(NoSuchTargetException e) {
      super(e, Transience.PERSISTENT);
    }

    public TargetMarkerFunctionException(NoSuchPackageException e) {
      super(e, Transience.PERSISTENT);
    }
  }
}
