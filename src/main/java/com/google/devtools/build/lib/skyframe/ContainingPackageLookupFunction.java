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
import com.google.devtools.build.lib.skyframe.PackageLookupValue.ErrorReason;
import com.google.devtools.build.lib.skyframe.PackageLookupValue.IncorrectRepositoryReferencePackageLookupValue;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import javax.annotation.Nullable;

/**
 * SkyFunction for {@link ContainingPackageLookupValue}s.
 */
public class ContainingPackageLookupFunction implements SkyFunction {

  @Override
  @Nullable
  public SkyValue compute(SkyKey skyKey, Environment env) throws InterruptedException {
    PackageIdentifier dir = (PackageIdentifier) skyKey.argument();
    PackageLookupValue pkgLookupValue =
        (PackageLookupValue) env.getValue(PackageLookupValue.key(dir));
    if (pkgLookupValue == null) {
      return null;
    }

    if (pkgLookupValue.packageExists()) {
      return ContainingPackageLookupValue.withContainingPackage(
          dir, pkgLookupValue.getRoot(), pkgLookupValue.hasProjectFile());
    }

    // Does the requested package cross into a sub-repository, which we should report via the
    // correct package identifier?
    if (pkgLookupValue instanceof IncorrectRepositoryReferencePackageLookupValue) {
      IncorrectRepositoryReferencePackageLookupValue incorrectPackageLookupValue =
          (IncorrectRepositoryReferencePackageLookupValue) pkgLookupValue;
      PackageIdentifier correctPackageIdentifier =
          incorrectPackageLookupValue.getCorrectedPackageIdentifier();
      return env.getValue(ContainingPackageLookupValue.key(correctPackageIdentifier));
    }

    if (ErrorReason.REPOSITORY_NOT_FOUND.equals(pkgLookupValue.getErrorReason())) {
      return ContainingPackageLookupValue.noContainingPackage(pkgLookupValue.getErrorMsg());
    }
    PathFragment parentDir = dir.getPackageFragment().getParentDirectory();
    if (parentDir == null) {
      return ContainingPackageLookupValue.NONE;
    }
    PackageIdentifier parentId = PackageIdentifier.create(dir.getRepository(), parentDir);
    return env.getValue(ContainingPackageLookupValue.key(parentId));
  }
}
