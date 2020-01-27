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
package com.google.devtools.build.lib.skyframe;

import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import javax.annotation.Nullable;

/** {@link SkyFunction} for {@link PackageErrorMessageValue}. */
public class PackageErrorMessageFunction implements SkyFunction {
  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) throws InterruptedException {
    PackageIdentifier pkgId = (PackageIdentifier) skyKey.argument();
    PackageValue pkgValue;
    try {
      pkgValue =
          (PackageValue) env.getValueOrThrow(PackageValue.key(pkgId), NoSuchPackageException.class);
    } catch (NoSuchPackageException e) {
      return PackageErrorMessageValue.ofNoSuchPackageException(e.getMessage());
    }
    if (pkgValue == null) {
      return null;
    }
    Package pkg = pkgValue.getPackage();
    return pkg.containsErrors()
        ? PackageErrorMessageValue.ofPackageWithErrors()
        : PackageErrorMessageValue.ofPackageWithNoErrors();
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }
}
