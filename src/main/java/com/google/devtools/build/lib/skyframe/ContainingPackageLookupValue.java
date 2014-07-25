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

import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.util.Objects;

import javax.annotation.Nullable;

/**
 * A value that represents the result of looking for the existence of a package that owns a
 * specific directory path. Compare with {@link PackageLookupValue}, which deals with existence of a
 * specific package.
 *
 * <p> Containing package lookups will always produce a value, for which
 * {@link #getContainingPackageNameOrNull} returns the name of the containing package, if there is
 * one, or {@code null} if there isn't.
 */
public class ContainingPackageLookupValue implements SkyValue {

  @Nullable private final PathFragment containingPackage;

  ContainingPackageLookupValue(@Nullable PathFragment containingPackage) {
    this.containingPackage = containingPackage;
  }

  @Nullable
  public PathFragment getContainingPackageNameOrNull() {
    return containingPackage;
  }

  static SkyKey key(PathFragment directory) {
    return new SkyKey(SkyFunctions.CONTAINING_PACKAGE_LOOKUP, directory);
  }

  @Override
  public int hashCode() {
    return Objects.hash(containingPackage);
  }

  @Override
  public boolean equals(Object other) {
    if (this == other) {
      return true;
    }
    if (!(other instanceof ContainingPackageLookupValue)) {
      return false;
    }
    ContainingPackageLookupValue otherValue = (ContainingPackageLookupValue) other;
    return Objects.equals(containingPackage, otherValue.containingPackage);
  }
}
