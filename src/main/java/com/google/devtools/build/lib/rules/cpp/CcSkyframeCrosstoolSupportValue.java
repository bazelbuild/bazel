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
package com.google.devtools.build.lib.rules.cpp;

import com.google.common.collect.Interner;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CrosstoolRelease;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Objects;

/**
 * A container for the CROSSTOOL file content.
 *
 * <p>{@link CcSkyframeCrosstoolSupportValue} is created from {@link
 * CcSkyframeCrosstoolSupportFunction} (a {@link SkyFunction}), which is requested from Skyframe by
 * the {@code cc_toolchain}/{@code cc_toolchain_suite} rules. It's done this way because the path
 * depends on both a command line argument and the location of the workspace and the latter is not
 * available either during configuration creation or during the analysis phase.
 */
@AutoCodec
@Immutable
public class CcSkyframeCrosstoolSupportValue implements SkyValue {
  public static final SkyFunctionName SKYFUNCTION =
      SkyFunctionName.createHermetic("CROSSTOOL_SUPPORT");

  /** {@link SkyKey} for {@link CcSkyframeCrosstoolSupportValue}. */
  @Immutable
  @AutoCodec
  public static class Key implements SkyKey {
    private static final Interner<Key> interner = BlazeInterners.newWeakInterner();

    private final PackageIdentifier packageWithCrosstoolInIt;

    private Key(PackageIdentifier packageWithCrosstoolInIt) {
      this.packageWithCrosstoolInIt = packageWithCrosstoolInIt;
    }

    @AutoCodec.Instantiator
    @AutoCodec.VisibleForSerialization
    static Key of(PackageIdentifier packageWithCrosstoolInIt) {
      return interner.intern(new Key(packageWithCrosstoolInIt));
    }

    public PackageIdentifier getPackageWithCrosstoolInIt() {
      return packageWithCrosstoolInIt;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (!(o instanceof Key)) {
        return false;
      }
      Key key = (Key) o;
      return Objects.equals(packageWithCrosstoolInIt, key.packageWithCrosstoolInIt);
    }

    @Override
    public int hashCode() {

      return Objects.hash(packageWithCrosstoolInIt);
    }

    @Override
    public SkyFunctionName functionName() {
      return SKYFUNCTION;
    }
  }

  private final CrosstoolRelease crosstoolRelease;

  CcSkyframeCrosstoolSupportValue(CrosstoolRelease crosstoolRelease) {
    this.crosstoolRelease = crosstoolRelease;
  }

  public CrosstoolRelease getCrosstoolRelease() {
    return crosstoolRelease;
  }

  public static SkyKey key(PackageIdentifier packageWithCrosstoolInIt) {
    return Key.of(packageWithCrosstoolInIt);
  }
}
