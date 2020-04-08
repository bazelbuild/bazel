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

import com.google.common.collect.Interner;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.skyframe.AbstractSkyKey;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyValue;

/** Dummy {@link SkyValue} for {@link CollectTestSuitesInPackageFunction}. */
public class CollectTestSuitesInPackageValue implements SkyValue {
  @AutoCodec
  public static final CollectTestSuitesInPackageValue INSTANCE =
      new CollectTestSuitesInPackageValue();

  private CollectTestSuitesInPackageValue() {}

  /**
   * Creates a key for evaluation of {@link CollectTargetsInPackageFunction}. See that class's
   * comment for what callers should have done beforehand.
   */
  public static Key key(PackageIdentifier packageId) {
    return Key.create(packageId);
  }

  /**
   * {@link com/google/devtools/build/lib/skyframe/CollectTestSuitesInPackageValue.java used only in
   * javadoc: com.google.devtools.build.skyframe.SkyKey} argument.
   */
  @AutoCodec
  public static class Key extends AbstractSkyKey<PackageIdentifier> {
    private static final Interner<Key> interner = BlazeInterners.newWeakInterner();

    private Key(PackageIdentifier arg) {
      super(arg);
    }

    @AutoCodec.VisibleForSerialization
    @AutoCodec.Instantiator
    static Key create(PackageIdentifier arg) {
      return interner.intern(new Key(arg));
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.COLLECT_TEST_SUITES_IN_PACKAGE;
    }
  }
}
