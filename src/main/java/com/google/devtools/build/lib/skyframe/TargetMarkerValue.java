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

import com.google.common.base.Preconditions;
import com.google.common.collect.Interner;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.skyframe.AbstractSkyKey;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyValue;

/**
 * Value represents visited target in the Skyframe graph after error checking.
 */
@Immutable
@ThreadSafe
public final class TargetMarkerValue implements SkyValue {

  // Note that this value does not guarantee singleton-like reference equality because we use Java
  // deserialization. java deserialization can create other instances.
  @AutoCodec public static final TargetMarkerValue TARGET_MARKER_INSTANCE = new TargetMarkerValue();

  private TargetMarkerValue() {
  }

  @Override
  public boolean equals(Object o) {
    return o instanceof TargetMarkerValue;
  }

  @Override
  public int hashCode() {
    return 42;
  }

  @ThreadSafe
  public static Key key(Label label) {
    Preconditions.checkArgument(!label.getPackageIdentifier().getRepository().isDefault());
    return Key.create(label);
  }

  @AutoCodec.VisibleForSerialization
  @AutoCodec
  static class Key extends AbstractSkyKey<Label> {
    private static final Interner<Key> interner = BlazeInterners.newWeakInterner();

    private Key(Label arg) {
      super(arg);
    }

    @AutoCodec.VisibleForSerialization
    @AutoCodec.Instantiator
    static Key create(Label arg) {
      return interner.intern(new Key(arg));
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.TARGET_MARKER;
    }
  }
}
