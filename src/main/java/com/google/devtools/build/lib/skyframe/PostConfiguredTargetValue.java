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
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Interner;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.skyframe.AbstractSkyKey;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

/**
 * A post-processed ConfiguredTarget which is known to be transitively error-free from action
 * conflict issues.
 */
@AutoCodec(explicitlyAllowClass = RuleConfiguredTarget.class)
class PostConfiguredTargetValue implements SkyValue {

  private final ConfiguredTarget ct;

  PostConfiguredTargetValue(ConfiguredTarget ct) {
    this.ct = Preconditions.checkNotNull(ct);
  }

  public static ImmutableList<SkyKey> keys(Iterable<ConfiguredTargetKey> lacs) {
    ImmutableList.Builder<SkyKey> keys = ImmutableList.builder();
    for (ConfiguredTargetKey lac : lacs) {
      keys.add(key(lac));
    }
    return keys.build();
  }

  public static Key key(ConfiguredTargetKey lac) {
    return Key.create(lac);
  }

  @AutoCodec.VisibleForSerialization
  @AutoCodec
  static class Key extends AbstractSkyKey<ConfiguredTargetKey> {
    private static final Interner<Key> interner = BlazeInterners.newWeakInterner();

    private Key(ConfiguredTargetKey arg) {
      super(arg);
    }

    @AutoCodec.VisibleForSerialization
    @AutoCodec.Instantiator
    static Key create(ConfiguredTargetKey arg) {
      return interner.intern(new Key(arg));
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.POST_CONFIGURED_TARGET;
    }
  }

  public ConfiguredTarget getCt() {
    return ct;
  }
}
