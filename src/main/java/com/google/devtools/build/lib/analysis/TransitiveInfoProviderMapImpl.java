// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.collect.ImmutableSharedKeyMap;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Implementation of {@link TransitiveInfoProvider} that uses {@link ImmutableSharedKeyMap}. For
 * memory efficiency, inheritance is used instead of aggregation as an implementation detail.
 */
@AutoCodec
class TransitiveInfoProviderMapImpl extends ImmutableSharedKeyMap<Object, Object>
    implements TransitiveInfoProviderMap {

  @VisibleForSerialization
  @AutoCodec.Instantiator
  TransitiveInfoProviderMapImpl(Object[] keys, Object[] values) {
    super(keys, values);
  }

  static TransitiveInfoProviderMapImpl create(Map<Object, Object> map) {
    int count = map.size();
    Object[] keys = new Object[count];
    Object[] values = new Object[count];
    int i = 0;
    for (Map.Entry<Object, Object> entry : map.entrySet()) {
      keys[i] = entry.getKey();
      values[i] = entry.getValue();
      ++i;
    }
    Preconditions.checkArgument(keys.length == values.length);
    return new TransitiveInfoProviderMapImpl(keys, values);
  }

  @SuppressWarnings("unchecked")
  @Nullable
  @Override
  public <P extends TransitiveInfoProvider> P getProvider(Class<P> providerClass) {
    Class<? extends TransitiveInfoProvider> effectiveClass =
        TransitiveInfoProviderEffectiveClassHelper.get(providerClass);
    return (P) get(effectiveClass);
  }

  @Nullable
  @Override
  public Info get(Provider.Key key) {
    return (Info) super.get(key);
  }

  @Nullable
  @Override
  public Object get(String legacyKey) {
    return super.get(legacyKey);
  }

  @Override
  public int getProviderCount() {
    return size();
  }

  @Override
  public Object getProviderKeyAt(int i) {
    return keyAt(i);
  }

  @Override
  public Object getProviderInstanceAt(int i) {
    return valueAt(i);
  }
}
