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

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Interner;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import java.util.Arrays;
import java.util.Map.Entry;
import javax.annotation.Nullable;
import javax.annotation.concurrent.Immutable;

/**
 * Provides a mapping between a TransitiveInfoProvider class and an instance.
 *
 * <p>This class implements a map where it is expected that a lot of the key sets will be the same.
 * These key sets are shared and an offset table of indices is computed. Each provider map instance
 * thus contains only a reference to the shared offset table, and a plain array of providers.
 */
@Immutable
final class TransitiveInfoProviderMapOffsetBased implements TransitiveInfoProviderMap {
  private static final Interner<OffsetTable> offsetTables = BlazeInterners.newWeakInterner();

  private final OffsetTable offsetTable;
  private final TransitiveInfoProvider[] providers;

  private static final class OffsetTable {
    private final Class<? extends TransitiveInfoProvider>[] providerClasses;
    // Keep a map around to speed up get lookups for larger maps.
    // We make this value lazy to avoid computing for values that end up being thrown away
    // during interning anyway (the majority).
    private volatile ImmutableMap<Class<? extends TransitiveInfoProvider>, Integer> indexMap;

    OffsetTable(Class<? extends TransitiveInfoProvider>[] providerClasses) {
      this.providerClasses = providerClasses;
    }

    private ImmutableMap<Class<? extends TransitiveInfoProvider>, Integer> getIndexMap() {
      if (indexMap == null) {
        synchronized (this) {
          if (indexMap == null) {
            ImmutableMap.Builder<Class<? extends TransitiveInfoProvider>, Integer> builder =
                ImmutableMap.builder();
            for (int i = 0; i < providerClasses.length; ++i) {
              builder.put(providerClasses[i], i);
            }
            this.indexMap = builder.build();
          }
        }
      }
      return indexMap;
    }

    int getOffsetForClass(Class effectiveClass) {
      return getIndexMap().getOrDefault(effectiveClass, -1);
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (!(o instanceof OffsetTable)) {
        return false;
      }
      OffsetTable that = (OffsetTable) o;
      return Arrays.equals(this.providerClasses, that.providerClasses);
    }

    @Override
    public int hashCode() {
      return Arrays.hashCode(providerClasses);
    }
  }

  TransitiveInfoProviderMapOffsetBased(
      ImmutableMap<Class<? extends TransitiveInfoProvider>, TransitiveInfoProvider> map) {
    int count = map.size();
    Class<? extends TransitiveInfoProvider>[] providerClasses = new Class[count];
    this.providers = new TransitiveInfoProvider[count];
    int i = 0;
    for (Entry<Class<? extends TransitiveInfoProvider>, TransitiveInfoProvider> entry :
        map.entrySet()) {
      providerClasses[i] = entry.getKey();
      providers[i] = entry.getValue();
      ++i;
    }
    OffsetTable offsetTable = new OffsetTable(providerClasses);
    this.offsetTable = offsetTables.intern(offsetTable);
  }

  /** Returns the instance for the provided providerClass, or <tt>null</tt> if not present. */
  @Override
  @Nullable
  public <P extends TransitiveInfoProvider> P getProvider(Class<P> providerClass) {
    Class effectiveClass = TransitiveInfoProviderEffectiveClassHelper.get(providerClass);
    int offset = offsetTable.getOffsetForClass(effectiveClass);
    return offset >= 0 ? (P) providers[offset] : null;
  }

  @Override
  public int getProviderCount() {
    return providers.length;
  }

  @Override
  public Class<? extends TransitiveInfoProvider> getProviderClassAt(int i) {
    return offsetTable.providerClasses[i];
  }

  @Override
  public TransitiveInfoProvider getProviderAt(int i) {
    return providers[i];
  }
}
