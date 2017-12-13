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
package com.google.devtools.build.lib.packages;

import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Sets;
import com.google.common.collect.Sets.SetView;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.NativeProvider.StructConstructor;
import com.google.devtools.build.lib.syntax.Concatable;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalUtils;
import java.util.Arrays;
import java.util.Map;

/** Implementation of {@link Info} created from Skylark. */
public abstract class SkylarkInfo extends Info implements Concatable {

  public SkylarkInfo(Provider provider, Location loc) {
    super(provider, loc);
  }

  public SkylarkInfo(StructConstructor provider, String message) {
    super(provider, message);
  }

  @Override
  public Concatter getConcatter() {
    return SkylarkInfoConcatter.INSTANCE;
  }

  @Override
  public boolean isImmutable() {
    // If the provider is not yet exported the hash code of the object is subject to change
    if (!getProvider().isExported()) {
      return false;
    }
    for (Object item : getValues()) {
      if (item != null && !EvalUtils.isImmutable(item)) {
        return false;
      }
    }
    return true;
  }

  /** Return all the values stored in the object. */
  protected abstract Iterable<Object> getValues();

  /**
   * {@link SkylarkInfo} implementation that stores its values in a map. This is mainly used for the
   * Skylark {@code struct()} constructor.
   */
  static final class MapBackedSkylarkInfo extends SkylarkInfo {
    protected final ImmutableMap<String, Object> values;

    public MapBackedSkylarkInfo(Provider provider, Map<String, Object> kwargs, Location loc) {
      super(provider, loc);
      this.values = copyValues(kwargs);
    }

    public MapBackedSkylarkInfo(
        StructConstructor provider, Map<String, Object> values, String message) {
      super(provider, message);
      this.values = copyValues(values);
    }

    @Override
    public Object getValue(String name) {
      return values.get(name);
    }

    @Override
    public boolean hasKey(String name) {
      return values.containsKey(name);
    }

    @Override
    public ImmutableCollection<String> getKeys() {
      return values.keySet();
    }

    @Override
    protected Iterable<Object> getValues() {
      return values.values();
    }
  }

  /** Create a {@link SkylarkInfo} instance from a provider and a map. */
  public static SkylarkInfo fromMap(Provider provider, Map<String, Object> values, Location loc) {
    return new MapBackedSkylarkInfo(provider, values, loc);
  }

  /** Implementation of {@link SkylarkInfo} that stores its values in array to save space. */
  static final class CompactSkylarkInfo extends SkylarkInfo implements Concatable {
    private final ImmutableMap<String, Integer> layout;
    private final Object[] values;

    public CompactSkylarkInfo(
        Provider provider, ImmutableMap<String, Integer> layout, Object[] values, Location loc) {
      super(provider, loc);
      Preconditions.checkState(layout.size() == values.length);
      this.layout = layout;
      this.values = values;
    }

    @Override
    public Concatter getConcatter() {
      return SkylarkInfoConcatter.INSTANCE;
    }

    @Override
    public Object getValue(String name) {
      Integer index = layout.get(name);
      if (index == null) {
        return null;
      }
      return values[index];
    }

    @Override
    public boolean hasKey(String name) {
      Integer index = layout.get(name);
      return index != null && values[index] != null;
    }

    @Override
    public ImmutableCollection<String> getKeys() {
      ImmutableSet.Builder<String> result = new ImmutableSet.Builder();
      for (Map.Entry<String, Integer> entry : layout.entrySet()) {
        if (values[entry.getValue()] != null) {
          result.add(entry.getKey());
        }
      }
      return result.build();
    }

    @Override
    protected Iterable<Object> getValues() {
      return Arrays.asList(values);
    }
  }

  /** Concatter for concrete {@link SkylarkInfo} subclasses. */
  private static final class SkylarkInfoConcatter implements Concatable.Concatter {
    private static final SkylarkInfoConcatter INSTANCE = new SkylarkInfoConcatter();

    private SkylarkInfoConcatter() {}

    @Override
    public Concatable concat(Concatable left, Concatable right, Location loc) throws EvalException {
      SkylarkInfo lval = (SkylarkInfo) left;
      SkylarkInfo rval = (SkylarkInfo) right;
      Provider provider = lval.getProvider();
      if (!provider.equals(rval.getProvider())) {
        throw new EvalException(
            loc,
            String.format(
                "Cannot concat %s with %s",
                provider.getPrintableName(), rval.getProvider().getPrintableName()));
      }
      SetView<String> commonFields =
          Sets.intersection(
              ImmutableSet.copyOf(lval.getKeys()), ImmutableSet.copyOf(rval.getKeys()));
      if (!commonFields.isEmpty()) {
        throw new EvalException(
            loc,
            "Cannot concat structs with common field(s): " + Joiner.on(",").join(commonFields));
      }
      // Keep homogeneous compact concatenations compact.
      if (lval instanceof CompactSkylarkInfo
          && rval instanceof CompactSkylarkInfo
          && ((CompactSkylarkInfo) lval).layout == ((CompactSkylarkInfo) rval).layout) {
        CompactSkylarkInfo compactLeft = (CompactSkylarkInfo) lval;
        CompactSkylarkInfo compactRight = (CompactSkylarkInfo) rval;
        int nvals = compactLeft.layout.size();
        Preconditions.checkState(nvals == compactRight.layout.size());
        Object[] newValues = new Object[nvals];
        for (int i = 0; i < nvals; i++) {
          newValues[i] =
              (compactLeft.values[i] != null) ? compactLeft.values[i] : compactRight.values[i];
        }
        return new CompactSkylarkInfo(
            compactLeft.getProvider(), compactLeft.layout, newValues, loc);
      }
      ImmutableMap.Builder<String, Object> newValues = ImmutableMap.builder();
      for (String key : lval.getKeys()) {
        newValues.put(key, lval.getValue(key));
      }
      for (String key : rval.getKeys()) {
        newValues.put(key, rval.getValue(key));
      }
      return new MapBackedSkylarkInfo(provider, newValues.build(), loc);
    }
  }
}
