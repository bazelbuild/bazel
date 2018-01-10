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
import com.google.devtools.build.lib.packages.NativeProvider.StructProvider;
import com.google.devtools.build.lib.syntax.Concatable;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalUtils;
import java.util.Arrays;
import java.util.Map;

/**
 * A provider instance of either 1) a Skylark-defined provider ({@link SkylarkInfo}), or 2) the
 * built-in "struct" type ({@link NativeProvider#STRUCT}).
 *
 * <p>There are two concrete subclasses corresponding to two different implementation strategies.
 * One is map-based and schemaless, the other has a fixed layout and is more memory-efficient.
 */
public abstract class SkylarkInfo extends Info implements Concatable {

  public SkylarkInfo(Provider provider, Location loc) {
    super(provider, loc);
  }

  public SkylarkInfo(StructProvider provider, String errorMessageFormatForUnknownField) {
    super(provider, errorMessageFormatForUnknownField);
  }

  @Override
  public Concatter getConcatter() {
    return SkylarkInfoConcatter.INSTANCE;
  }

  @Override
  public boolean isImmutable() {
    // If the provider is not yet exported, the hash code of the object is subject to change.
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

  /**
   * Returns all the field values stored in the object, in the canonical order.
   *
   * <p>{@code protected} because this is only used for {@link #isImmutable}. It saves us having to
   * get values one-by-one.
   */
  protected abstract Iterable<Object> getValues();

  /**
   * A {@link SkylarkInfo} implementation that stores its values in a map. This is used for structs
   * and for schemaless Skylark-defined providers.
   */
  static final class MapBackedSkylarkInfo extends SkylarkInfo {

    private final ImmutableMap<String, Object> values;

    public MapBackedSkylarkInfo(Provider provider, Map<String, Object> kwargs, Location loc) {
      super(provider, loc);
      this.values = copyValues(kwargs);
    }

    public MapBackedSkylarkInfo(
        StructProvider provider,
        Map<String, Object> values,
        String errorMessageFormatForUnknownField) {
      super(provider, errorMessageFormatForUnknownField);
      this.values = copyValues(values);
    }

    @Override
    public boolean hasField(String name) {
      return values.containsKey(name);
    }

    @Override
    public Object getValue(String name) {
      return values.get(name);
    }

    @Override
    public ImmutableCollection<String> getFieldNames() {
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
    public boolean hasField(String name) {
      Integer index = layout.get(name);
      return index != null && values[index] != null;
    }

    @Override
    public ImmutableCollection<String> getFieldNames() {
      ImmutableSet.Builder<String> result = ImmutableSet.builder();
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
      // Casts are safe because SkylarkInfoConcatter is only used by SkylarkInfo.
      SkylarkInfo lval = (SkylarkInfo) left;
      SkylarkInfo rval = (SkylarkInfo) right;
      Provider provider = lval.getProvider();
      if (!provider.equals(rval.getProvider())) {
        throw new EvalException(
            loc,
            String.format(
                "Cannot use '+' operator on instances of different providers (%s and %s)",
                provider.getPrintableName(), rval.getProvider().getPrintableName()));
      }
      SetView<String> commonFields =
          Sets.intersection(
              ImmutableSet.copyOf(lval.getFieldNames()), ImmutableSet.copyOf(rval.getFieldNames()));
      if (!commonFields.isEmpty()) {
        throw new EvalException(
            loc,
            "Cannot use '+' operator on provider instances with overlapping field(s): "
                + Joiner.on(",").join(commonFields));
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
      for (String field : lval.getFieldNames()) {
        newValues.put(field, lval.getValue(field));
      }
      for (String field : rval.getFieldNames()) {
        newValues.put(field, rval.getValue(field));
      }
      return new MapBackedSkylarkInfo(provider, newValues.build(), loc);
    }
  }
}
