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
import com.google.devtools.build.lib.syntax.Concatable;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalUtils;
import java.util.Arrays;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * A standard implementation for provider instances.
 *
 * <p>Instances may be either schemaless or schemaful (corresponding to the two different concrete
 * implementing classes). Schemaless instances are map-based, while schemaful instances have a fixed
 * layout and array and are therefore more efficient.
 */
public abstract class SkylarkInfo extends Info implements Concatable {

  // Private because this should not be subclassed outside this file.
  private SkylarkInfo(
      Provider provider,
      @Nullable Location loc,
      @Nullable String errorMessageFormatForUnknownField) {
    super(provider, loc, errorMessageFormatForUnknownField);
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
   * Creates a schemaless (map-based) provider instance with the given provider type and field
   * values.
   *
   * <p>{@code loc} is the creation location for this instance. Built-in provider instances may use
   * {@link Location#BUILTIN}, which is the default if null.
   */
  public static SkylarkInfo createSchemaless(
      Provider provider, Map<String, Object> values, @Nullable Location loc) {
    return new MapBackedSkylarkInfo(
        provider, values, loc, /*errorMessageFormatForUnknownField=*/ null);
  }

  /**
   * Creates a schemaless (map-based) provider instance with the given provider type, field values,
   * and unknown-field error message.
   *
   * <p>The creation location will be {@link Location#BUILTIN}.
   *
   * <p>This is used to create structs for special purposes, such as {@code ctx.attr} and the
   * {@code native} module.
   */
  // TODO(bazel-team): Make the special structs that need a custom error message use a different
  // provider (subclassing NativeProvider) and a different Info implementation. Then remove this
  // functionality, thereby saving a string pointer field for the majority of providers that don't
  // need it.
  public static SkylarkInfo createSchemalessWithCustomMessage(
      Provider provider, Map<String, Object> values, String errorMessageFormatForUnknownField) {
    Preconditions.checkNotNull(errorMessageFormatForUnknownField);
    return new MapBackedSkylarkInfo(
        provider, values, Location.BUILTIN, errorMessageFormatForUnknownField);
  }

  /**
   * Creates a schemaful (array-based) provider instance with the given provider type and values,
   * where the layout is specified by the provider type.
   *
   * <p>This factory method requires a {@link SkylarkProvider} in order to retrieve the layout. To
   * obtain a schemaful provider instance for another kind of provider type, use {@link
   * #createSchemafulWithCustomLayout} instead.
   *
   * <p>{@code provider} must be schemaful. The order of {@code values} must correspond to {@code
   * provider}'s layout.
   *
   * <p>{@code loc} is the creation location for this instance. Built-in provider instances may use
   * {@link Location#BUILTIN}, which is the default if null.
   */
  public static SkylarkInfo createSchemaful(
      SkylarkProvider provider, Object[] values, @Nullable Location loc) {
    Preconditions.checkArgument(provider.getLayout() != null, "provider cannot be schemaless");
    return new CompactSkylarkInfo(
        provider, provider.getLayout(), values, loc, /*errorMessageFormatForUnknownField=*/ null);
  }

  /**
   * Creates a schemaful (array-based) provider instance with the given provider type and values,
   * and with a custom layout.
   *
   * <p>The order of the values must correspond to the given layout. Any layout specified by the
   * provider (i.e., if it is a schemaful {@link SkylarkProvider}) is ignored.
   *
   * <p>{@code loc} is the creation location for this instance. Built-in provider instances may use
   * {@link Location#BUILTIN}, which is the default if null.
   */
  public static SkylarkInfo createSchemafulWithCustomLayout(
      Provider provider,
      ImmutableMap<String, Integer> layout,
      Object[] values,
      @Nullable Location loc) {
    return new CompactSkylarkInfo(
        provider, layout, values, loc, /*errorMessageFormatForUnknownField=*/ null);
  }

  /** Returns the layout for this provider if it is schemaful, null otherwise. */
  public abstract ImmutableMap<String, Integer> getLayout();

  /** Returns true if this provider is schemaful (array-based), false otherwise. */
  public boolean isCompact() {
    return getLayout() != null;
  }

  /** A {@link SkylarkInfo} implementation that stores its values in a map. */
  private static final class MapBackedSkylarkInfo extends SkylarkInfo {

    private final ImmutableMap<String, Object> values;

    MapBackedSkylarkInfo(
        Provider provider,
        Map<String, Object> values,
        @Nullable Location loc,
        @Nullable String errorMessageFormatForUnknownField) {
      super(provider, loc, errorMessageFormatForUnknownField);
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

    @Override
    public ImmutableMap<String, Integer> getLayout() {
      return null;
    }
  }

  /** A {@link SkylarkInfo} implementation that stores its values in array to save space. */
  private static final class CompactSkylarkInfo extends SkylarkInfo implements Concatable {

    private final ImmutableMap<String, Integer> layout;
    /** Treated as immutable. */
    private final Object[] values;

    CompactSkylarkInfo(
        Provider provider,
        ImmutableMap<String, Integer> layout,
        Object[] values,
        @Nullable Location loc,
        @Nullable String errorMessageFormatForUnknownField) {
      super(provider, loc, errorMessageFormatForUnknownField);
      this.layout = Preconditions.checkNotNull(layout);
      Preconditions.checkArgument(
          layout.size() == values.length,
          "Layout has length %s, but number of given values was %s", layout.size(), values.length);
      this.values = Arrays.copyOf(values, values.length);
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

    @Override
    public ImmutableMap<String, Integer> getLayout() {
      return layout;
    }
  }

  /** Concatter for concrete {@link SkylarkInfo} subclasses. */
  private static final class SkylarkInfoConcatter implements Concatable.Concatter {
    private static final SkylarkInfoConcatter INSTANCE = new SkylarkInfoConcatter();

    private SkylarkInfoConcatter() {}

    @Override
    public Concatable concat(Concatable left, Concatable right, Location loc) throws EvalException {
      // Casts are safe because SkylarkInfoConcatter is only used by SkylarkInfo.
      SkylarkInfo leftInfo = (SkylarkInfo) left;
      SkylarkInfo rightInfo = (SkylarkInfo) right;
      Provider provider = leftInfo.getProvider();
      if (!provider.equals(rightInfo.getProvider())) {
        throw new EvalException(
            loc,
            String.format(
                "Cannot use '+' operator on instances of different providers (%s and %s)",
                provider.getPrintableName(), rightInfo.getProvider().getPrintableName()));
      }
      SetView<String> commonFields =
          Sets.intersection(
              ImmutableSet.copyOf(leftInfo.getFieldNames()),
              ImmutableSet.copyOf(rightInfo.getFieldNames()));
      if (!commonFields.isEmpty()) {
        throw new EvalException(
            loc,
            "Cannot use '+' operator on provider instances with overlapping field(s): "
                + Joiner.on(",").join(commonFields));
      }
      // Keep homogeneous compact concatenations compact.
      if (leftInfo instanceof CompactSkylarkInfo && rightInfo instanceof CompactSkylarkInfo) {
        CompactSkylarkInfo compactLeft = (CompactSkylarkInfo) leftInfo;
        CompactSkylarkInfo compactRight = (CompactSkylarkInfo) rightInfo;
        ImmutableMap<String, Integer> layout = compactLeft.layout;
        if (layout == compactRight.layout) {
          int nvals = layout.size();
          Object[] newValues = new Object[nvals];
          for (int i = 0; i < nvals; i++) {
            newValues[i] =
                (compactLeft.values[i] != null) ? compactLeft.values[i] : compactRight.values[i];
          }
          return createSchemafulWithCustomLayout(provider, layout, newValues, loc);
        }
      }
      // Fall back on making a map-based instance.
      ImmutableMap.Builder<String, Object> newValues = ImmutableMap.builder();
      for (String field : leftInfo.getFieldNames()) {
        newValues.put(field, leftInfo.getValue(field));
      }
      for (String field : rightInfo.getFieldNames()) {
        newValues.put(field, rightInfo.getValue(field));
      }
      return createSchemaless(provider, newValues.build(), loc);
    }
  }
}
