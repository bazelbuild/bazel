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
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.Sets;
import com.google.common.collect.Sets.SetView;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.syntax.BuiltinCallable;
import com.google.devtools.build.lib.syntax.Concatable;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.SkylarkClassObject;
import com.google.devtools.build.lib.syntax.Starlark;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * A standard implementation for provider instances.
 *
 * <p>Instances may be either schemaless or schemaful (corresponding to the two different concrete
 * implementing classes). Schemaless instances are map-based, while schemaful instances have a fixed
 * layout and array and are therefore more efficient.
 */
public abstract class SkylarkInfo extends StructImpl implements Concatable, SkylarkClassObject {

  // Private because this should not be subclassed outside this file.
  private SkylarkInfo(Provider provider, @Nullable Location loc) {
    super(provider, loc);
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
    // TODO(bazel-team): If we export at the end of a full module's evaluation, instead of at the
    // end of every top-level statement, then we can assume that exported implies frozen, and just
    // return true here without a traversal.
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
   * Returns the custom (i.e. per-instance, as opposed to per-provider-type) error message string
   * format used by this provider instance, or null if not set.
   */
  @Nullable
  public abstract String getCustomErrorMessageFormatForUnknownField();

  /** Returns the layout for this provider if it is schemaful, null otherwise. */
  @Nullable
  public abstract Layout getLayout();

  /** Returns true if this provider is schemaful (array-based), false otherwise. */
  public boolean isCompact() {
    return getLayout() != null;
  }

  @Override
  public Object getValue(Location loc, StarlarkSemantics starlarkSemantics, String name)
      throws EvalException {
    // By default, a SkylarkInfo's field values are not affected by the Starlark semantics.
    Object x = getValue(name);
    if (x != null) {
      return x;
    } else if (name.equals("to_json") || name.equals("to_proto")) {
      // to_json and to_proto should not be methods of struct or provider instances.
      // However, they are, for now, and it is important that they be consistently
      // returned by attribute lookup operations regardless of whether a field or method
      // is desired. TODO(adonovan): eliminate this hack.
      return new BuiltinCallable(this, name);
    } else {
      return null;
    }
  }

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
   * <p>This is used to create structs for special purposes, such as {@code ctx.attr} and the
   * {@code native} module. The creation location will be {@link Location#BUILTIN}.
   *
   * <p>{@code errorMessageFormatForUnknownField} is a string format, as for {@link
   * Provider#getErrorMessageFormatForUnknownField}.
   *
   * <p>It is preferred to not use this method. Instead, create a new subclass of {@link
   * NativeProvider} with the desired error message format, and create a corresponding {@link
   * NativeInfo} subclass.
   */
  // TODO(bazel-team): Make the special structs that need a custom error message use a different
  // provider (subclassing NativeProvider) and a different StructImpl implementation. Then remove
  // this functionality, thereby saving a string pointer field for the majority of providers that
  // don't need it.
  public static SkylarkInfo createSchemalessWithCustomMessage(
      Provider provider, Map<String, Object> values, String errorMessageFormatForUnknownField) {
    Preconditions.checkNotNull(errorMessageFormatForUnknownField);
    return new MapBackedSkylarkInfo(
        provider, values, Location.BUILTIN, errorMessageFormatForUnknownField);
  }

  /**
   * Creates a schemaful (array-based) provider instance with the given provider type, layout, and
   * values.
   *
   * <p>The order of the values must correspond to the given layout.
   *
   * <p>{@code loc} is the creation location for this instance. Built-in provider instances may use
   * {@link Location#BUILTIN}, which is the default if null.
   */
  public static SkylarkInfo createSchemaful(
      Provider provider,
      Layout layout,
      Object[] values,
      @Nullable Location loc) {
    return new CompactSkylarkInfo(provider, layout, values, loc);
  }

  /**
   * Returns the concrete implementation classes of this abstract class.
   *
   * <p>This is useful for code that depends on reflection.
   */
  public static List<Class<? extends SkylarkInfo>> getImplementationClasses() {
    return ImmutableList.of(MapBackedSkylarkInfo.class, CompactSkylarkInfo.class);
  }

  /**
   * A specification of what fields a provider instance has, and how they are ordered in an
   * array-backed implementation.
   *
   * <p>The provider instance may only have fields that appear in its layout. Not all fields in the
   * layout need be present on the instance.
   */
  @Immutable
  @AutoCodec
  public static final class Layout {

    /**
     * A map from field names to a contiguous range of integers [0, n), ordered by integer value.
     */
    private final ImmutableMap<String, Integer> map;

    /**
     * Constructs a {@link Layout} from the given field names.
     *
     * <p>The order of the field names is preserved in the layout.
     *
     * @throws IllegalArgumentException if any field names are given more than once
     */
    public Layout(Iterable<String> fields) {
      this(makeMap(fields));
    }

    @AutoCodec.VisibleForSerialization
    @AutoCodec.Instantiator
    Layout(ImmutableMap<String, Integer> map) {
      this.map = map;
    }

    private static ImmutableMap<String, Integer> makeMap(Iterable<String> fields) {
      ImmutableMap.Builder<String, Integer> layoutBuilder = ImmutableMap.builder();
      int i = 0;
      for (String field : fields) {
        layoutBuilder.put(field, i++);
      }
      return layoutBuilder.build();
    }

    @Override
    public boolean equals(Object other) {
      if (!(other instanceof Layout)) {
        return false;
      }
      if (map == other) {
        return true;
      }
      return map.equals(((Layout) other).map);
    }

    @Override
    public int hashCode() {
      return map.hashCode();
    }

    /** Returns the number of fields in the layout. */
    public int size() {
      return map.size();
    }

    /**
     * Returns the index position associated with the given field, or null if the field is not
     * mentioned by the layout.
     */
    public Integer getFieldIndex(String field) {
      return map.get(field);
    }

    /** Returns the field names specified by this layout, in order. */
    public ImmutableCollection<String> getFields() {
      return map.keySet();
    }

    /** Returns the entry set of the underlying map, in order. */
    public ImmutableCollection<Map.Entry<String, Integer>> entrySet() {
      return map.entrySet();
    }
  }

  /** A {@link SkylarkInfo} implementation that stores its values in a map. */
  // TODO(b/72448383): Make private.
  public static final class MapBackedSkylarkInfo extends SkylarkInfo {
    private final ImmutableSortedMap<String, Object> values;

    /**
     * Formattable string with one {@code '%s'} placeholder for the missing field name.
     *
     * <p>If null, uses the default format specified by the provider.
     */
    @Nullable
    private final String errorMessageFormatForUnknownField;

    MapBackedSkylarkInfo(
        Provider provider,
        Map<String, Object> values,
        @Nullable Location loc,
        @Nullable String errorMessageFormatForUnknownField) {
      super(provider, loc);
      // TODO(b/74396075): Phase out the unnecessary conversions done by this call to copyValues.
      this.values = copyValues(values);
      this.errorMessageFormatForUnknownField = errorMessageFormatForUnknownField;
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
    protected String getErrorMessageFormatForUnknownField() {
      return errorMessageFormatForUnknownField != null
          ? errorMessageFormatForUnknownField : super.getErrorMessageFormatForUnknownField();
    }

    @Override
    public String getCustomErrorMessageFormatForUnknownField() {
      return errorMessageFormatForUnknownField;
    }

    @Override
    public Layout getLayout() {
      return null;
    }
  }

  /** A {@link SkylarkInfo} implementation that stores its values in array to save space. */
  private static final class CompactSkylarkInfo extends SkylarkInfo implements Concatable {

    private final Layout layout;
    /** Treated as immutable. */
    private final Object[] values;

    CompactSkylarkInfo(
        Provider provider,
        Layout layout,
        Object[] values,
        @Nullable Location loc) {
      super(provider, loc);
      this.layout = Preconditions.checkNotNull(layout);
      Preconditions.checkArgument(
          layout.size() == values.length,
          "Layout has length %s, but number of given values was %s", layout.size(), values.length);
      this.values = new Object[values.length];
      for (int i = 0; i < values.length; i++) {
        // TODO(b/74396075): Phase out this unnecessary conversion.
        // NB: fromJava treats null as None, but we need nulls to indicate a field is not present.
        if (values[i] != null) {
          this.values[i] = Starlark.fromJava(values[i], null);
        }
      }
    }

    @Override
    public Object getValue(String name) {
      Integer index = layout.getFieldIndex(name);
      if (index == null) {
        return null;
      }
      return values[index];
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
    public String getCustomErrorMessageFormatForUnknownField() {
      return null;
    }

    @Override
    public Layout getLayout() {
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
        Layout layout = compactLeft.layout;
        if (layout.equals(compactRight.layout)) {
          int nvals = layout.size();
          Object[] newValues = new Object[nvals];
          for (int i = 0; i < nvals; i++) {
            newValues[i] =
                (compactLeft.values[i] != null) ? compactLeft.values[i] : compactRight.values[i];
          }
          return createSchemaful(provider, layout, newValues, loc);
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
