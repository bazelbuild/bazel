// Copyright 2023 The Bazel Authors. All rights reserved.
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
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Interner;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.skyframe.serialization.VisibleForSerialization;
import com.google.devtools.build.lib.util.HashCodes;
import com.google.errorprone.annotations.ForOverride;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import javax.annotation.Nullable;
import net.starlark.java.eval.Compactable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkFloat;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.syntax.TokenKind;

/**
 * A struct-like Info (provider instance) for providers defined in Starlark that have a schema.
 *
 * <p>Maintainer's note: This class is memory-optimized in a way that can cause profiling
 * instability in some pathological cases. See {@link StarlarkProvider#maybeUnwrapDepset} for more
 * information.
 *
 * <p>Schemas with <= 5 fields (covering the majority of provider types in practice) each have their
 * own dedicated subclass to optimize for memory by forgoing an array.
 */
public abstract sealed class StarlarkInfoWithSchema extends StarlarkInfo {

  /**
   * Interner for {@linkplain #isInternable internable} instances.
   *
   * <p>Interning is limited to instances without truthy values for two reasons:
   *
   * <ol>
   *   <li>This covers the most frequent category of duplicates in practice. Interning further may
   *       not be worth the cost.
   *   <li>Hashing truthy values can be arbitrarily expensive and potentially even dangerous due to
   *       the possibility of object graph cycles.
   * </ol>
   */
  private static final Interner<StarlarkInfoWithSchema> interner = BlazeInterners.newWeakInterner();

  private final StarlarkProvider provider;

  private StarlarkInfoWithSchema(StarlarkProvider provider) {
    this.provider = provider;
  }

  @Override
  public final Provider getProvider() {
    return provider;
  }

  @ForOverride
  abstract Object getValueAt(int i);

  @ForOverride
  abstract void setValueAt(int i, Object val);

  @VisibleForSerialization
  Object[] getValuesForSerialization() {
    int n = provider.getFields().size();
    Object[] table = new Object[n];
    for (int i = 0; i < n; i++) {
      table[i] = getValueAt(i);
    }
    return table;
  }

  @VisibleForSerialization
  static StarlarkInfoWithSchema create(StarlarkProvider provider, Object[] vs) {
    return switch (vs.length) {
      case 0 -> new Schema0(provider);
      case 1 -> new Schema1(provider, vs[0]);
      case 2 -> new Schema2(provider, vs[0], vs[1]);
      case 3 -> new Schema3(provider, vs[0], vs[1], vs[2]);
      case 4 -> new Schema4(provider, vs[0], vs[1], vs[2], vs[3]);
      case 5 -> new Schema5(provider, vs[0], vs[1], vs[2], vs[3], vs[4]);
      default -> new SchemaN(provider, vs);
    };
  }

  static StarlarkProvider.StarlarkInfoFactory newStarlarkInfoFactory(
      StarlarkProvider provider, StarlarkThread thread) {
    return new StarlarkInfoFactory(provider, thread);
  }

  /**
   * Constructs a StarlarkInfo with calls forwarded from one of the StarlarkInfo ArgumentProcessor
   * implementations. Checks that each key is provided at most once, and is defined by the schema,
   * which must be sorted. This class exists solely for the StarlarkInfo ArgumentProcessors.
   */
  static final class StarlarkInfoFactory extends StarlarkProvider.StarlarkInfoFactory {
    private final ImmutableMap<String, Integer> fields;
    private final Object[] valueTable;
    private List<String> unexpected;

    StarlarkInfoFactory(StarlarkProvider provider, StarlarkThread thread) {
      super(provider, thread);
      this.fields = provider.getFields();
      this.valueTable = new Object[fields.size()];
      this.unexpected = null;
    }

    @Override
    public void addNamedArg(String name, Object value) throws EvalException {
      int pos = indexOfField(name, fields);
      if (pos >= 0) {
        if (valueTable[pos] != null) {
          throw Starlark.errorf(
              "got multiple values for parameter %s in call to instantiate provider %s",
              name, provider.getPrintableName());
        }
        valueTable[pos] =
            value instanceof Depset depset ? provider.maybeUnwrapDepset(pos, depset) : value;
      } else {
        if (unexpected == null) {
          unexpected = new ArrayList<>();
        }
        unexpected.add(name);
      }
    }

    @Override
    public StarlarkInfoWithSchema createFromArgs() throws EvalException {
      if (unexpected != null) {
        throw Starlark.errorf(
            "got unexpected field%s '%s' in call to instantiate provider %s",
            unexpected.size() > 1 ? "s" : "",
            Joiner.on("', '").join(unexpected),
            provider.getPrintableName());
      }
      return create(provider, valueTable);
    }

    @Override
    public StarlarkInfo createFromMap(Map<String, Object> map) throws EvalException {
      for (Map.Entry<String, Object> e : map.entrySet()) {
        addNamedArg(e.getKey(), e.getValue());
      }
      return createFromArgs();
    }
  }

  @Override
  public final ImmutableList<String> getFieldNames() {
    ImmutableList.Builder<String> fieldNames = new ImmutableList.Builder<>();
    int i = 0;
    for (String field : provider.getFields().keySet()) {
      if (getValueAt(i) != null) {
        fieldNames.add(field);
      }
      i++;
    }
    return fieldNames.build();
  }

  @Override
  public final boolean isImmutable() {
    // If the provider is not yet exported, the hash code of the object is subject to change.
    if (!provider.isExported()) {
      return false;
    }
    int n = provider.getFields().size();
    for (int i = 0; i < n; i++) {
      Object val = getValueAt(i);
      // Unwrapped depsets (NestedSets) are not Starlark values, but are immutable.
      if (val != null && !(val instanceof NestedSet<?> || Starlark.isImmutable(val))) {
        return false;
      }
    }
    return true;
  }

  @Nullable
  @Override
  public final Object getValue(String name) {
    ImmutableMap<String, Integer> fields = provider.getFields();
    int i = indexOfField(name, fields);
    if (i < 0) {
      return null;
    }
    Object val = getValueAt(i);
    return val instanceof NestedSet<?> nestedSet ? provider.rewrapDepset(i, nestedSet) : val;
  }

  @Nullable
  @Override
  public final StarlarkInfoWithSchema binaryOp(TokenKind op, Object that, boolean thisLeft)
      throws EvalException {
    if (op == TokenKind.PLUS && that instanceof StarlarkInfo thatInfo) {
      Provider thatProvider = thatInfo.getProvider();
      if (!provider.equals(thatProvider)) {
        throw Starlark.errorf(
            "Cannot use '+' operator on instances of different providers (%s and %s)",
            provider.getPrintableName(), thatProvider.getPrintableName());
      }
      Preconditions.checkArgument(thatInfo instanceof StarlarkInfoWithSchema, thatInfo);
      return thisLeft
          ? plus(this, (StarlarkInfoWithSchema) that) //
          : plus((StarlarkInfoWithSchema) that, this);
    }
    return null;
  }

  private static StarlarkInfoWithSchema plus(StarlarkInfoWithSchema x, StarlarkInfoWithSchema y)
      throws EvalException {
    int n = x.provider.getFields().size();

    Object[] ztable = new Object[n];
    for (int i = 0; i < n; i++) {
      Object xVal = x.getValueAt(i);
      Object yVal = y.getValueAt(i);
      if (xVal != null && yVal != null) {
        ImmutableMap<String, Integer> schema = x.provider.getFields();
        throw Starlark.errorf(
            "cannot add struct instances with common field '%s'", schema.keySet().asList().get(i));
      }
      ztable[i] = xVal != null ? xVal : yVal;
    }
    return create(x.provider, ztable);
  }

  @Override
  public final StarlarkInfoWithSchema unsafeOptimizeMemoryLayout() {
    boolean internable = true;
    int n = provider.getFields().size();
    for (int i = 0; i < n; i++) {
      Object val = getValueAt(i);
      internable = internable && valueIsInternable(val);
      if (val instanceof Compactable compactable) {
        setValueAt(i, compactable.unsafeOptimizeMemoryLayout());
      }
    }
    return internable ? interner.intern(this) : this;
  }

  /**
   * Returns true if this instance is internable (i.e. it contains only values that are considered
   * internable).
   *
   * <p>Internable instances are guaranteed to contain no object reference cycles, so they can be
   * interned by value equality.
   */
  final boolean isInternable() {
    int n = provider.getFields().size();
    for (int i = 0; i < n; i++) {
      if (!valueIsInternable(getValueAt(i))) {
        return false;
      }
    }
    return true;
  }

  /**
   * A value is considered internable if its {@linkplain Starlark#truth truthiness} is false (except
   * for {@link StarlarkFloat} values, which are never internable), or it is an empty {@link
   * NestedSet}, or a {@code null}.
   */
  private static boolean valueIsInternable(@Nullable Object val) {
    return switch (val) {
      case NestedSet<?> nestedSet -> nestedSet.isEmpty();
      // ``+0.0`, `-0.0`, and `0` are all `equals()` but have different memory representations.
      case StarlarkFloat sf -> false;
      case null -> true;
      default -> !Starlark.truth(val);
    };
  }

  /** Returns the index of the given named field in the given map of fields, or -1 if not found. */
  private static int indexOfField(String name, ImmutableMap<String, Integer> fields) {
    Integer idx = fields.get(name);
    return idx != null ? idx : -1;
  }

  /** For providers with no fields. */
  private static final class Schema0 extends StarlarkInfoWithSchema {
    Schema0(StarlarkProvider provider) {
      super(provider);
    }

    @Override
    Object getValueAt(int i) {
      throw new IndexOutOfBoundsException(i);
    }

    @Override
    void setValueAt(int i, Object val) {
      throw new IndexOutOfBoundsException(i);
    }

    @Override
    public int hashCode() {
      return 31 * getProvider().hashCode() + 1;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (!(o instanceof Schema0 other)) {
        return false;
      }
      return getProvider().equals(other.getProvider());
    }
  }

  /** For providers with 1 field. */
  private static final class Schema1 extends StarlarkInfoWithSchema {
    private Object v0;

    Schema1(StarlarkProvider provider, Object v0) {
      super(provider);
      this.v0 = v0;
    }

    @Override
    Object getValueAt(int i) {
      if (i == 0) {
        return v0;
      }
      throw new IndexOutOfBoundsException(i);
    }

    @Override
    void setValueAt(int i, Object val) {
      if (i == 0) {
        this.v0 = val;
        return;
      }
      throw new IndexOutOfBoundsException(i);
    }

    @Override
    public int hashCode() {
      return HashCodes.hashObjects(getProvider(), v0);
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (!(o instanceof Schema1 other)) {
        return false;
      }
      return getProvider().equals(other.getProvider()) && Objects.equals(v0, other.v0);
    }
  }

  /** For providers with 2 fields. */
  private static final class Schema2 extends StarlarkInfoWithSchema {
    private Object v0;
    private Object v1;

    Schema2(StarlarkProvider provider, Object v0, Object v1) {
      super(provider);
      this.v0 = v0;
      this.v1 = v1;
    }

    @Override
    Object getValueAt(int i) {
      return switch (i) {
        case 0 -> v0;
        case 1 -> v1;
        default -> throw new IndexOutOfBoundsException(i);
      };
    }

    @Override
    void setValueAt(int i, Object val) {
      switch (i) {
        case 0 -> this.v0 = val;
        case 1 -> this.v1 = val;
        default -> throw new IndexOutOfBoundsException(i);
      }
    }

    @Override
    public int hashCode() {
      return HashCodes.hashObjects(getProvider(), v0, v1);
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (!(o instanceof Schema2 other)) {
        return false;
      }
      return getProvider().equals(other.getProvider())
          && Objects.equals(v0, other.v0)
          && Objects.equals(v1, other.v1);
    }
  }

  /** For providers with 3 fields. */
  private static final class Schema3 extends StarlarkInfoWithSchema {
    private Object v0;
    private Object v1;
    private Object v2;

    Schema3(StarlarkProvider provider, Object v0, Object v1, Object v2) {
      super(provider);
      this.v0 = v0;
      this.v1 = v1;
      this.v2 = v2;
    }

    @Override
    Object getValueAt(int i) {
      return switch (i) {
        case 0 -> v0;
        case 1 -> v1;
        case 2 -> v2;
        default -> throw new IndexOutOfBoundsException(i);
      };
    }

    @Override
    void setValueAt(int i, Object val) {
      switch (i) {
        case 0 -> this.v0 = val;
        case 1 -> this.v1 = val;
        case 2 -> this.v2 = val;
        default -> throw new IndexOutOfBoundsException(i);
      }
    }

    @Override
    public int hashCode() {
      return HashCodes.hashObjects(getProvider(), v0, v1, v2);
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (!(o instanceof Schema3 other)) {
        return false;
      }
      return getProvider().equals(other.getProvider())
          && Objects.equals(v0, other.v0)
          && Objects.equals(v1, other.v1)
          && Objects.equals(v2, other.v2);
    }
  }

  /** For providers with 4 fields. */
  private static final class Schema4 extends StarlarkInfoWithSchema {
    private Object v0;
    private Object v1;
    private Object v2;
    private Object v3;

    Schema4(StarlarkProvider provider, Object v0, Object v1, Object v2, Object v3) {
      super(provider);
      this.v0 = v0;
      this.v1 = v1;
      this.v2 = v2;
      this.v3 = v3;
    }

    @Override
    Object getValueAt(int i) {
      return switch (i) {
        case 0 -> v0;
        case 1 -> v1;
        case 2 -> v2;
        case 3 -> v3;
        default -> throw new IndexOutOfBoundsException(i);
      };
    }

    @Override
    void setValueAt(int i, Object val) {
      switch (i) {
        case 0 -> this.v0 = val;
        case 1 -> this.v1 = val;
        case 2 -> this.v2 = val;
        case 3 -> this.v3 = val;
        default -> throw new IndexOutOfBoundsException(i);
      }
    }

    @Override
    public int hashCode() {
      return HashCodes.hashObjects(getProvider(), v0, v1, v2, v3);
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (!(o instanceof Schema4 other)) {
        return false;
      }
      return getProvider().equals(other.getProvider())
          && Objects.equals(v0, other.v0)
          && Objects.equals(v1, other.v1)
          && Objects.equals(v2, other.v2)
          && Objects.equals(v3, other.v3);
    }
  }

  /** For providers with 5 fields. */
  private static final class Schema5 extends StarlarkInfoWithSchema {
    private Object v0;
    private Object v1;
    private Object v2;
    private Object v3;
    private Object v4;

    Schema5(StarlarkProvider provider, Object v0, Object v1, Object v2, Object v3, Object v4) {
      super(provider);
      this.v0 = v0;
      this.v1 = v1;
      this.v2 = v2;
      this.v3 = v3;
      this.v4 = v4;
    }

    @Override
    Object getValueAt(int i) {
      return switch (i) {
        case 0 -> v0;
        case 1 -> v1;
        case 2 -> v2;
        case 3 -> v3;
        case 4 -> v4;
        default -> throw new IndexOutOfBoundsException(i);
      };
    }

    @Override
    void setValueAt(int i, Object val) {
      switch (i) {
        case 0 -> this.v0 = val;
        case 1 -> this.v1 = val;
        case 2 -> this.v2 = val;
        case 3 -> this.v3 = val;
        case 4 -> this.v4 = val;
        default -> throw new IndexOutOfBoundsException(i);
      }
    }

    @Override
    public int hashCode() {
      return HashCodes.hashObjects(getProvider(), v0, v1, v2, v3, v4);
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (!(o instanceof Schema5 other)) {
        return false;
      }
      return getProvider().equals(other.getProvider())
          && Objects.equals(v0, other.v0)
          && Objects.equals(v1, other.v1)
          && Objects.equals(v2, other.v2)
          && Objects.equals(v3, other.v3)
          && Objects.equals(v4, other.v4);
    }
  }

  /** For providers with 6 or more fields. */
  private static final class SchemaN extends StarlarkInfoWithSchema {
    private final Object[] vs;

    SchemaN(StarlarkProvider provider, Object[] vs) {
      super(provider);
      this.vs = vs;
    }

    @Override
    Object getValueAt(int i) {
      return vs[i];
    }

    @Override
    void setValueAt(int i, Object val) {
      vs[i] = val;
    }

    @Override
    Object[] getValuesForSerialization() {
      return vs;
    }

    @Override
    public int hashCode() {
      return 31 * getProvider().hashCode() + Arrays.hashCode(vs);
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (!(o instanceof SchemaN other)) {
        return false;
      }
      return getProvider().equals(other.getProvider()) && Arrays.equals(vs, other.vs);
    }
  }
}
