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
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Interner;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.skyframe.serialization.VisibleForSerialization;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;
import net.starlark.java.eval.Compactable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.syntax.TokenKind;

/**
 * A struct-like Info (provider instance) for providers defined in Starlark that have a schema.
 *
 * <p>Maintainer's note: This class is memory-optimized in a way that can cause profiling
 * instability in some pathological cases. See {@link StarlarkProvider#optimizeField} for more
 * information.
 */
public final class StarlarkInfoWithSchema extends StarlarkInfo {

  /**
   * Interner for instances that have no {@linkplain Starlark#truth truthy} values.
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
  private static final Interner<StarlarkInfoWithSchema> nonTruthyInterner =
      BlazeInterners.newWeakInterner();

  private final StarlarkProvider provider;

  // For each field in provider.getFields the table contains on corresponding position either null,
  // a legal Starlark value, or an optimized value (see StarlarkProvider#optimizeField).
  private final Object[] table;

  // `table` elements should already be optimized by caller, see StarlarkProvider#optimizeField
  @VisibleForSerialization // private
  StarlarkInfoWithSchema(StarlarkProvider provider, Object[] table) {
    this.provider = provider;
    this.table = table;
  }

  @Override
  public Provider getProvider() {
    return provider;
  }

  @VisibleForSerialization // private
  Object[] getTable() {
    return table;
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
  static class StarlarkInfoFactory extends StarlarkProvider.StarlarkInfoFactory {
    private final ImmutableList<String> fields;
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
        valueTable[pos] = provider.optimizeField(pos, value);
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
      return new StarlarkInfoWithSchema(provider, valueTable);
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
  public ImmutableCollection<String> getFieldNames() {
    ImmutableList.Builder<String> fieldNames = new ImmutableList.Builder<>();
    ImmutableList<String> fields = provider.getFields();
    for (int i = 0; i < fields.size(); i++) {
      if (table[i] != null) {
        fieldNames.add(fields.get(i));
      }
    }
    return fieldNames.build();
  }

  @Override
  public boolean isImmutable() {
    // If the provider is not yet exported, the hash code of the object is subject to change.
    if (!provider.isExported()) {
      return false;
    }
    for (int i = 0; i < table.length; i++) {
      if (table[i] != null
          && !(provider.isOptimised(i, table[i]) // optimised fields might not be Starlark values
              || Starlark.isImmutable(table[i]))) {
        return false;
      }
    }
    return true;
  }

  @Nullable
  @Override
  public Object getValue(String name) {
    ImmutableList<String> fields = provider.getFields();
    int i = indexOfField(name, fields);
    return i >= 0 ? provider.retrieveOptimizedField(i, table[i]) : null;
  }

  @Nullable
  @Override
  public StarlarkInfoWithSchema binaryOp(TokenKind op, Object that, boolean thisLeft)
      throws EvalException {
    if (op == TokenKind.PLUS && that instanceof StarlarkInfo) {
      final Provider thatProvider = ((StarlarkInfo) that).getProvider();
      if (!provider.equals(thatProvider)) {
        throw Starlark.errorf(
            "Cannot use '+' operator on instances of different providers (%s and %s)",
            provider.getPrintableName(), thatProvider.getPrintableName());
      }
      Preconditions.checkArgument(that instanceof StarlarkInfoWithSchema);
      return thisLeft
          ? plus(this, (StarlarkInfoWithSchema) that) //
          : plus((StarlarkInfoWithSchema) that, this);
    }
    return null;
  }

  private static StarlarkInfoWithSchema plus(StarlarkInfoWithSchema x, StarlarkInfoWithSchema y)
      throws EvalException {
    int n = x.table.length;

    Object[] ztable = new Object[n];
    for (int i = 0; i < n; i++) {
      if (x.table[i] != null && y.table[i] != null) {
        ImmutableList<String> schema = x.provider.getFields();
        throw Starlark.errorf("cannot add struct instances with common field '%s'", schema.get(i));
      }
      ztable[i] = x.table[i] != null ? x.table[i] : y.table[i];
    }
    return new StarlarkInfoWithSchema(x.provider, ztable);
  }

  @Override
  public StarlarkInfoWithSchema unsafeOptimizeMemoryLayout() {
    boolean sawTruthyValue = false;
    int n = table.length;
    for (int i = 0; i < n; i++) {
      Object val = table[i];
      sawTruthyValue = sawTruthyValue || truth(val);
      if (val instanceof Compactable compactable) {
        table[i] = compactable.unsafeOptimizeMemoryLayout();
      }
    }
    return sawTruthyValue ? this : nonTruthyInterner.intern(this);
  }

  @Override
  public boolean equals(Object otherObject) {
    if (this == otherObject) {
      return true;
    }
    if (!(otherObject instanceof StarlarkInfoWithSchema other)) {
      return false;
    }
    if (!this.provider.equals(other.provider)) {
      return false;
    }
    return Arrays.equals(this.table, other.table);
  }

  @Override
  public int hashCode() {
    return 31 * provider.hashCode() + Arrays.hashCode(table);
  }

  /** Returns the index of the given named field in the given list of fields, or -1 if not found. */
  private static int indexOfField(String name, ImmutableList<String> fields) {
    if (fields.size() <= BINARY_SEARCH_THRESHOLD) {
      return fields.indexOf(name);
    }
    int idx = Collections.binarySearch(fields, name);
    return idx >= 0 ? idx : -1;
  }

  /**
   * Augmented version of {@link Starlark#truth} that handles {@link NestedSet} and {@code null}.
   */
  private static boolean truth(Object val) {
    return switch (val) {
      case NestedSet<?> nestedSet -> !nestedSet.isEmpty();
      case null -> false;
      default -> Starlark.truth(val);
    };
  }
}
