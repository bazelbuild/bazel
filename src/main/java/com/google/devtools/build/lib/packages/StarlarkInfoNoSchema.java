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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;
import net.starlark.java.eval.Compactable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.syntax.StarlarkType;
import net.starlark.java.syntax.TokenKind;
import net.starlark.java.syntax.Types;

/**
 * A struct-like Info (provider instance) for providers defined in Starlark that don't have a
 * schema.
 */
public class StarlarkInfoNoSchema extends StarlarkInfo {
  // TODO(bazel-team): require this to be either a StarlarkProvider or StructProvider.
  private final Provider provider;

  // For a n-element info, the table contains n key strings, sorted,
  // followed by the n corresponding legal Starlark values.
  private final Object[] table;

  private StarlarkInfoNoSchema(Provider provider, Object[] table) {
    this.provider = provider;
    this.table = table;
  }

  StarlarkInfoNoSchema(Provider provider, Map<String, Object> values) {
    this.provider = provider;
    this.table = toTable(values);
  }

  @Override
  public Provider getProvider() {
    return provider;
  }

  private StarlarkType getStructType(StarlarkSemantics semantics) {
    int n = table.length / 2;
    ImmutableMap.Builder<String, StarlarkType> fieldTypes = ImmutableMap.builderWithExpectedSize(n);
    for (int i = 0; i < n; i++) {
      String name = (String) table[i];
      StarlarkType type = Starlark.getStarlarkType(table[n + i], semantics);
      fieldTypes.put(name, type);
    }
    return Types.struct(fieldTypes.buildOrThrow());
  }

  @Override
  public StarlarkType getStarlarkType(StarlarkSemantics semantics) {
    if (provider instanceof StarlarkType type) {
      // This is the case for StarlarkProvider. Do nominal typing.
      // TODO: #27370 - Should we emit the fields and types for schemaless starlark providers?
      return type;
    } else {
      // Untyped struct; provider is either StructProvider (`struct` in BUILD API) or one of a few
      // non-StructProvider builtin providers that deliberately construct untyped struct values
      // (e.g. Actions or ApplePlatform). Do structural typing.
      // TODO: #27370 - Should we synthesize a nominal StarlarkType for the value type of such
      // non-StructProvider builtin providers?
      return getStructType(semantics);
    }
  }

  /**
   * Creates a schemaless provider instance with the given provider type and field values.
   *
   * @param provider A {@code Provider} without a schema. {@code StarlarkProvider} with a schema is
   *     not supported by this call.
   * @param values the field values
   */
  static StarlarkInfo createSchemaless(Provider provider, Map<String, Object> values) {
    Preconditions.checkArgument(
        !(provider instanceof StarlarkProvider)
            || ((StarlarkProvider) provider).getFields() == null);
    return new StarlarkInfoNoSchema(provider, values);
  }

  // Converts a map to a table of sorted keys followed by corresponding values.
  private static Object[] toTable(Map<String, Object> values) {
    int n = values.size();
    Object[] table = new Object[n + n];
    int i = 0;
    // TODO(b/380824219): Once fastcall and thus createFromNamedArgs is removed, consider whether
    // we can wrap values.entrySet() in a SortedSet and avoid and remove sortPairs().
    // Maybe an overloaded constructor StarlarkInfoNoSchema(Provider, SortedMap<>, Location)
    // could also be useful in this context. Connection with b/380824219: StarlarkInfoFactory
    // assembles values into a TreeMap and calls StarlarkInfoNoSchema(Provider, Map<>, Location).
    for (Map.Entry<String, Object> e : values.entrySet()) {
      table[i] = e.getKey();
      table[n + i] = Starlark.checkValid(e.getValue());
      i++;
    }
    // Sort keys, permuting values in parallel.
    if (n > 1) {
      sortPairs(table, 0, n - 1);
    }
    return table;
  }

  static StarlarkProvider.StarlarkInfoFactory newStarlarkInfoFactory(
      StarlarkProvider provider, StarlarkThread thread) {
    return new StarlarkInfoFactory(provider, thread);
  }

  /**
   * Constructs a StarlarkInfo with calls forwarded from one of the StarlarkInfo ArgumentProcessor
   * implementations. Checks that each key is provided at most once. This class exists solely for
   * the StarlarkInfo ArgumentProcessors.
   */
  static class StarlarkInfoFactory extends StarlarkProvider.StarlarkInfoFactory {
    private final Map<String, Object> namedArgMap;

    StarlarkInfoFactory(StarlarkProvider provider, StarlarkThread thread) {
      super(provider, thread);
      this.namedArgMap = new HashMap<>();
    }

    @Override
    public void addNamedArg(String name, Object value) throws EvalException {
      // TODO(b/380824219): Evaluate whether we can know the number of named args here, and then
      // place the args into the table directly.
      Object oldValue = namedArgMap.put(name, value);
      if (oldValue != null) {
        throw Starlark.errorf(
            "got multiple values for parameter %s in call to instantiate provider %s",
            name, provider.getPrintableName());
      }
    }

    @Override
    public StarlarkInfo createFromArgs() {
      return new StarlarkInfoNoSchema(provider, namedArgMap);
    }

    @Override
    public StarlarkInfo createFromMap(Map<String, Object> map) {
      return new StarlarkInfoNoSchema(provider, map);
    }
  }

  // Sorts non-empty slice a[lo:hi] (inclusive) in place.
  // Elements a[n:2n) are permuted the same way as a[0:n),
  // where n = a.length / 2. The lower half must be strings.
  // Precondition: 0 <= lo <= hi < n.
  static void sortPairs(Object[] a, int lo, int hi) {
    String pivot = (String) a[lo + (hi - lo) / 2];

    int i = lo;
    int j = hi;
    while (i <= j) {
      while (((String) a[i]).compareTo(pivot) < 0) {
        i++;
      }
      while (((String) a[j]).compareTo(pivot) > 0) {
        j--;
      }
      if (i <= j) {
        int n = a.length >> 1;
        swap(a, i, j);
        swap(a, i + n, j + n);
        i++;
        j--;
      }
    }
    if (lo < j) {
      sortPairs(a, lo, j);
    }
    if (i < hi) {
      sortPairs(a, i, hi);
    }
  }

  private static void swap(Object[] a, int i, int j) {
    Object tmp = a[i];
    a[i] = a[j];
    a[j] = tmp;
  }

  @Override
  public ImmutableCollection<String> getFieldNames() {
    // TODO(adonovan): opt: can we avoid allocating three objects?
    @SuppressWarnings("unchecked")
    List<String> keys = (List<String>) (List<?>) Arrays.asList(table).subList(0, table.length / 2);
    return ImmutableList.copyOf(keys);
  }

  @Override
  public boolean isImmutable() {
    // If the provider is not yet exported, the hash code of the object is subject to change.
    if (!provider.isExported()) {
      return false;
    }
    for (int i = table.length / 2; i < table.length; i++) {
      if (!Starlark.isImmutable(table[i])) {
        return false;
      }
    }
    return true;
  }

  @Override
  public void checkHashable() throws EvalException {
    super.checkHashable(); // Verifies that the values are immutable.
    // Bazel has historically allowed structs of immutable values to be considered hashable even if
    // those values are not Starlark-hashable by themselves (e.g. frozen lists). This is
    // inconsistent and arguably wrong, but fixing it would be a breaking change.
    // Thus, instead of checking whether the values are Starlark-hashable, below we only check
    // whether they have a usable hashCode() implementation.
    for (int i = table.length / 2; i < table.length; i++) {
      Object val = table[i];
      if (!Starlark.isAcyclic(val)) {
        // A self-referential value's hashCode() can cause a stack overflow. Trigger it early; the
        // StackOverflowError will be caught by Starlark.checkHashable() and rethrown as an
        // EvalException.
        var unused = val.hashCode();
      }
    }
  }

  @Nullable
  @Override
  public Object getValue(String name) {
    int n = table.length / 2;
    int i;
    if (n <= BINARY_SEARCH_THRESHOLD) {
      i = -1;
      for (int j = 0; j < n; j++) {
        if (table[j].equals(name)) {
          i = j;
          break;
        }
      }
    } else {
      i = Arrays.binarySearch(table, 0, n, name);
    }
    if (i < 0) {
      return null;
    }
    return table[n + i];
  }

  @Nullable
  @Override
  public StarlarkInfo binaryOp(TokenKind op, Object that, boolean thisLeft) throws EvalException {
    if (op == TokenKind.PLUS && that instanceof StarlarkInfo) {
      final Provider thatProvider = ((StarlarkInfo) that).getProvider();
      if (!provider.equals(thatProvider)) {
        throw Starlark.errorf(
            "Cannot use '+' operator on instances of different providers (%s and %s)",
            provider.getPrintableName(), thatProvider.getPrintableName());
      }
      Preconditions.checkArgument(that instanceof StarlarkInfoNoSchema);
      return thisLeft
          ? plus(this, (StarlarkInfoNoSchema) that) //
          : plus((StarlarkInfoNoSchema) that, this);
    }
    return null;
  }

  private static StarlarkInfo plus(StarlarkInfoNoSchema x, StarlarkInfoNoSchema y)
      throws EvalException {
    // ztable = merge(x.table, y.table)
    int xsize = x.table.length / 2;
    int ysize = y.table.length / 2;
    int zsize = xsize + ysize;
    Object[] ztable = new Object[zsize + zsize];
    int xi = 0;
    int yi = 0;
    int zi = 0;
    while (xi < xsize && yi < ysize) {
      String xk = (String) x.table[xi];
      String yk = (String) y.table[yi];
      int cmp = xk.compareTo(yk);
      if (cmp < 0) {
        ztable[zi] = xk;
        ztable[zi + zsize] = x.table[xi + xsize];
        xi++;
      } else if (cmp > 0) {
        ztable[zi] = yk;
        ztable[zi + zsize] = y.table[yi + ysize];
        yi++;
      } else {
        throw Starlark.errorf("cannot add struct instances with common field '%s'", xk);
      }
      zi++;
    }
    while (xi < xsize) {
      ztable[zi] = x.table[xi];
      ztable[zi + zsize] = x.table[xi + xsize];
      xi++;
      zi++;
    }
    while (yi < ysize) {
      ztable[zi] = y.table[yi];
      ztable[zi + zsize] = y.table[yi + ysize];
      yi++;
      zi++;
    }

    return new StarlarkInfoNoSchema(x.provider, ztable);
  }

  @Override
  public final boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (!(o instanceof StarlarkInfoNoSchema other)) {
      return false;
    }
    return provider.equals(other.provider) && Arrays.equals(table, other.table);
  }

  @Override
  public final int hashCode() {
    return 31 * provider.hashCode() + Arrays.hashCode(table);
  }

  @Override
  public StarlarkInfoNoSchema unsafeOptimizeMemoryLayout() {
    for (int i = table.length / 2; i < table.length; i++) {
      if (table[i] instanceof Compactable compactable) {
        table[i] = compactable.unsafeOptimizeMemoryLayout();
      }
    }
    return this;
  }
}
