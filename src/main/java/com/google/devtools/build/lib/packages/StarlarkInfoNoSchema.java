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
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.syntax.Location;
import net.starlark.java.syntax.TokenKind;

/**
 * A struct-like Info (provider instance) for providers defined in Starlark that don't have a
 * schema.
 */
public class StarlarkInfoNoSchema extends StarlarkInfo {
  private final Provider provider;

  // For a n-element info, the table contains n key strings, sorted,
  // followed by the n corresponding legal Starlark values.
  private final Object[] table;

  // TODO(adonovan): restrict type of provider to StarlarkProvider?
  // Do we ever need StarlarkInfos of BuiltinProviders? Such BuiltinProviders could
  // be  moved to Starlark using bzl builtins injection.
  // Alternatively: what about this implementation is specific to StarlarkProvider?
  // It's really just a "generic" or "dynamic" representation of a struct,
  // analogous to reflection versus generated message classes in the protobuf world.
  // The efficient table algorithms would be a nice addition to the Starlark
  // interpreter, to allow other clients to define their own fast structs
  // (or to define a standard one). See also comments at Info about upcoming clean-ups.
  private StarlarkInfoNoSchema(Provider provider, Object[] table, @Nullable Location loc) {
    super(loc);
    this.provider = provider;
    this.table = table;
  }

  StarlarkInfoNoSchema(Provider provider, Map<String, Object> values, @Nullable Location loc) {
    super(loc);
    this.provider = provider;
    this.table = toTable(values);
  }

  @Override
  public Provider getProvider() {
    return provider;
  }

  /**
   * Creates a schemaless provider instance with the given provider type and field values.
   *
   * @param provider A {@code Provider} without a schema. {@code StarlarkProvider} with a schema is
   *     not supported by this call.
   * @param values the field values
   * @param loc the creation location for this instance. Built-in provider instances may use {@link
   *     Location#BUILTIN}, which is the default if null.
   */
  static StarlarkInfo createSchemaless(
      Provider provider, Map<String, Object> values, @Nullable Location loc) {
    Preconditions.checkArgument(
        !(provider instanceof StarlarkProvider)
            || ((StarlarkProvider) provider).getFields() == null);
    return new StarlarkInfoNoSchema(provider, values, loc);
  }

  // Converts a map to a table of sorted keys followed by corresponding values.
  private static Object[] toTable(Map<String, Object> values) {
    int n = values.size();
    Object[] table = new Object[n + n];
    int i = 0;
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

  /**
   * Constructs a StarlarkInfo from an array of alternating key/value pairs as provided by
   * Starlark.fastcall. Checks that each key is provided at most once. This optimized
   * zero-allocation function exists solely for the StarlarkProvider constructor.
   */
  static StarlarkInfo createFromNamedArgs(StarlarkProvider provider, Object[] table, Location loc)
      throws EvalException {
    // Permute fastcall form (k, v, ..., k, v) into table form (k, k, ..., v, v).
    permute(table);

    int n = table.length >> 1; // number of K/V pairs

    // Sort keys, permuting values in parallel.
    if (n > 1) {
      sortPairs(table, 0, n - 1);
    }

    // Check for duplicate keys, which are now adjacent.
    for (int i = 0; i < n - 1; i++) {
      if (table[i].equals(table[i + 1])) {
        throw Starlark.errorf(
            "got multiple values for parameter %s in call to instantiate provider %s",
            table[i], provider.getPrintableName());
      }
    }

    return new StarlarkInfoNoSchema(provider, table, loc);
  }

  // Permutes array elements from alternating keys/values form,
  // (as used by fastcall's named array) into keys-then-corresponding-values form,
  // as used by StarlarkInfo.table.
  // The permutation preserves the key/value association but not the order of keys.
  static void permute(Object[] named) {
    int n = named.length >> 1; // number of K/V pairs

    // Thanks to Murali Ganapathy for the algorithm.
    // See https://play.golang.org/p/QOKnrj_bIwk.
    //
    // i and j are the indices bracketing successive pairs of cells,
    // working from the outside to the middle.
    //
    //   i                  j
    //   [KV]KVKVKVKVKVKV[KV]
    //     i              j
    //   KK[KV]KVKVKVKV[KV]VV
    //       i          j
    //   KKKK[KV]KVKV[KV]VVVV
    //   etc...
    for (int i = 0; i < n - 1; i += 2) {
      int j = named.length - i;
      // rotate two pairs [KV]...[kv] -> [Kk]...[vV]
      Object tmp = named[i + 1];
      named[i + 1] = named[j - 2];
      named[j - 2] = named[j - 1];
      named[j - 1] = tmp;
    }
    // reverse lower half containing keys: [KkvV] -> [kKvV]
    for (int i = 0; i < n >> 1; i++) {
      Object tmp = named[n - 1 - i];
      named[n - 1 - i] = named[i];
      named[i] = tmp;
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

  @Nullable
  @Override
  public Object getValue(String name) {
    int n = table.length / 2;
    int i = Arrays.binarySearch(table, 0, n, name);
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

    return new StarlarkInfoNoSchema(x.provider, ztable, Location.BUILTIN);
  }

  @Override
  public StarlarkInfoNoSchema unsafeOptimizeMemoryLayout() {
    for (int i = table.length / 2; i < table.length; i++) {
      if (table[i] instanceof StarlarkList<?>) {
        // On duplicated lists, ImmutableStarlarkLists objects are duplicated, but not underlying
        // Object arrays
        table[i] = ((StarlarkList<?>) table[i]).unsafeOptimizeMemoryLayout();
      } else if (table[i] instanceof StarlarkInfo) {
        table[i] = ((StarlarkInfo) table[i]).unsafeOptimizeMemoryLayout();
      }
    }
    return this;
  }
}
