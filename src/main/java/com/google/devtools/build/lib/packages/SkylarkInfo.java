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
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.syntax.ClassObject;
import com.google.devtools.build.lib.syntax.Concatable;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.Starlark;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/** An Info (provider instance) for providers defined in Starlark. */
public final class SkylarkInfo extends StructImpl implements Concatable, ClassObject {

  // For a n-element info, the table contains n key strings, sorted,
  // followed by the n corresponding legal Starlark values.
  private final Object[] table;

  // A format string with one %s placeholder for the missing field name.
  // If null, uses the default format specified by the provider.
  // TODO(adonovan): make the provider determine the error message
  // (but: this has implications for struct+struct, the equivalence
  // relation, and other observable behaviors).
  @Nullable private final String unknownFieldError;

  private SkylarkInfo(
      Provider provider,
      Object[] table,
      @Nullable Location loc,
      @Nullable String unknownFieldError) {
    super(provider, loc);
    this.table = table;
    this.unknownFieldError = unknownFieldError;
  }

  // Converts a map to a table of sorted keys followed by corresponding values.
  private static Object[] toTable(Map<String, Object> values) {
    int n = values.size();
    Object[] table = new Object[n + n];
    int i = 0;
    for (String key : values.keySet()) {
      table[i++] = key;
    }
    Arrays.sort(table, 0, n);
    for (i = 0; i < n; i++) {
      Object x = values.get(table[i]);
      table[n + i] = Starlark.checkValid(x);
    }
    return table;
  }

  /**
   * Constructs a SkylarkInfo from a list of (possibly null but otherwise legal) Starlark values
   * corresponding to a list of field names, which must be sorted. This exists solely for the
   * SkylarkProvider constructor form {@code p(a=..., b=...)}.
   */
  static SkylarkInfo fromSortedFieldList(
      Provider provider,
      ImmutableList<String> sortedFieldNames,
      Object[] values,
      @Nullable Location loc) {
    // Count non-null values.
    int n = 0;
    for (Object x : values) {
      if (x != null) {
        n++;
      }
    }

    // Copy the non-null values and their keys.
    Object[] table = new Object[n + n];
    for (int i = 0, j = 0; i < values.length; i++) {
      Object x = values[i];
      if (x != null) {
        table[j] = sortedFieldNames.get(i);
        table[n + j] = Starlark.checkValid(x); // redundant defensive check
        if (++j == n) {
          break; // remaining values are all null
        }
      }
    }

    return new SkylarkInfo(provider, table, loc, /*unknownFieldError=*/ null);
  }

  @Override
  public ImmutableCollection<String> getFieldNames() {
    // TODO(adonovan): opt: can we avoid allocating three objects?
    @SuppressWarnings("unchecked")
    List<String> keys = (List<String>) (List<?>) Arrays.asList(table).subList(0, table.length / 2);
    return ImmutableList.copyOf(keys);
  }

  /**
   * Returns the custom (i.e. per-instance, as opposed to per-provider-type) error message string
   * format used by this provider instance, or null if not set.
   */
  @Nullable
  @Override
  protected String getErrorMessageFormatForUnknownField() {
    return unknownFieldError != null
        ? unknownFieldError
        : super.getErrorMessageFormatForUnknownField();
  }

  @Override
  public boolean isImmutable() {
    // If the provider is not yet exported, the hash code of the object is subject to change.
    // TODO(adonovan): implement isHashable?
    if (!getProvider().isExported()) {
      return false;
    }
    // TODO(bazel-team): If we export at the end of a full module's evaluation, instead of at the
    // end of every top-level statement, then we can assume that exported implies frozen, and just
    // return true here without a traversal.
    for (int i = table.length / 2; i < table.length; i++) {
      if (!EvalUtils.isImmutable(table[i])) {
        return false;
      }
    }
    return true;
  }

  @Override
  public Object getValue(String name) {
    int n = table.length / 2;
    int i = Arrays.binarySearch(table, 0, n, name);
    if (i < 0) {
      return null;
    }
    return table[n + i];
  }

  /**
   * Creates a schemaless provider instance with the given provider type and field values.
   *
   * <p>{@code loc} is the creation location for this instance. Built-in provider instances may use
   * {@link Location#BUILTIN}, which is the default if null.
   */
  public static SkylarkInfo create(
      Provider provider, Map<String, Object> values, @Nullable Location loc) {
    return new SkylarkInfo(provider, toTable(values), loc, /*unknownFieldError=*/ null);
  }

  /**
   * Creates a schemaless provider instance with the given provider type, field values, and
   * unknown-field error message.
   *
   * <p>This is used to create structs for special purposes, such as {@code ctx.attr} and the {@code
   * native} module. The creation location will be {@link Location#BUILTIN}.
   *
   * <p>{@code unknownFieldError} is a string format, as for {@link
   * Provider#getErrorMessageFormatForUnknownField}.
   *
   * @deprecated Do not use this method. Instead, create a new subclass of {@link NativeProvider}
   *     with the desired error message format, and create a corresponding {@link NativeInfo}
   *     subclass.
   */
  // TODO(bazel-team): Make the special structs that need a custom error message use a different
  // provider (subclassing NativeProvider) and a different StructImpl implementation. Then remove
  // this functionality, thereby saving a string pointer field for the majority of providers that
  // don't need it.
  @Deprecated
  public static SkylarkInfo createWithCustomMessage(
      Provider provider, Map<String, Object> values, String unknownFieldError) {
    Preconditions.checkNotNull(unknownFieldError);
    return new SkylarkInfo(provider, toTable(values), Location.BUILTIN, unknownFieldError);
  }

  @Override
  public Concatter getConcatter() {
    return SkylarkInfo::concat;
  }

  private static Concatable concat(Concatable left, Concatable right, Location loc)
      throws EvalException {
    // Casts are safe because this Concatter is only used by SkylarkInfo.
    SkylarkInfo x = (SkylarkInfo) left;
    SkylarkInfo y = (SkylarkInfo) right;
    Provider xprov = x.getProvider();
    Provider yprov = y.getProvider();
    if (!xprov.equals(yprov)) {
      throw new EvalException(
          loc,
          String.format(
              "Cannot use '+' operator on instances of different providers (%s and %s)",
              xprov.getPrintableName(), yprov.getPrintableName()));
    }

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
        throw new EvalException(
            loc, String.format("cannot add struct instances with common field '%s'", xk));
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

    return new SkylarkInfo(xprov, ztable, loc, x.unknownFieldError);
  }
}
