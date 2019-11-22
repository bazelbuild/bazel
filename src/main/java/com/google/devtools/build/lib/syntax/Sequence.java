// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.syntax;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.skylarkinterface.SkylarkValue;
import java.util.Collections;
import java.util.List;
import java.util.RandomAccess;
import javax.annotation.Nullable;

/**
 * A Sequence is a finite sequence of Starlark values, such as a list or tuple.
 *
 * <p>Although this implements the {@link List} interface, it is not mutable via that interface's
 * methods. Instead, use the mutators that take in a {@link Mutability} object.
 */
@SkylarkModule(
    name = "sequence",
    documented = false,
    category = SkylarkModuleCategory.BUILTIN,
    doc = "common type of lists and tuples.")
public interface Sequence<E> extends SkylarkValue, List<E>, RandomAccess, SkylarkIndexable {

  @Override
  default boolean truth() {
    return !isEmpty();
  }

  /** Returns an ImmutableList object with the current underlying contents of this Sequence. */
  default ImmutableList<E> getImmutableList() {
    return ImmutableList.copyOf(this);
  }

  /**
   * Retrieve an entry from a Sequence.
   *
   * @param key the index
   * @param loc a {@link Location} in case of error
   * @throws EvalException if the key is invalid
   */
  @Override
  default E getIndex(Object key, Location loc) throws EvalException {
    return get(EvalUtils.getSequenceIndex(key, size(), loc));
  }

  @Override
  default boolean containsKey(Object key, Location loc) throws EvalException {
    return contains(key);
  }

  /**
   * Constructs a version of this {@code Sequence} containing just the items in a slice.
   *
   * <p>{@code mutability} will be used for the resulting list. If it is null, the list will be
   * immutable. For {@code Tuple}s, which are always immutable, this argument is ignored.
   *
   * @see EvalUtils#getSliceIndices
   * @throws EvalException if the key is invalid; uses {@code loc} for error reporting
   */
  Sequence<E> getSlice(Object start, Object end, Object step, Location loc, Mutability mutability)
      throws EvalException;

  /**
   * Casts a {@code List<?>} to an unmodifiable {@code List<T>}, after checking that its contents
   * all have type {@code type}.
   *
   * <p>The returned list may or may not be a view that is affected by updates to the original list.
   *
   * @param list the original list to cast
   * @param type the expected type of all the list's elements
   * @param description a description of the argument being converted, or null, for debugging
   */
  // We could have used bounded generics to ensure that only downcasts are possible (i.e. cast
  // List<S> to List<T extends S>), but this would be less convenient for some callers, and it would
  // disallow casting an empty list to any type.
  // TODO(adonovan): this method doesn't belong here.
  @SuppressWarnings("unchecked")
  public static <T> List<T> castList(List<?> list, Class<T> type, @Nullable String description)
      throws EvalException {
    Object desc = description == null ? null : Printer.formattable("'%s' element", description);
    for (Object value : list) {
      SkylarkType.checkType(value, type, desc);
    }
    return Collections.unmodifiableList((List<T>) list);
  }

  /**
   * If {@code obj} is a {@code Sequence}, casts it to an unmodifiable {@code List<T>} after
   * checking that each element has type {@code type}. If {@code obj} is {@code None} or null,
   * treats it as an empty list. For all other values, throws an {@link EvalException}.
   *
   * <p>The returned list may or may not be a view that is affected by updates to the original list.
   *
   * @param obj the object to cast. null and None are treated as an empty list.
   * @param type the expected type of all the list's elements
   * @param description a description of the argument being converted, or null, for debugging
   */
  // TODO(adonovan): this method doesn't belong here.
  public static <T> List<T> castSkylarkListOrNoneToList(
      Object obj, Class<T> type, @Nullable String description) throws EvalException {
    if (EvalUtils.isNullOrNone(obj)) {
      return ImmutableList.of();
    }
    if (obj instanceof Sequence) {
      return ((Sequence<?>) obj).getContents(type, description);
    }
    throw new EvalException(
        null,
        String.format(
            "Illegal argument: %s is not of expected type list or NoneType",
            description == null ? Starlark.repr(obj) : String.format("'%s'", description)));
  }

  /**
   * Casts this list as an unmodifiable {@code List<T>}, after checking that each element has type
   * {@code type}.
   *
   * @param type the expected type of all the list's elements
   * @param description a description of the argument being converted, or null, for debugging
   */
  // TODO(adonovan): this method doesn't belong here.
  default <T> List<T> getContents(Class<T> type, @Nullable String description)
      throws EvalException {
    return castList(this, type, description);
  }
}
