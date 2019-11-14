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
import java.util.List;

/**
 * A Skylark tuple, i.e. the value represented by {@code (1, 2, 3)}. Tuples are always immutable
 * (regardless of the {@link StarlarkThread} they are created in).
 */
@SkylarkModule(
    name = "tuple",
    category = SkylarkModuleCategory.BUILTIN,
    doc =
        "The built-in tuple type. Example tuple expressions:<br>"
            + "<pre class=language-python>x = (1, 2, 3)</pre>"
            + "Accessing elements is possible using indexing (starts from <code>0</code>):<br>"
            + "<pre class=language-python>e = x[1]   # e == 2</pre>"
            + "Lists support the <code>+</code> operator to concatenate two tuples. Example:<br>"
            + "<pre class=language-python>x = (1, 2) + (3, 4)   # x == (1, 2, 3, 4)\n"
            + "x = (\"a\", \"b\")\n"
            + "x += (\"c\",)            # x == (\"a\", \"b\", \"c\")</pre>"
            + "Similar to lists, tuples support slice operations:"
            + "<pre class=language-python>('a', 'b', 'c', 'd')[1:3]   # ('b', 'c')\n"
            + "('a', 'b', 'c', 'd')[::2]  # ('a', 'c')\n"
            + "('a', 'b', 'c', 'd')[3:0:-1]  # ('d', 'c', 'b')</pre>"
            + "Tuples are immutable, therefore <code>x[1] = \"a\"</code> is not supported.")
public final class Tuple<E> extends Sequence<E> {

  private final ImmutableList<E> contents;

  private Tuple(ImmutableList<E> contents) {
    this.contents = contents;
  }

  /**
   * A shared instance for the empty tuple.
   *
   * <p>This instance should be the only empty tuple.
   */
  private static final Tuple<?> EMPTY = new Tuple<>(ImmutableList.of());

  /** Returns the empty tuple, cast to have an arbitrary content type. */
  @SuppressWarnings("unchecked")
  public static <T> Tuple<T> empty() {
    return (Tuple<T>) EMPTY;
  }

  /**
   * Creates a {@code Tuple} from an {@link ImmutableList}, reusing the empty instance if
   * applicable.
   */
  private static <T> Tuple<T> create(ImmutableList<T> contents) {
    if (contents.isEmpty()) {
      return empty();
    }
    return new Tuple<>(contents);
  }

  /** Returns a {@code Tuple} whose items are given by an iterable. */
  public static <T> Tuple<T> copyOf(Iterable<? extends T> contents) {
    return create(ImmutableList.<T>copyOf(contents));
  }

  /**
   * Returns a {@code Tuple} whose items are given by an immutable list.
   *
   * <p>This method is a specialization of a {@link #copyOf(Iterable)} that avoids an unnecessary
   * {@code copyOf} invocation.
   */
  public static <T> Tuple<T> copyOf(ImmutableList<T> contents) {
    return create(contents);
  }

  /** Returns a {@code Tuple} with the given items. */
  public static <T> Tuple<T> of(T... elements) {
    return Tuple.create(ImmutableList.copyOf(elements));
  }

  @Override
  public boolean isImmutable() {
    for (Object item : this) {
      if (!EvalUtils.isImmutable(item)) {
        return false;
      }
    }
    return true;
  }

  @Override
  public boolean isHashable() {
    for (Object item : this) {
      if (!EvalUtils.isHashable(item)) {
        return false;
      }
    }
    return true;
  }

  @Override
  public ImmutableList<E> getImmutableList() {
    return contents;
  }

  @Override
  protected List<E> getContentsUnsafe() {
    return contents;
  }

  /** Returns a {@code Tuple} that is the concatenation of two {@code Tuple}s. */
  public static <T> Tuple<T> concat(Tuple<? extends T> left, Tuple<? extends T> right) {
    // Build the ImmutableList directly rather than use Iterables.concat, to avoid unnecessary
    // array resizing.
    return create(
        ImmutableList.<T>builderWithExpectedSize(left.size() + right.size())
            .addAll(left)
            .addAll(right)
            .build());
  }

  @Override
  public Tuple<E> getSlice(
      Object start, Object end, Object step, Location loc, Mutability mutability)
      throws EvalException {
    List<Integer> sliceIndices = EvalUtils.getSliceIndices(start, end, step, this.size(), loc);
    ImmutableList.Builder<E> builder = ImmutableList.builderWithExpectedSize(sliceIndices.size());
    // foreach is not used to avoid iterator overhead
    for (int i = 0; i < sliceIndices.size(); ++i) {
      builder.add(this.get(sliceIndices.get(i)));
    }
    return copyOf(builder.build());
  }

  @Override
  public Tuple<E> repeat(int times, Mutability mutability) {
    if (times <= 0) {
      return empty();
    }

    ImmutableList.Builder<E> builder = ImmutableList.builderWithExpectedSize(this.size() * times);
    for (int i = 0; i < times; i++) {
      builder.addAll(this);
    }
    return copyOf(builder.build());
  }
}
