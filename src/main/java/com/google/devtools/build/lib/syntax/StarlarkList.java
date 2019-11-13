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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import javax.annotation.Nullable;

/** A StarlarkList is a mutable finite sequence of values. */
@SkylarkModule(
    name = "list",
    category = SkylarkModuleCategory.BUILTIN,
    doc =
        "The built-in list type. Example list expressions:<br>"
            + "<pre class=language-python>x = [1, 2, 3]</pre>"
            + "Accessing elements is possible using indexing (starts from <code>0</code>):<br>"
            + "<pre class=language-python>e = x[1]   # e == 2</pre>"
            + "Lists support the <code>+</code> operator to concatenate two lists. Example:<br>"
            + "<pre class=language-python>x = [1, 2] + [3, 4]   # x == [1, 2, 3, 4]\n"
            + "x = [\"a\", \"b\"]\n"
            + "x += [\"c\"]            # x == [\"a\", \"b\", \"c\"]</pre>"
            + "Similar to strings, lists support slice operations:"
            + "<pre class=language-python>['a', 'b', 'c', 'd'][1:3]   # ['b', 'c']\n"
            + "['a', 'b', 'c', 'd'][::2]  # ['a', 'c']\n"
            + "['a', 'b', 'c', 'd'][3:0:-1]  # ['d', 'c', 'b']</pre>"
            + "Lists are mutable, as in Python.")
public final class StarlarkList<E> extends Sequence<E> {

  private final ArrayList<E> contents;

  /** Final except for {@link #unsafeShallowFreeze}; must not be modified any other way. */
  private Mutability mutability;

  private StarlarkList(ArrayList<E> rawContents, @Nullable Mutability mutability) {
    this.contents = Preconditions.checkNotNull(rawContents);
    this.mutability = mutability == null ? Mutability.IMMUTABLE : mutability;
    }

  /**
   * Creates an instance, taking ownership of the supplied {@link ArrayList}. This is exposed for
   * performance reasons. May be used when the calling code will not modify the supplied list after
   * calling (honor system).
   */
  static <T> StarlarkList<T> wrapUnsafe(@Nullable StarlarkThread thread, ArrayList<T> rawContents) {
    return wrapUnsafe(thread == null ? null : thread.mutability(), rawContents);
  }

  /**
   * Create an instance, taking ownership of the supplied {@link ArrayList}. This is exposed for
   * performance reasons. May be used when the calling code will not modify the supplied list after
   * calling (honor system).
   */
  static <T> StarlarkList<T> wrapUnsafe(@Nullable Mutability mutability, ArrayList<T> rawContents) {
    return new StarlarkList<>(rawContents, mutability);
  }

  /**
   * A shared instance for the empty list with immutable mutability.
   *
   * <p>Other immutable empty list objects can exist, e.g. lists that were once mutable but whose
   * environments were then frozen. This instance is for empty lists that were always frozen from
   * the beginning.
   */
  private static final StarlarkList<?> EMPTY =
      StarlarkList.copyOf(Mutability.IMMUTABLE, ImmutableList.of());

  /** Returns an empty frozen list, cast to have an arbitrary content type. */
  @SuppressWarnings("unchecked")
  public static <T> StarlarkList<T> empty() {
    return (StarlarkList<T>) EMPTY;
    }

  /**
   * Returns a {@code StarlarkList} whose items are given by an iterable and which has the given
   * {@link Mutability}. If {@code mutability} is null, the list is immutable.
   */
  public static <T> StarlarkList<T> copyOf(
      @Nullable Mutability mutability, Iterable<? extends T> contents) {
    return new StarlarkList<>(Lists.newArrayList(contents), mutability);
    }

  /**
   * Returns a {@code StarlarkList} whose items are given by an iterable and which has the {@link
   * Mutability} belonging to the given {@link StarlarkThread}. If {@code thread} is null, the list
   * is immutable.
   */
  public static <T> StarlarkList<T> copyOf(
      @Nullable StarlarkThread thread, Iterable<? extends T> contents) {
    return StarlarkList.copyOf(thread == null ? null : thread.mutability(), contents);
  }

  /**
   * Returns a {@code StarlarkList} with the given items and the {@link Mutability} of the given
   * {@link StarlarkThread}. If {@code thread} is null, the list is immutable.
   */
  public static <T> StarlarkList<T> of(@Nullable StarlarkThread thread, T... contents) {
    // Safe since we're taking a copy of the input.
    return StarlarkList.wrapUnsafe(
        thread == null ? null : thread.mutability(), Lists.newArrayList(contents));
    }

    @Override
    public Mutability mutability() {
      return mutability;
    }

    @Override
    public void unsafeShallowFreeze() {
      Mutability.Freezable.checkUnsafeShallowFreezePrecondition(this);
      this.mutability = Mutability.IMMUTABLE;
    }

    @Override
    public ImmutableList<E> getImmutableList() {
      return ImmutableList.copyOf(contents);
    }

    @Override
    protected List<E> getContentsUnsafe() {
      return contents;
    }

  /**
   * Returns a new {@code StarlarkList} that is the concatenation of two {@code StarlarkList}s. The
   * new list will have the given {@link Mutability}.
   */
  public static <T> StarlarkList<T> concat(
      StarlarkList<? extends T> left, StarlarkList<? extends T> right, Mutability mutability) {

      ArrayList<T> newContents = new ArrayList<>(left.size() + right.size());
      addAll(newContents, left.contents);
      addAll(newContents, right.contents);
    return new StarlarkList<>(newContents, mutability);
    }

    /** More efficient {@link List#addAll} replacement when both lists are {@link ArrayList}s. */
    private static <T> void addAll(ArrayList<T> addTo, ArrayList<? extends T> addFrom) {
      // Hot code path, skip iterator.
      for (int i = 0; i < addFrom.size(); i++) {
        addTo.add(addFrom.get(i));
      }
    }

  @Override
  public StarlarkList<E> repeat(int times, Mutability mutability) {
      if (times <= 0) {
      return StarlarkList.wrapUnsafe(mutability, new ArrayList<>());
      }

      ArrayList<E> repeated = new ArrayList<>(this.size() * times);
      for (int i = 0; i < times; i++) {
        repeated.addAll(this);
      }
    return StarlarkList.wrapUnsafe(mutability, repeated);
    }

  @Override
  public StarlarkList<E> getSlice(
      Object start, Object end, Object step, Location loc, Mutability mutability)
      throws EvalException {
      List<Integer> sliceIndices = EvalUtils.getSliceIndices(start, end, step, this.size(), loc);
      ArrayList<E> list = new ArrayList<>(sliceIndices.size());
      // foreach is not used to avoid iterator overhead
      for (int i = 0; i < sliceIndices.size(); ++i) {
        list.add(this.get(sliceIndices.get(i)));
      }
    return StarlarkList.wrapUnsafe(mutability, list);
    }

    /**
     * Appends an element to the end of the list, after validating that mutation is allowed.
     *
     * @param element the element to add
     * @param loc the location to use for error reporting
     * @param mutability the {@link Mutability} associated with the operation
     */
    public void add(E element, Location loc, Mutability mutability) throws EvalException {
      checkMutable(loc, mutability);
      contents.add(element);
    }

    /**
     * Inserts an element at a given position to the list.
     *
     * @param index the new element's index
     * @param element the element to add
     * @param loc the location to use for error reporting
     * @param mutability the {@link Mutability} associated with the operation
     */
    public void add(int index, E element, Location loc, Mutability mutability)
        throws EvalException {
      checkMutable(loc, mutability);
      contents.add(index, element);
    }

    /**
     * Appends all the elements to the end of the list.
     *
     * @param elements the elements to add
     * @param loc the location to use for error reporting
     * @param mutability the {@link Mutability} associated with the operation
     */
    public void addAll(Iterable<? extends E> elements, Location loc, Mutability mutability)
        throws EvalException {
      checkMutable(loc, mutability);
      Iterables.addAll(contents, elements);
    }

    /**
     * Removes the element at a given index. The index must already have been validated to be in
     * range.
     *
     * @param index the index of the element to remove
     * @param loc the location to use for error reporting
     * @param mutability the {@link Mutability} associated with the operation
     */
    public void remove(int index, Location loc, Mutability mutability) throws EvalException {
      checkMutable(loc, mutability);
      contents.remove(index);
    }

    @SkylarkCallable(
        name = "remove",
        doc =
            "Removes the first item from the list whose value is x. "
                + "It is an error if there is no such item.",
        parameters = {@Param(name = "x", type = Object.class, doc = "The object to remove.")},
        useLocation = true,
        useStarlarkThread = true)
    public NoneType removeObject(Object x, Location loc, StarlarkThread thread)
        throws EvalException {
      for (int i = 0; i < size(); i++) {
        if (get(i).equals(x)) {
          remove(i, loc, thread.mutability());
          return Starlark.NONE;
        }
      }
      throw new EvalException(loc, Printer.format("item %r not found in list", x));
    }

    /**
     * Sets the position at the given index to contain the given value. The index must already have
     * been validated to be in range.
     *
     * @param index the position to change
     * @param value the new value
     * @param loc the location to use for error reporting
     * @param mutability the {@link Mutability} associated with the operation
     */
    public void set(int index, E value, Location loc, Mutability mutability) throws EvalException {
      checkMutable(loc, mutability);
      contents.set(index, value);
    }

    @SkylarkCallable(
        name = "append",
        doc = "Adds an item to the end of the list.",
        parameters = {
          @Param(
              name = "item",
              type = Object.class,
              doc = "Item to add at the end.",
              noneable = true)
        },
        useLocation = true,
        useStarlarkThread = true)
    @SuppressWarnings("unchecked") // Cast of Object item to E
    public NoneType append(Object item, Location loc, StarlarkThread thread) throws EvalException {
      add((E) item, loc, thread.mutability());
      return Starlark.NONE;
    }

    @SkylarkCallable(
        name = "clear",
        doc = "Removes all the elements of the list.",
        useLocation = true,
        useStarlarkThread = true)
    public NoneType clearMethod(Location loc, StarlarkThread thread) throws EvalException {
      checkMutable(loc, thread.mutability());
      contents.clear();
      return Starlark.NONE;
    }

    @SkylarkCallable(
        name = "insert",
        doc = "Inserts an item at a given position.",
        parameters = {
          @Param(name = "index", type = Integer.class, doc = "The index of the given position."),
          @Param(name = "item", type = Object.class, doc = "The item.", noneable = true)
        },
        useLocation = true,
        useStarlarkThread = true)
    @SuppressWarnings("unchecked") // Cast of Object item to E
    public NoneType insert(Integer index, Object item, Location loc, StarlarkThread thread)
        throws EvalException {
      add(EvalUtils.clampRangeEndpoint(index, size()), (E) item, loc, thread.mutability());
      return Starlark.NONE;
    }

    @SkylarkCallable(
        name = "extend",
        doc = "Adds all items to the end of the list.",
        parameters = {
          @Param(name = "items", type = Object.class, doc = "Items to add at the end.")
        },
        useLocation = true,
        useStarlarkThread = true)
    @SuppressWarnings("unchecked")
    public NoneType extend(Object items, Location loc, StarlarkThread thread) throws EvalException {
      addAll(
          (Collection<? extends E>) EvalUtils.toCollection(items, loc, thread),
          loc,
          thread.mutability());
      return Starlark.NONE;
    }

    @SkylarkCallable(
        name = "index",
        doc =
            "Returns the index in the list of the first item whose value is x. "
                + "It is an error if there is no such item.",
        parameters = {
          @Param(name = "x", type = Object.class, doc = "The object to search."),
          @Param(
              name = "start",
              type = Integer.class,
              defaultValue = "None",
              noneable = true,
              named = true,
              doc = "The start index of the list portion to inspect."),
          @Param(
              name = "end",
              type = Integer.class,
              defaultValue = "None",
              noneable = true,
              named = true,
              doc = "The end index of the list portion to inspect.")
        },
        useLocation = true)
    public Integer index(Object x, Object start, Object end, Location loc) throws EvalException {
      int i =
          start == Starlark.NONE ? 0 : EvalUtils.clampRangeEndpoint((Integer) start, this.size());
      int j =
          end == Starlark.NONE
              ? this.size()
              : EvalUtils.clampRangeEndpoint((Integer) end, this.size());

      while (i < j) {
        if (this.get(i).equals(x)) {
          return i;
        }
        i++;
      }
      throw new EvalException(loc, Printer.format("item %r not found in list", x));
    }

    @SkylarkCallable(
        name = "pop",
        doc =
            "Removes the item at the given position in the list, and returns it. "
                + "If no <code>index</code> is specified, "
                + "it removes and returns the last item in the list.",
        parameters = {
          @Param(
              name = "i",
              type = Integer.class,
              noneable = true,
              defaultValue = "None",
              doc = "The index of the item.")
        },
        useLocation = true,
        useStarlarkThread = true)
    public Object pop(Object i, Location loc, StarlarkThread thread) throws EvalException {
      int arg = i == Starlark.NONE ? -1 : (Integer) i;
      int index = EvalUtils.getSequenceIndex(arg, size(), loc);
      Object result = get(index);
      remove(index, loc, thread.mutability());
      return result;
    }
  }
