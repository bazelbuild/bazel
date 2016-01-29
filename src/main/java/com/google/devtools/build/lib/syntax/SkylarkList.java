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
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.syntax.SkylarkMutable.MutableCollection;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.ListIterator;
import java.util.RandomAccess;

import javax.annotation.Nullable;

/**
 * A class to handle lists and tuples in Skylark.
 */
@SkylarkModule(name = "sequence", documented = false,
    doc = "common type of lists and tuples")
  public abstract class SkylarkList<E>
    extends MutableCollection<E> implements List<E>, RandomAccess {

  /**
   * Returns an ImmutableList object with the current underlying contents of this SkylarkList.
   */
  public abstract ImmutableList<E> getImmutableList();

  /**
   * Returns a List object with the current underlying contents of this SkylarkList.
   * This object must not be mutated, but need not be an {@link ImmutableList}.
   * Indeed it can sometimes be a {@link GlobList}.
   */
  // TODO(bazel-team): move GlobList out of Skylark, into an extension.
  public abstract List<E> getContents();

  /**
   * The underlying contents are a (usually) mutable data structure.
   * Read access is forwarded to these contents.
   * This object must not be modified outside an {@link Environment}
   * with a correct matching {@link Mutability},
   * which should be checked beforehand using {@link #checkMutable}.
   * it need not be an instance of {@link com.google.common.collect.ImmutableList}.
   */
  @Override
  protected abstract List<E> getContentsUnsafe();

  /**
   * Returns true if this list is a tuple.
   */
  public abstract boolean isTuple();

  // A SkylarkList forwards all read-only access to the getContentsUnsafe().
  @Override
  public final E get(int i) {
    return getContentsUnsafe().get(i);
  }

  @Override
  public int indexOf(Object element) {
    return getContentsUnsafe().indexOf(element);
  }

  @Override
  public int lastIndexOf(Object element) {
    return getContentsUnsafe().lastIndexOf(element);
  }

  @Override
  public ListIterator<E> listIterator() {
    return getContentsUnsafe().listIterator();
  }

  @Override
  public ListIterator<E> listIterator(int index) {
    return getContentsUnsafe().listIterator(index);
  }

  // For subList, use the immutable getContents() rather than getContentsUnsafe,
  // to prevent subsequent mutation. To get a mutable SkylarkList,
  // use a method that takes an Environment into account.
  @Override
  public List<E> subList(int fromIndex, int toIndex) {
    return getContents().subList(fromIndex, toIndex);
  }

  // A SkylarkList disables all direct mutation methods.
  @Override
  public void add(int index, E element) {
    throw new UnsupportedOperationException();
  }

  @Override
  public boolean addAll(int index, Collection<? extends E> elements) {
    throw new UnsupportedOperationException();
  }

  @Override
  public E remove(int index) {
    throw new UnsupportedOperationException();
  }

  @Override
  public E set(int index, E element) {
    throw new UnsupportedOperationException();
  }

  // Other methods
  @Override
  public void write(Appendable buffer, char quotationMark) {
    Printer.printList(buffer, getContentsUnsafe(), isTuple(), quotationMark);
  }

  // Note that the following two functions slightly violate the Java List protocol,
  // in that it does NOT consider that a SkylarkList .equals() an arbitrary List with same contents.
  // This is because we use .equals() to model skylark equality, which like Python
  // distinguishes a MutableList from a Tuple.
  @Override
  public boolean equals(Object object) {
    return (this == object)
        || ((this.getClass() == object.getClass())
            && getContentsUnsafe().equals(((SkylarkList) object).getContentsUnsafe()));
  }

  @Override
  public int hashCode() {
    return getClass().hashCode() + 31 * getContentsUnsafe().hashCode();
  }

  /**
   * Cast a {@code List<?>} to a {@code List<T>} after checking its current contents.
   * @param list the List to cast
   * @param type the expected class of elements
   * @param description a description of the argument being converted, or null, for debugging
   */
  @SuppressWarnings("unchecked")
  public static <TYPE> List<TYPE> castList(
      List<?> list, Class<TYPE> type, @Nullable String description)
      throws EvalException {
    Object desc = description == null ? null : Printer.formattable("'%s' element", description);
    for (Object value : list) {
      SkylarkType.checkType(value, type, desc);
    }
    return (List<TYPE>) list;
  }

  /**
   * Cast a SkylarkList to a {@code List<T>} after checking its current contents.
   * Treat None as meaning the empty List.
   * @param obj the Object to cast. null and None are treated as an empty list.
   * @param type the expected class of elements
   * @param description a description of the argument being converted, or null, for debugging
   */
  public static <TYPE> List<TYPE> castSkylarkListOrNoneToList(
      Object obj, Class<TYPE> type, @Nullable String description)
      throws EvalException {
    if (EvalUtils.isNullOrNone(obj)) {
      return ImmutableList.of();
    }
    if (obj instanceof SkylarkList) {
      return ((SkylarkList<?>) obj).getContents(type, description);
    }
    throw new EvalException(null,
        Printer.format("Illegal argument: %s is not of expected type list or NoneType",
            description == null ? Printer.repr(obj) : String.format("'%s'", description)));
  }

  /**
   * Cast the SkylarkList object into a List of the given type.
   * @param type the expected class of elements
   * @param description a description of the argument being converted, or null, for debugging
   */
  public <TYPE> List<TYPE> getContents(Class<TYPE> type, @Nullable String description)
      throws EvalException {
    return castList(getContentsUnsafe(), type, description);
  }

  /**
   * A class for mutable lists.
   */
  @SkylarkModule(
    name = "list",
    doc =
        "A language built-in type to support lists. Example of list literal:<br>"
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
            + "Lists are mutable, as in Python."
  )
  public static final class MutableList<E> extends SkylarkList<E> {

    private final ArrayList<E> contents = new ArrayList<>();

    // Treat GlobList specially: external code depends on it.
    // TODO(bazel-team): make data structures *and binary operators* extensible
    // (via e.g. interface classes for each binary operator) so that GlobList
    // can be implemented outside of the core of Skylark.
    @Nullable private GlobList<E> globList;

    private final Mutability mutability;

    /**
     * Creates a MutableList from contents and a Mutability.
     * @param contents the contents of the list
     * @param mutability a Mutability context
     * @return a MutableList containing the elements
     */
    @SuppressWarnings("unchecked")
    MutableList(Iterable<? extends E> contents, Mutability mutability) {
      super();
      addAllUnsafe(contents);
      if (contents instanceof GlobList) {
        globList = (GlobList<E>) contents;
      }
      this.mutability = mutability;
    }

    /**
     * Creates a MutableList from contents and an Environment.
     * @param contents the contents of the list
     * @param env an Environment from which to inherit Mutability, or null for immutable
     * @return a MutableList containing the elements
     */
    public MutableList(Iterable<? extends E> contents, @Nullable Environment env) {
      this(contents, env == null ? Mutability.IMMUTABLE : env.mutability());
    }

    /**
     * Creates a MutableList from contents.
     * @param contents the contents of the list
     * @return an actually immutable MutableList containing the elements
     */
    public MutableList(Iterable<? extends E> contents) {
      this(contents, Mutability.IMMUTABLE);
    }

    /**
     * Creates a mutable or immutable MutableList depending on the given {@link Mutability}.
     */
    public MutableList(Mutability mutability) {
      this(Collections.EMPTY_LIST, mutability);
    }

    /**
     * Builds a Skylark list (actually immutable) from a variable number of arguments.
     * @param env an Environment from which to inherit Mutability, or null for immutable
     * @param contents the contents of the list
     * @return a Skylark list containing the specified arguments as elements.
     */
    public static <E> MutableList<E> of(@Nullable Environment env, E... contents) {
      return new MutableList(ImmutableList.copyOf(contents), env);
    }

    /**
     * Adds all the elements at the end of the MutableList.
     * @param elements the elements to add
     * Assumes that you already checked for Mutability.
     */
    private void addAllUnsafe(Iterable<? extends E> elements) {
      for (E elem : elements) {
        contents.add(elem);
      }
    }

    @Override
    protected void checkMutable(Location loc, Environment env) throws EvalException {
      super.checkMutable(loc, env);
      globList = null; // If you're going to mutate it, invalidate the underlying GlobList.
    }

    @Nullable public GlobList<E> getGlobList() {
      return globList;
    }

    /**
     * @return the GlobList if there is one, otherwise an Immutable copy of the regular contents.
     */
    @Override
    @SuppressWarnings("unchecked")
    public List<E> getContents() {
      if (globList != null) {
        return globList;
      }
      return getImmutableList();
    }

    @Override
    protected List<E> getContentsUnsafe() {
      return contents;
    }

    /**
     * @return the GlobList if there is one, otherwise the regular contents.
     */
    private List<?> getGlobListOrContentsUnsafe() {
      if (globList != null) {
        return globList;
      }
      return contents;
    }

    /**
     * Concatenate two MutableList
     * @param left the start of the new list
     * @param right the end of the new list
     * @param env the Environment in which to create a new list
     * @return a new MutableList
     */
    public static <E> MutableList<E> concat(
        MutableList<? extends E> left,
        MutableList<? extends E> right,
        Environment env) {
      if (left.getGlobList() == null && right.getGlobList() == null) {
        return new MutableList(Iterables.concat(left, right), env);
      }
      return new MutableList(GlobList.concat(
          left.getGlobListOrContentsUnsafe(), right.getGlobListOrContentsUnsafe()), env);
    }

    /**
     * Adds one element at the end of the MutableList.
     * @param element the element to add
     * @param loc the Location at which to report any error
     * @param env the Environment requesting the modification
     */
    public void add(E element, Location loc, Environment env) throws EvalException {
      checkMutable(loc, env);
      contents.add(element);
    }

    public void remove(int index, Location loc, Environment env) throws EvalException {
      checkMutable(loc, env);
      contents.remove(index);
    }

    /**
     * Adds all the elements at the end of the MutableList.
     * @param elements the elements to add
     * @param loc the Location at which to report any error
     * @param env the Environment requesting the modification
     */
    public void addAll(Iterable<? extends E> elements, Location loc, Environment env)
        throws EvalException {
      checkMutable(loc, env);
      addAllUnsafe(elements);
    }

    @Override
    public ImmutableList<E> getImmutableList() {
      return ImmutableList.copyOf(contents);
    }

    @Override
    public Mutability mutability() {
      return mutability;
    }

    @Override
    public boolean isTuple() {
      return false;
    }

    @Override
    public boolean isImmutable() {
      return false;
    }

    /**
     * An empty IMMUTABLE MutableList.
     */
    public static final MutableList EMPTY = new MutableList(Tuple.EMPTY);
  }

  /**
   * An immutable tuple, e.g. in (1, 2, 3)
   */
  @SkylarkModule(
    name = "tuple",
    doc =
        "A language built-in type to support tuples. Example of tuple literal:<br>"
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
            + "Tuples are immutable, therefore <code>x[1] = \"a\"</code> is not supported."
  )
  @Immutable
  public static final class Tuple<E> extends SkylarkList<E> {

    private final ImmutableList<E> contents;

    private Tuple(ImmutableList<E> contents) {
      super();
      this.contents = contents;
    }

    @Override
    public Mutability mutability() {
      return Mutability.IMMUTABLE;
    }

    /**
     * THE empty Skylark tuple.
     */
    private static final Tuple<?> EMPTY = new Tuple<>(ImmutableList.of());

    @SuppressWarnings("unchecked")
    public static final <E> Tuple<E> empty() {
      return (Tuple<E>) EMPTY;
    }

    /**
     * Creates a Tuple from an ImmutableList.
     */
    public static<E> Tuple<E> create(ImmutableList<E> contents) {
      if (contents.isEmpty()) {
        return empty();
      }
      return new Tuple(contents);
    }

    /**
     * Creates a Tuple from an Iterable.
     */
    public static <E> Tuple<E> copyOf(Iterable<? extends E> contents) {
      return create(ImmutableList.<E>copyOf(contents));
    }

    /**
     * Builds a Skylark tuple from a variable number of arguments.
     * @param elements a variable number of arguments (or an Array of Object-s)
     * @return a Skylark tuple containing the specified arguments as elements.
     */
    public static <E> Tuple<E> of(E... elements) {
      return Tuple.create(ImmutableList.copyOf(elements));
    }

    @Override
    public ImmutableList<E> getImmutableList() {
      return contents;
    }

    @Override
    public List<E> getContents() {
      return contents;
    }

    @Override
    protected List<E> getContentsUnsafe() {
      return contents;
    }

    @Override
    public boolean isTuple() {
      return true;
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
  }
}
