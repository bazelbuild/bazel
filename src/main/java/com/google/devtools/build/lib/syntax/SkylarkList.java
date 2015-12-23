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
import com.google.devtools.build.lib.skylarkinterface.SkylarkValue;
import com.google.devtools.build.lib.syntax.Mutability.Freezable;
import com.google.devtools.build.lib.syntax.Mutability.MutabilityException;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

import javax.annotation.Nullable;

/**
 * A class to handle lists and tuples in Skylark.
 */
@SkylarkModule(name = "sequence", documented = false,
    doc = "common type of lists and tuples")
public abstract class SkylarkList implements Iterable<Object>, SkylarkValue {

  /**
   * Returns the List object underlying this SkylarkList.
   * Mutating it (if mutable) will actually mutate the contents of the list.
   */
  protected abstract List<Object> getList();

  /**
   * Returns an ImmutableList object with the current underlying contents of this SkylarkList.
   */
  public abstract ImmutableList<Object> getImmutableList();

  /**
   * Returns a List object with the current underlying contents of this SkylarkList.
   * This object must not be modified, but may not be an ImmutableList.
   * It may notably be a GlobList, where appropriate.
   */
  // TODO(bazel-team): move GlobList out of Skylark, into an extension,
  // and maybe get rid of this method?
  protected abstract List<Object> getContents();

  /**
   * Returns true if this list is a tuple.
   */
  public abstract boolean isTuple();

  /**
   * The size of the list.
   */
  public final int size() {
    return getList().size();
  }

  /**
   * Returns true if the list is empty.
   */
  public final boolean isEmpty() {
    return getList().isEmpty();
  }

  /**
   * Returns the i-th element of the list.
   */
  public final Object get(int i) {
    return getList().get(i);
  }

  @Override
  public void write(Appendable buffer, char quotationMark) {
    Printer.printList(buffer, getList(), isTuple(), quotationMark);
  }

  @Override
  public final Iterator<Object> iterator() {
    return getList().iterator();
  }

  @Override
  public String toString() {
    return Printer.repr(this);
  }

  @Override
  public boolean equals(Object object) {
    return (this == object)
        || ((this.getClass() == object.getClass())
            && getList().equals(((SkylarkList) object).getList()));
  }

  @Override
  public int hashCode() {
    return getClass().hashCode() + 31 * getList().hashCode();
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
    for (Object value : list) {
      if (!type.isInstance(value)) {
        throw new EvalException(null,
            Printer.format("Illegal argument: expected type %r %sbut got type %s instead",
                type,
                description == null ? "" : String.format("for '%s' element ", description),
                EvalUtils.getDataTypeName(value)));
      }
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
      return ((SkylarkList) obj).getContents(type, description);
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
    return castList(getContents(), type, description);
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
  public static final class MutableList extends SkylarkList implements Freezable {

    private final ArrayList<Object> contents = new ArrayList<>();

    // Treat GlobList specially: external code depends on it.
    // TODO(bazel-team): make data structures *and binary operators* extensible
    // (via e.g. interface classes for each binary operator) so that GlobList
    // can be implemented outside of the core of Skylark.
    @Nullable private GlobList<?> globList;

    private final Mutability mutability;

    /**
     * Creates a MutableList from contents and a Mutability.
     * @param contents the contents of the list
     * @param mutability a Mutability context
     * @return a MutableList containing the elements
     */
    MutableList(Iterable<?> contents, Mutability mutability) {
      super();
      addAll(contents);
      if (contents instanceof GlobList<?>) {
        globList = (GlobList<?>) contents;
      }
      this.mutability = mutability;
    }

    /**
     * Creates a MutableList from contents and an Environment.
     * @param contents the contents of the list
     * @param env an Environment from which to inherit Mutability, or null for immutable
     * @return a MutableList containing the elements
     */
    public MutableList(Iterable<?> contents, @Nullable Environment env) {
      this(contents, env == null ? Mutability.IMMUTABLE : env.mutability());
    }

    /**
     * Creates a MutableList from contents.
     * @param contents the contents of the list
     * @return an actually immutable MutableList containing the elements
     */
    public MutableList(Iterable<?> contents) {
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
    public static MutableList of(@Nullable Environment env, Object... contents) {
      return new MutableList(ImmutableList.copyOf(contents), env);
    }

    /**
     * Adds one element at the end of the MutableList.
     * @param element the element to add
     */
    private void add(Object element) {
      this.contents.add(element);
    }

    /**
     * Adds all the elements at the end of the MutableList.
     * @param elements the elements to add
     */
    private void addAll(Iterable<?> elements) {
      for (Object elem : elements) {
        add(elem);
      }
    }

    private void checkMutable(Location loc, Environment env) throws EvalException {
      try {
        Mutability.checkMutable(this, env);
      } catch (MutabilityException ex) {
        throw new EvalException(loc, ex);
      }
      globList = null; // If you're going to mutate it, invalidate the underlying GlobList.
    }

    @Nullable public GlobList<?> getGlobList() {
      return globList;
    }

    /**
     * @return the GlobList if there is one, otherwise an Immutable copy of the regular contents.
     */
    @Override
    @SuppressWarnings("unchecked")
    public List<Object> getContents() {
      if (globList != null) {
        return (List<Object>) (List<?>) globList;
      }
      return getImmutableList();
    }

    /**
     * @return the GlobList if there is one, otherwise the regular contents.
     */
    private List<?> getContentsUnsafe() {
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
    public static MutableList concat(MutableList left, MutableList right, Environment env) {
      if (left.getGlobList() == null && right.getGlobList() == null) {
        return new MutableList(Iterables.concat(left, right), env);
      }
      return new MutableList(GlobList.concat(
          left.getContentsUnsafe(), right.getContentsUnsafe()), env);
    }

    /**
     * Adds one element at the end of the MutableList.
     * @param element the element to add
     * @param loc the Location at which to report any error
     * @param env the Environment requesting the modification
     */
    public void add(Object element, Location loc, Environment env) throws EvalException {
      checkMutable(loc, env);
      add(element);
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
    public void addAll(Iterable<?> elements, Location loc, Environment env) throws EvalException {
      checkMutable(loc, env);
      addAll(elements);
    }


    @Override
    public List<Object> getList() {
      return contents;
    }

    @Override
    public ImmutableList<Object> getImmutableList() {
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
  public static final class Tuple extends SkylarkList {

    private final ImmutableList<Object> contents;

    private Tuple(ImmutableList<Object> contents) {
      super();
      this.contents = contents;
    }

    /**
     * THE empty Skylark tuple.
     */
    public static final Tuple EMPTY = new Tuple(ImmutableList.of());

    /**
     * Creates a Tuple from an ImmutableList.
     */
    public static Tuple create(ImmutableList<Object> contents) {
      if (contents.isEmpty()) {
        return EMPTY;
      }
      return new Tuple(contents);
    }

    /**
     * Creates a Tuple from an Iterable.
     */
    public static Tuple copyOf(Iterable<?> contents) {
      // Do not remove <Object>: workaround for Java 7 type inference.
      return create(ImmutableList.<Object>copyOf(contents));
    }

    /**
     * Builds a Skylark tuple from a variable number of arguments.
     * @param elements a variable number of arguments (or an Array of Object-s)
     * @return a Skylark tuple containing the specified arguments as elements.
     */
    public static Tuple of(Object... elements) {
      return Tuple.create(ImmutableList.copyOf(elements));
    }

    @Override
    public List<Object> getList() {
      return contents;
    }

    @Override
    public ImmutableList<Object> getImmutableList() {
      return contents;
    }

    @Override
    public List<Object> getContents() {
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
