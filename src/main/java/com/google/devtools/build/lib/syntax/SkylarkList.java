// Copyright 2014 Google Inc. All rights reserved.
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
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.syntax.Mutability.Freezable;
import com.google.devtools.build.lib.syntax.Mutability.MutabilityException;

import java.util.ArrayList;
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
  // TODO(bazel-team): make this public no more.
  public abstract List<Object> getList();

  /**
   * Returns an ImmutableList object with the current underlying contents of this SkylarkList.
   */
  public abstract ImmutableList<Object> getImmutableList();

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

  @SuppressWarnings("unchecked")
  public <T> Iterable<T> to(Class<T> type) {
    for (Object value : getList()) {
      Preconditions.checkArgument(
          type.isInstance(value),
          Printer.formattable("list element %r is not of type %r", value, type));
    }
    return (Iterable<T>) this;
  }

  /**
   * A class for mutable lists.
   */
  @SkylarkModule(name = "list",
      doc = "A language built-in type to support lists. Example of list literal:<br>"
      + "<pre class=language-python>x = [1, 2, 3]</pre>"
      + "Accessing elements is possible using indexing (starts from <code>0</code>):<br>"
      + "<pre class=language-python>e = x[1]   # e == 2</pre>"
      + "Lists support the <code>+</code> operator to concatenate two lists. Example:<br>"
      + "<pre class=language-python>x = [1, 2] + [3, 4]   # x == [1, 2, 3, 4]\n"
      + "x = [\"a\", \"b\"]\n"
      + "x += [\"c\"]            # x == [\"a\", \"b\", \"c\"]</pre>"
      + "Lists are mutable, as in Python.")
  public static final class MutableList extends SkylarkList implements Freezable {

    private final ArrayList<Object> contents = new ArrayList<>();

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
     * Creates a MutableList from contents and an Environment.
     * @param contents the contents of the list
     * @return an actually immutable MutableList containing the elements
     */
    public MutableList(Iterable<?> contents) {
      this(contents, Mutability.IMMUTABLE);
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
  @SkylarkModule(name = "tuple",
      doc = "A language built-in type to support tuples. Example of tuple literal:<br>"
      + "<pre class=language-python>x = (1, 2, 3)</pre>"
      + "Accessing elements is possible using indexing (starts from <code>0</code>):<br>"
      + "<pre class=language-python>e = x[1]   # e == 2</pre>"
      + "Lists support the <code>+</code> operator to concatenate two tuples. Example:<br>"
      + "<pre class=language-python>x = (1, 2) + (3, 4)   # x == (1, 2, 3, 4)\n"
      + "x = (\"a\", \"b\")\n"
      + "x += (\"c\",)            # x == (\"a\", \"b\", \"c\")</pre>"
      + "Tuples are immutable, therefore <code>x[1] = \"a\"</code> is not supported.")
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
      return create(ImmutableList.copyOf(contents));
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
