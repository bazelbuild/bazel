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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.events.Location;

import java.util.Collection;
import java.util.Iterator;
import java.util.List;

/**
 * A class to handle lists and tuples in Skylark.
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
        + "List elements have to be of the same type, <code>[1, 2, \"c\"]</code> results in an "
        + "error. Lists - just like everything - are immutable, therefore <code>x[1] = \"a\""
        + "</code> is not supported.")
// TODO(bazel-team): should we instead have it implement List<Object> like ImmutableList does?
public abstract class SkylarkList implements Iterable<Object> {

  private final boolean tuple;
  private final SkylarkType contentType;

  private SkylarkList(boolean tuple, SkylarkType contentType) {
    this.tuple = tuple;
    this.contentType = contentType;
  }

  /**
   * The size of the list.
   */
  public abstract int size();

  /**
   * Returns true if the list is empty.
   */
  public abstract boolean isEmpty();

  /**
   * Returns the i-th element of the list.
   */
  public abstract Object get(int i);

  /**
   * Returns true if this list is a tuple.
   */
  public boolean isTuple() {
    return tuple;
  }

  @VisibleForTesting
  public SkylarkType getContentType() {
    return contentType;
  }

  @Override
  public String toString() {
    return EvalUtils.prettyPrintValue(this);
  }

  // TODO(bazel-team): we should be very careful using this method. Check and remove
  // auto conversions on the Java-Skylark interface if possible.
  /**
   * Returns a mutable Java list copy of this SkylarkList if it's a list or an
   * immutable copy if it's a tuple.
   */
  public abstract List<Object> toList();

  @SuppressWarnings("unchecked")
  public <T> Iterable<T> to(Class<T> type) {
    Preconditions.checkArgument(this == EMPTY_LIST || contentType.canBeCastTo(type));
    return (Iterable<T>) this;
  }

  private static final class EmptySkylarkList extends SkylarkList {
    private EmptySkylarkList(boolean tuple) {
      super(tuple, SkylarkType.TOP);
    }

    @Override
    public Iterator<Object> iterator() {
      return ImmutableList.of().iterator();
    }

    @Override
    public int size() {
      return 0;
    }

    @Override
    public boolean isEmpty() {
      return true;
    }

    @Override
    public Object get(int i) {
      throw new UnsupportedOperationException();
    }

    @Override
    public List<Object> toList() {
      return isTuple() ? ImmutableList.of() : Lists.newArrayList();
    }

    @Override
    public String toString() {
      return "[]";
    }
  }

  /**
   * An empty Skylark list.
   */
  public static final SkylarkList EMPTY_LIST = new EmptySkylarkList(false);

  /**
   * An empty Skylark tuple.
   */
  public static final SkylarkList EMPTY_TUPLE = new EmptySkylarkList(true);

  private static final class SimpleSkylarkList extends SkylarkList {
    private final ImmutableList<Object> list;

    private SimpleSkylarkList(ImmutableList<Object> list, boolean tuple, SkylarkType contentType) {
      super(tuple, contentType);
      this.list = Preconditions.checkNotNull(list);
    }

    @Override
    public Iterator<Object> iterator() {
      return list.iterator();
    }

    @Override
    public int size() {
      return list.size();
    }

    @Override
    public boolean isEmpty() {
      return list.isEmpty();
    }

    @Override
    public Object get(int i) {
      return list.get(i);
    }

    @Override
    public List<Object> toList() {
      return isTuple() ? list : Lists.newArrayList(list);
    }

    @Override
    public String toString() {
      return EvalUtils.prettyPrintValue(this);
    }

    @Override
    public int hashCode() {
      return list.hashCode();
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      }
      if (!(obj instanceof SimpleSkylarkList)) {
        return false;
      }
      SimpleSkylarkList other = (SimpleSkylarkList) obj;
      return other.list.equals(this.list);
    }
  }

  /**
   * A Skylark list to support lazy iteration (i.e. we only call iterator on the object this
   * list masks when it's absolutely necessary). This is useful if iteration is expensive
   * (e.g. NestedSet-s). Size(), get() and isEmpty() are expensive operations but
   * concatenation is quick.
   */
  private static final class LazySkylarkList extends SkylarkList {
    private final Iterable<Object> iterable;
    private ImmutableList<Object> list = null;

    private LazySkylarkList(Iterable<Object> iterable, boolean tuple, SkylarkType contentType) {
      super(tuple, contentType);
      this.iterable = Preconditions.checkNotNull(iterable);
    }

    @Override
    public Iterator<Object> iterator() {
      return iterable.iterator();
    }

    @Override
    public int size() {
      return Iterables.size(iterable);
    }

    @Override
    public boolean isEmpty() {
      return Iterables.isEmpty(iterable);
    }

    @Override
    public Object get(int i) {
      return getList().get(i);
    }

    @Override
    public List<Object> toList() {
      ImmutableList<Object> result = getList();
      return isTuple() ? result : Lists.newArrayList(result);
    }

    private ImmutableList<Object> getList() {
      if (list == null) {
        list = ImmutableList.copyOf(iterable);
      }
      return list;
    }
  }

  /**
   * A Skylark list to support quick concatenation of lists. Concatenation is O(1),
   * size(), isEmpty() is O(n), get() is O(h).
   */
  private static final class ConcatenatedSkylarkList extends SkylarkList {
    private final SkylarkList left;
    private final SkylarkList right;

    private ConcatenatedSkylarkList(
        SkylarkList left, SkylarkList right, boolean tuple, SkylarkType contentType) {
      super(tuple, contentType);
      this.left = Preconditions.checkNotNull(left);
      this.right = Preconditions.checkNotNull(right);
    }

    @Override
    public Iterator<Object> iterator() {
      return Iterables.concat(left, right).iterator();
    }

    @Override
    public int size() {
      // We shouldn't evaluate the size function until it's necessary, because it can be expensive
      // for lazy lists (e.g. lists containing a NestedSet).
      // TODO(bazel-team): make this class more clever to store the size and empty parameters
      // for every non-LazySkylarkList member.
      return left.size() + right.size();
    }

    @Override
    public boolean isEmpty() {
      return left.isEmpty() && right.isEmpty();
    }

    @Override
    public Object get(int i) {
      int leftSize = left.size();
      if (i < leftSize) {
        return left.get(i);
      } else {
        return right.get(i - leftSize);
      }
    }

    @Override
    public List<Object> toList() {
      List<Object> result = ImmutableList.<Object>builder().addAll(left).addAll(right).build();
      return isTuple() ? result : Lists.newArrayList(result);
    }
  }

  /**
   * @param elements the contents of the list
   * @param contentType a SkylarkType for the contents of the list
   * @return a Skylark list containing elements without a type check
   * Only use if you already know for sure all elements are of the specified type.
   */
  public static SkylarkList list(Collection<?> elements, SkylarkType contentType) {
    if (elements.isEmpty()) {
      return EMPTY_LIST;
    }
    return new SimpleSkylarkList(ImmutableList.copyOf(elements), false, contentType);
  }

  /**
   * @param elements the contents of the list
   * @param contentType a Java class for the contents of the list
   * @return a Skylark list containing elements without a type check
   * Only use if you already know for sure all elements are of the specified type.
   */
  @SuppressWarnings("unchecked")
  public static SkylarkList list(Collection<?> elements, Class<?> contentType) {
    return list(elements, SkylarkType.of(contentType));
  }

  /**
   * @param elements the contents of the list
   * @param contentType a SkylarkType for the contents of the list
   * @return a Skylark list without a type check and without creating an immutable copy
   * Therefore the iterable containing elements must be immutable
   * (which is not checked here so callers must be extra careful).
   * This way it's possibly to create a SkylarkList without requesting the original iterator.
   * This can be useful for nested set - list conversions.
   * Only use if you already know for sure all elements are of the specified type.
   */
  @SuppressWarnings("unchecked")
  public static SkylarkList lazyList(Iterable<?> elements, SkylarkType contentType) {
    return new LazySkylarkList((Iterable<Object>) elements, false, contentType);
  }

  /**
   * @param elements the contents of the list
   * @param contentType a Java class for the contents of the list
   * @return a Skylark list without a type check and without creating an immutable copy
   * Therefore the iterable containing elements must be immutable
   * (which is not checked here so callers must be extra careful).
   * This way it's possibly to create a SkylarkList without requesting the original iterator.
   * This can be useful for nested set - list conversions.
   * Only use if you already know for sure all elements are of the specified type.
   */
  @SuppressWarnings("unchecked")
  public static SkylarkList lazyList(Iterable<?> elements, Class<?> contentType) {
    return lazyList(elements, SkylarkType.of(contentType));
  }

  /**
   * @param elements the contents of the list
   * @return a Skylark list containing elements
   * @throws EvalException in case the list is not monomorphic
   */
  public static SkylarkList list(Collection<?> elements, Location loc) throws EvalException {
    if (elements.isEmpty()) {
      return EMPTY_LIST;
    }
    return new SimpleSkylarkList(
        ImmutableList.copyOf(elements), false, getContentType(elements, loc));
  }

  private static SkylarkType getContentType(Collection<?> elements, Location loc)
      throws EvalException {
    SkylarkType type = SkylarkType.TOP;
    for (Object element : elements) {
      SkylarkType elementType = SkylarkType.typeOf(element);
      SkylarkType inter = SkylarkType.intersection(type, elementType);
      if (inter == SkylarkType.BOTTOM) {
        throw new EvalException(loc, String.format(
            "Incompatible types in list: found a %s but the previous elements were %ss",
            elementType, type));
      } else {
        type = inter;
      }
    }
    return type;
  }

  /**
   * Returns a Skylark list created from Skylark lists left and right. Throws an exception
   * if they are not of the same generic type.
   */
  public static SkylarkList concat(SkylarkList left, SkylarkList right, Location loc)
      throws EvalException {
    if (left.isTuple() != right.isTuple()) {
      throw new EvalException(loc, "cannot concatenate lists and tuples");
    }
    if (left == EMPTY_LIST) {
      return right;
    }
    if (right == EMPTY_LIST) {
      return left;
    }
    SkylarkType type = SkylarkType.intersection(left.contentType, right.contentType);
    if (type == SkylarkType.BOTTOM) {
      throw new EvalException(loc, String.format("cannot concatenate list of %s with list of %s",
              left.contentType, right.contentType));
    }
    return new ConcatenatedSkylarkList(left, right, left.isTuple(), type);
  }

  /**
   * Build a Skylark tuple from a Java List
   * @param elements a List of objects
   * @return a Skylark tuple containing the specified List of objects as elements.
   */
  public static SkylarkList tuple(List<?> elements) {
    // Tuple elements do not have to have the same type.
    return new SimpleSkylarkList(ImmutableList.copyOf(elements), true, SkylarkType.TOP);
  }

  /**
   * Build a Skylark tuple from a variable number of arguments
   * @param elements a variable number of arguments (or an array of objects)
   * @return a Skylark tuple containing the specified arguments as elements.
   */
  public static SkylarkList tuple(Object... elements) {
    return new SimpleSkylarkList(ImmutableList.copyOf(elements), true, SkylarkType.TOP);
  }
}
