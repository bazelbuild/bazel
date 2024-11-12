// Copyright 2024 The Bazel Authors. All rights reserved.
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

package net.starlark.java.eval;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Sets;
import java.util.AbstractSet;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;

/** A finite, mutable set of Starlark values. */
@StarlarkBuiltin(
    name = "set",
    category = "core",
    doc =
        """
<b>Experimental</b>. This API is experimental and may change at any time. Please do not depend on
it. It may be enabled on an experimental basis by setting
<code>--experimental_enable_starlark_set</code>.

<p>The built-in mutable set type. Example set expressions:

<pre class=language-python>
x = set()           # x is an empty set
y = set([1, 2, 3])  # y is a set with 3 elements
3 in y              # True
0 in y              # False
len(x)              # 0
len(y)              # 3
</pre>

<p>A set used in Boolean context is true if and only if it is non-empty.

<pre class=language-python>
s = set()
"non-empty" if s else "empty"  # "empty"
t = set(["x", "y"])
"non-empty" if t else "empty"  # "non-empty"
</pre>

<p>The elements of a set must be hashable; <code>x</code> may be an element of a set if and only if
<code>x</code> may be used as a key of a dict.

<p>A set itself is <em>not</em> hashable; therefore, you cannot have a set with another set as an
element.

<p>You cannot access the elements of a set by index, but you can iterate over them, and you can
obtain the list of a set's elements in iteration order using the <code>list()</code> built-in
function. Just like for lists, it is an error to mutate a set while it is being iterated over. The
order of iteration matches insertion order:

<pre class=language-python>
s = set([3, 1, 3])
s.add(2)
# prints 3, 1, 2
for item in s:
    print(item)
list(s)  # [3, 1, 2]
</pre>

<p>A set <code>s</code> is equal to <code>t</code> if and only if <code>t</code> is a set containing
the same elements, possibly with a different iteration order. In particular, a set is
<code>not</code> equal to its list of elements.

<p>Sets are not ordered; the <code>&lt;</code>, <code>&lt;=</code>, <code>&gt;</code>, and
<code>&gt;=</code> operations are not defined for sets, and a list of sets cannot be sorted - unlike
in Python.

<p>The <code>|</code> operation on two sets returns the union of the two sets: a set containing the
elements found in either one or both of the original sets. The <code>|</code> operation has an
augmented assignment version; <code>s |= t</code> adds to <code>s</code> all the elements of
<code>t</code>.

<pre class=language-python>
set([1, 2]) | set([3, 2])  # set([1, 2, 3])
s = set([1, 2])
s |= set([2, 3, 4])        # s now equals set([1, 2, 3, 4])
</pre>

<p>The <code>&amp;</code> operation on two sets returns the intersection of the two sets: a set
containing only the elements found in both of the original sets. The <code>&amp;</code> operation
has an augmented assignment version; <code>s &amp;= t</code> removes from <code>s</code> all the
elements not found in <code>t</code>.

<pre class=language-python>
set([1, 2]) & set([2, 3])  # set([2])
set([1, 2]) & set([3, 4])  # set()
s = set([1, 2])
s &amp;= set([0, 1])           # s now equals set([1])
</pre>

<p>The <code>-</code> operation on two sets returns the difference of the two sets: a set containing
the elements found in the left-hand side set but not the right-hand site set. The <code>-</code>
operation has an augmented assignment version; <code>s -= t</code> removes from <code>s</code> all
the elements found in <code>t</code>.

<pre class=language-python>
set([1, 2]) - set([2, 3])  # set([1])
set([1, 2]) - set([3, 4])  # set([1, 2])
s = set([1, 2])
s -= set([0, 1])           # s now equals set([2])
</pre>

<p>The <code>^</code> operation on two sets returns the symmetric difference of the two sets: a set
containing the elements found in exactly one of the two original sets, but not in both. The
<code>^</code> operation has an augmented assignment version; <code>s ^= t</code> removes from
<code>s</code> any element of <code>t</code> found in <code>s</code> and adds to <code>s</code> any
element of <code>t</code> not found in <code>s</code>.

<pre class=language-python>
set([1, 2]) ^ set([2, 3])  # set([1, 3])
set([1, 2]) ^ set([3, 4])  # set([1, 2, 3, 4])
s = set([1, 2])
s ^= set([0, 1])           # s now equals set([2, 0])
</pre>
""")
public final class StarlarkSet<E> extends AbstractSet<E>
    implements Mutability.Freezable, StarlarkMembershipTestable, StarlarkIterable<E> {

  private static final StarlarkSet<?> EMPTY = new StarlarkSet<>(ImmutableSet.of());

  // Either LinkedHashSet<E> or ImmutableSet<E>.
  private final Set<E> contents;
  // Number of active iterators (unused once frozen).
  private transient int iteratorCount; // transient for serialization by Bazel

  /** Final except for {@link #unsafeShallowFreeze}; must not be modified any other way. */
  private Mutability mutability;

  @SuppressWarnings("NonApiType")
  private StarlarkSet(Mutability mutability, LinkedHashSet<E> contents) {
    checkNotNull(mutability);
    checkArgument(mutability != Mutability.IMMUTABLE);
    this.mutability = mutability;
    this.contents = contents;
  }

  private StarlarkSet(ImmutableSet<E> contents) {
    // An immutable set might as well store its contents as an ImmutableSet, since ImmutableSet
    // both is more memory-efficient than LinkedHashSet and also it has the requisite deterministic
    // iteration order.
    this.mutability = Mutability.IMMUTABLE;
    this.contents = contents;
  }

  @Override
  public boolean truth() {
    return !isEmpty();
  }

  @Override
  public boolean isImmutable() {
    return mutability().isFrozen();
  }

  @Override
  public boolean updateIteratorCount(int delta) {
    if (mutability().isFrozen()) {
      return false;
    }
    if (delta > 0) {
      iteratorCount++;
    } else if (delta < 0) {
      iteratorCount--;
    }
    return iteratorCount > 0;
  }

  @Override
  public void checkHashable() throws EvalException {
    // Even a frozen set is unhashable.
    throw Starlark.errorf("unhashable type: 'set'");
  }

  @Override
  public int hashCode() {
    return contents.hashCode();
  }

  @Override
  public void repr(Printer printer) {
    if (isEmpty()) {
      printer.append("set()");
    } else {
      printer.printList(this, "set([", ", ", "])");
    }
  }

  @Override
  public String toString() {
    return Starlark.repr(this);
  }

  @Override
  public boolean equals(Object o) {
    return contents.equals(o);
  }

  @Override
  public Iterator<E> iterator() {
    if (contents instanceof ImmutableSet) {
      return contents.iterator();
    } else {
      // Prohibit mutation through Iterator.remove().
      return Collections.unmodifiableSet(contents).iterator();
    }
  }

  @Override
  public int size() {
    return contents.size();
  }

  @Override
  public boolean isEmpty() {
    return contents.isEmpty();
  }

  @Override
  public Object[] toArray() {
    return contents.toArray();
  }

  @Override
  public <T> T[] toArray(T[] a) {
    return contents.toArray(a);
  }

  @Override
  public boolean contains(Object o) {
    return contents.contains(o);
  }

  @Override
  public boolean containsAll(Collection<?> c) {
    return contents.containsAll(c);
  }

  @Override
  public boolean containsKey(StarlarkSemantics semantics, Object element) {
    return contents.contains(element);
  }

  /** Returns an immutable empty set. */
  // Safe because the empty singleton is immutable.
  @SuppressWarnings("unchecked")
  public static <E> StarlarkSet<E> empty() {
    return (StarlarkSet<E>) EMPTY;
  }

  /** Returns a new empty set with the specified mutability. */
  public static <E> StarlarkSet<E> of(@Nullable Mutability mu) {
    if (mu == null) {
      mu = Mutability.IMMUTABLE;
    }
    if (mu == Mutability.IMMUTABLE) {
      return empty();
    } else {
      return new StarlarkSet<>(mu, Sets.newLinkedHashSetWithExpectedSize(1));
    }
  }

  /**
   * Returns a set with the specified mutability containing the entries of {@code elements}. Tries
   * to elide copying if {@code elements} is immutable.
   *
   * @param elements a collection of elements, which must be Starlark-hashable (note that this
   *     method assumes but does not verify their hashability), to add to the new set.
   */
  public static <E> StarlarkSet<E> copyOf(
      @Nullable Mutability mu, Collection<? extends E> elements) {
    if (elements.isEmpty()) {
      return of(mu);
    }

    if (mu == null) {
      mu = Mutability.IMMUTABLE;
    }

    if (mu == Mutability.IMMUTABLE) {
      if (elements instanceof ImmutableSet) {
        elements.forEach(Starlark::checkValid);
        @SuppressWarnings("unchecked")
        ImmutableSet<E> immutableSet = (ImmutableSet<E>) elements;
        return new StarlarkSet<>(immutableSet);
      }

      if (elements instanceof StarlarkSet && ((StarlarkSet<?>) elements).isImmutable()) {
        @SuppressWarnings("unchecked")
        StarlarkSet<E> starlarkSet = (StarlarkSet<E>) elements;
        return starlarkSet;
      }

      ImmutableSet.Builder<E> immutableSetBuilder =
          ImmutableSet.builderWithExpectedSize(elements.size());
      elements.forEach(e -> immutableSetBuilder.add(Starlark.checkValid(e)));
      return new StarlarkSet<>(immutableSetBuilder.build());
    } else {
      LinkedHashSet<E> linkedHashSet = Sets.newLinkedHashSetWithExpectedSize(elements.size());
      elements.forEach(e -> linkedHashSet.add(Starlark.checkValid(e)));
      return new StarlarkSet<>(mu, linkedHashSet);
    }
  }

  private static <E> StarlarkSet<E> wrapOrImmutableCopy(Mutability mu, LinkedHashSet<E> elements) {
    checkNotNull(mu);
    if (mu == Mutability.IMMUTABLE) {
      return elements.isEmpty() ? empty() : new StarlarkSet<>(ImmutableSet.copyOf(elements));
    } else {
      return new StarlarkSet<>(mu, elements);
    }
  }

  /**
   * A variant of {@link #copyOf} intended to be used from Starlark. Unlike {@link #copyOf}, this
   * method does verify that the elements being added to the set are Starlark-hashable.
   *
   * @param elements a collection of elements to add to the new set, or a map whose keys will be
   *     added to the new set.
   */
  public static StarlarkSet<Object> checkedCopyOf(@Nullable Mutability mu, Object elements)
      throws EvalException {
    @SuppressWarnings("unchecked")
    Collection<Object> collection =
        (Collection<Object>) toHashableCollection(elements, "set constructor argument");
    return copyOf(mu, collection);
  }

  /**
   * Returns an immutable set containing the entries of {@code elements}. Tries to elide copying if
   * {@code elements} is already immutable.
   *
   * @param elements a collection of elements, which must be Starlark-hashable (note that this
   *     method assumes but does not verify their hashability), to add to the new set.
   */
  public static <E> StarlarkSet<E> immutableCopyOf(Collection<? extends E> elements) {
    return copyOf(null, elements);
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

  @StarlarkMethod(
      name = "issubset",
      doc =
          """
Returns true of this set is a subset of another.

<p>For example,
<pre class=language-python>
set([1, 2]).issubset([1, 2, 3]) == True
set([1, 2]).issubset([1, 2]) == True
set([1, 2]).issubset([2, 3]) == False
</pre>
""",
      parameters = {@Param(name = "other", doc = "A set, sequence, or dict.")})
  public boolean isSubset(Object other) throws EvalException {
    return toCollection(other, "issubset argument").containsAll(this.contents);
  }

  @StarlarkMethod(
      name = "issuperset",
      doc =
          """
Returns true of this set is a superset of another.

<p>For example,
<pre class=language-python>
set([1, 2, 3]).issuperset([1, 2]) == True
set([1, 2, 3]).issuperset([1, 2, 3]) == True
set([1, 2, 3]).issuperset([2, 3, 4]) == False
</pre>
""",
      parameters = {@Param(name = "other", doc = "A set, sequence, or dict.")})
  public boolean isSuperset(Object other) throws EvalException {
    return contents.containsAll(toCollection(other, "issuperset argument"));
  }

  @StarlarkMethod(
      name = "isdisjoint",
      doc =
          """
Returns true if this set has no elements in common with another.

<p>For example,
<pre class=language-python>
set([1, 2]).isdisjoint([3, 4]) == True
set().isdisjoint(set()) == True
set([1, 2]).isdisjoint([2, 3]) == False
</pre>
""",
      parameters = {@Param(name = "other", doc = "A set, sequence, or dict.")})
  public boolean isDisjoint(Object other) throws EvalException {
    return Collections.disjoint(this.contents, toCollection(other, "isdisjoint argument"));
  }

  /**
   * Intended for use from Starlark; if used from Java, the caller should ensure that the elements
   * to be added are instances of {@code E}.
   */
  @StarlarkMethod(
      name = "update",
      doc =
          """
Adds the elements found in others to this set.

<p>For example,
<pre class=language-python>
x = set([1, 2])
x.update([2, 3], [3, 4])
# x is now set([1, 2, 3, 4])
</pre>
""",
      extraPositionals = @Param(name = "others", doc = "Sets, sequences, or dicts."))
  public void update(Tuple others) throws EvalException {
    Starlark.checkMutable(this);
    for (Object other : others) {
      @SuppressWarnings("unchecked")
      Collection<? extends E> otherCollection =
          (Collection<? extends E>) toHashableCollection(other, "update argument");
      contents.addAll(otherCollection);
    }
  }

  @StarlarkMethod(
      name = "add",
      doc = "Adds an element to the set.",
      parameters = {@Param(name = "element", doc = "Element to add.")})
  public void addElement(E element) throws EvalException {
    Starlark.checkMutable(this);
    Starlark.checkHashable(element);
    contents.add(element);
  }

  @StarlarkMethod(
      name = "remove",
      doc =
          """
Removes an element, which must be present in the set, from the set. Fails if the element was not
present in the set.
""",
      parameters = {@Param(name = "element", doc = "Element to remove.")})
  public void removeElement(E element) throws EvalException {
    Starlark.checkMutable(this);
    if (!contents.remove(element)) {
      throw Starlark.errorf("element %s not found in set", Starlark.repr(element));
    }
  }

  @StarlarkMethod(
      name = "discard",
      doc = "Removes an element from the set if it is present.",
      parameters = {@Param(name = "element", doc = "Element to discard.")})
  public void discard(E element) throws EvalException {
    Starlark.checkMutable(this);
    contents.remove(element);
  }

  @StarlarkMethod(
      name = "pop",
      doc = "Removes and returns the first element of the set. Fails if the set is empty.")
  public E pop() throws EvalException {
    Starlark.checkMutable(this);
    if (isEmpty()) {
      throw Starlark.errorf("set is empty");
    }
    E element = contents.iterator().next();
    contents.remove(element);
    return element;
  }

  @StarlarkMethod(name = "clear", doc = "Removes all the elements of the set.")
  public void clearElements() throws EvalException {
    Starlark.checkMutable(this);
    contents.clear();
  }

  @StarlarkMethod(
      name = "union",
      doc =
          """
Returns a new mutable set containing the union of this set with others.

<p>For example,
<pre class=language-python>
set([1, 2]).union([2, 3, 4], [4, 5]) == set([1, 2, 3, 4, 5])
</pre>
""",
      extraPositionals = @Param(name = "others", doc = "Sets, sequences, or dicts."),
      useStarlarkThread = true)
  public StarlarkSet<?> union(Tuple others, StarlarkThread thread) throws EvalException {
    LinkedHashSet<Object> newContents = new LinkedHashSet<>(contents);
    for (Object other : others) {
      newContents.addAll(toHashableCollection(other, "union argument"));
    }
    return wrapOrImmutableCopy(thread.mutability(), newContents);
  }

  @StarlarkMethod(
      name = "intersection",
      doc =
          """
Returns a new mutable set containing the intersection of this set with others.

<p>For example,
<pre class=language-python>
set([1, 2, 3]).intersection([1, 2], [2, 3]) == set([2])
</pre>
""",
      extraPositionals = @Param(name = "others", doc = "Sets, sequences, or dicts."),
      useStarlarkThread = true)
  public StarlarkSet<?> intersection(Tuple others, StarlarkThread thread) throws EvalException {
    LinkedHashSet<Object> newContents = new LinkedHashSet<>(contents);
    for (Object other : others) {
      newContents.retainAll(toCollection(other, "intersection argument"));
    }
    return wrapOrImmutableCopy(thread.mutability(), newContents);
  }

  @StarlarkMethod(
      name = "intersection_update",
      doc =
          """
Removes any elements not found in all others from this set.

<p>For example,
<pre class=language-python>
x = set([1, 2, 3, 4])
x.intersection_update([2, 3], [3, 4])
# x is now set([3])
</pre>
""",
      extraPositionals = @Param(name = "others", doc = "Sets, sequences, or dicts."))
  public void intersectionUpdate(Tuple others) throws EvalException {
    Starlark.checkMutable(this);
    for (Object other : others) {
      contents.retainAll(toCollection(other, "intersection_update argument"));
    }
  }

  @StarlarkMethod(
      name = "difference",
      doc =
          """
Returns a new mutable set containing the difference of this set with others.

<p>For example,
<pre class=language-python>
set([1, 2, 3]).intersection([1, 2], [2, 3]) == set([2])
</pre>
""",
      extraPositionals = @Param(name = "others", doc = "Sets, sequences, or dicts."),
      useStarlarkThread = true)
  public StarlarkSet<?> difference(Tuple others, StarlarkThread thread) throws EvalException {
    LinkedHashSet<Object> newContents = new LinkedHashSet<>(contents);
    for (Object other : others) {
      newContents.removeAll(toCollection(other, "difference argument"));
    }
    return wrapOrImmutableCopy(thread.mutability(), newContents);
  }

  @StarlarkMethod(
      name = "difference_update",
      doc =
          """
Removes any elements found in any others from this set.

<p>For example,
<pre class=language-python>
x = set([1, 2, 3, 4])
x.difference_update([2, 3], [3, 4])
# x is now set([1])
</pre>
""",
      extraPositionals = @Param(name = "others", doc = "Sets, sequences, or dicts."))
  public void differenceUpdate(Tuple others) throws EvalException {
    Starlark.checkMutable(this);
    for (Object other : others) {
      contents.removeAll(toCollection(other, "intersection_update argument"));
    }
  }

  @StarlarkMethod(
      name = "symmetric_difference",
      doc =
          """
Returns a new mutable set containing the symmetric difference of this set with another set,
sequence, or dict.

<p>For example,
<pre class=language-python>
set([1, 2, 3]).symmetric_difference([2, 3, 4]) == set([1, 4])
</pre>
""",
      parameters = {@Param(name = "other", doc = "A set, sequence, or dict.")},
      useStarlarkThread = true)
  public StarlarkSet<?> symmetricDifference(Object other, StarlarkThread thread)
      throws EvalException {
    LinkedHashSet<Object> newContents = new LinkedHashSet<>(contents);
    for (Object element : toHashableCollection(other, "symmetric_difference argument")) {
      if (contents.contains(element)) {
        newContents.remove(element);
      } else {
        newContents.add(element);
      }
    }
    return wrapOrImmutableCopy(thread.mutability(), newContents);
  }

  /**
   * Intended for use from Starlark; if used from Java, the caller should ensure that the elements
   * to be added are instances of {@code E}.
   */
  @StarlarkMethod(
      name = "symmetric_difference_update",
      doc =
          """
Returns a new mutable set containing the symmetric difference of this set with another set,
sequence, or dict.

<p>For example,
<pre class=language-python>
set([1, 2, 3]).symmetric_difference([2, 3, 4]) == set([1, 4])
</pre>
""",
      parameters = {@Param(name = "other", doc = "A set, sequence, or dict.")})
  public void symmetricDifferenceUpdate(Object other) throws EvalException {
    Starlark.checkMutable(this);
    ImmutableSet<E> originalContents = ImmutableSet.copyOf(contents);
    for (Object element : toHashableCollection(other, "symmetric_difference_update argument")) {
      if (originalContents.contains(element)) {
        contents.remove(element);
      } else {
        @SuppressWarnings("unchecked")
        E castElement = (E) element;
        contents.add(castElement);
      }
    }
  }

  /**
   * Verifies that {@code other} is either a collection or a map.
   *
   * @return {@code other} if it is a collection, or the key set of {@code other} if it is a map.
   */
  private static Collection<?> toCollection(Object other, String what) throws EvalException {
    if (other instanceof Collection) {
      return (Collection<?>) other;
    } else if (other instanceof Map) {
      return ((Map<?, ?>) other).keySet();
    }
    throw notSizedIterableError(other, what);
  }

  /**
   * A variant of {@link #toCollection} which additionally checks whether the returned collection's
   * elements are Starlark-hashable.
   *
   * @return {@code other} if it is a collection, or the key set of {@code other} if it is a map.
   */
  private static Collection<?> toHashableCollection(Object other, String what)
      throws EvalException {
    if (other instanceof Collection) {
      Collection<?> collection = (Collection<?>) other;
      // Assume that elements of a StarlarkSet have already been checked to be hashable.
      if (!(collection instanceof StarlarkSet)) {
        for (Object element : collection) {
          Starlark.checkHashable(element);
        }
      }
      return collection;
    } else if (other instanceof Map) {
      Set<?> keySet = ((Map<?, ?>) other).keySet();
      // Assume that keys of a Dict have already been checked to be hashable.
      if (!(other instanceof Dict)) {
        for (Object element : keySet) {
          Starlark.checkHashable(element);
        }
      }
      return keySet;
    }
    throw notSizedIterableError(other, what);
  }

  // Starlark doesn't have a "sized iterable" interface - so we enumerate the types we expect.
  private static EvalException notSizedIterableError(Object other, String what) {
    return Starlark.errorf(
        "for %s got value of type '%s', want a set, sequence, or dict", what, Starlark.type(other));
  }

  // Prohibit Java Set mutators.

  /**
   * @deprecated use {@link #addElement} instead.
   */
  @Deprecated
  @Override
  public boolean add(E e) {
    throw new UnsupportedOperationException();
  }

  /**
   * @deprecated use {@link #update} instead.
   */
  @Deprecated
  @Override
  public boolean addAll(Collection<? extends E> c) {
    throw new UnsupportedOperationException();
  }

  /**
   * @deprecated use {@link #clearElements} instead.
   */
  @Deprecated
  @Override
  public void clear() {
    throw new UnsupportedOperationException();
  }

  /**
   * @deprecated use {@link #removeElement} instead.
   */
  @Deprecated
  @Override
  public boolean remove(Object o) {
    throw new UnsupportedOperationException();
  }

  /**
   * @deprecated use {@link #differenceUpdate} instead.
   */
  @Deprecated
  @Override
  public boolean removeAll(Collection<?> c) {
    throw new UnsupportedOperationException();
  }

  /**
   * @deprecated use {@link #intersectionUpdate} instead.
   */
  @Deprecated
  @Override
  public boolean retainAll(Collection<?> c) {
    throw new UnsupportedOperationException();
  }
}
