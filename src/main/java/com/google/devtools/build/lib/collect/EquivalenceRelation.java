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

package com.google.devtools.build.lib.collect;

/**
 * A comparison function, which imposes an equivalence relation on some collection of objects.
 *
 * <p>The ordering imposed by an EquivalenceRelation <tt>e</tt> on a set of elements <tt>S</tt> is
 * said to be <i>consistent with equals</i> if and only if <tt>(compare((Object)e1,
 * (Object)e2)==0)</tt> has the same boolean value as <tt>e1.equals((Object)e2)</tt> for every
 * <tt>e1</tt> and <tt>e2</tt> in <tt>S</tt>.
 *
 * <p>
 *
 * <p>Unlike {@link java.util.Comparator}, whose implementations are often consistent with equals,
 * the applications for which EquivalenceRelation instances are used means that its implementations
 * rarely are. They may are usually more or less discriminative than the default equivalence
 * relation for the type.
 *
 * <p>For example, consider possible equivalence relations for {@link java.lang.Integer}: the
 * default equivalence defined by Integer.equals() is based on the integer value is represents, but
 * two alternative equivalences would be {@link EquivalenceRelation#IDENTITY} (object
 * identity&mdash;a more discriminative relation) or <i>parity</i> (under which all even numbers,
 * odd numbers are considered equivalent to each other&mdash;a less discriminative relation).
 */
// TODO: Consider phasing out this interface in favour of com.google.common.base.Equivalence
@FunctionalInterface
public interface EquivalenceRelation<T> {
  // This should be a superinterface of Comparator.

  /**
   * Compares its two arguments for equivalence.  Returns zero if they are
   * considered equivalent, or non-zero otherwise.<p>
   *
   * The implementor must ensure that the relation is
   *
   * reflexive (<tt>compare(x,x)==0</tt> for all x),
   *
   * symmetric (<tt>compare(x,y)==compare(y,x)<tt> for all x, y),
   *
   * and transitive <tt>(compare(x, y)==0 &amp;&amp; compare(y,
   * z)==0</tt> implies <tt>compare(x, z)==0</tt>.<p>
   *
   * @param o1 the first object to be compared.
   * @param o2 the second object to be compared.
   * @return zero if the two objects are equivalent; some other integer value
   *   otherwise.
   * @throws ClassCastException if the arguments' types prevent them from
   *   being compared by this EquivalenceRelation.
   */
  int compare(T o1, T o2);

  /**
   * The object-identity equivalence relation. This is the strictest possible equivalence relation
   * for objects, and considers two values equal iff they are references to the same object
   * instance.
   */
  EquivalenceRelation<?> IDENTITY = (EquivalenceRelation<Object>) (o1, o2) -> (o1 == o2) ? 0 : -1;

  /**
   * The default equivalence relation for type T, using T.equals(). This relation considers two
   * values equivalent if either they are both null, or o1.equals(o2).
   */
  EquivalenceRelation<?> DEFAULT =
      (EquivalenceRelation<Object>) (o1, o2) -> (o1 == null ? o2 == null : o1.equals(o2)) ? 0 : -1;
}
