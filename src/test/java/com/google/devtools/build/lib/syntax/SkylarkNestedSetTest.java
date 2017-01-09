// Copyright 2015 The Bazel Authors. All rights reserved.
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

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.syntax.SkylarkList.Tuple;
import com.google.devtools.build.lib.syntax.util.EvaluationTestCase;
import java.util.Collection;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for SkylarkNestedSet.
 */
@RunWith(JUnit4.class)
public class SkylarkNestedSetTest extends EvaluationTestCase {

  @Test
  public void testLegacySetConstructor() throws Exception {
    eval("ds = set([1, 2, 3], order='compile')");
    SkylarkNestedSet ds = get("ds");
    assertThat(ds.getOrder().getName()).isEqualTo("compile");
    assertThat(ds.expandedSet()).isEqualTo(ImmutableSet.of(1, 2, 3));
  }

  @Test
  public void testNsetBuilder() throws Exception {
    eval("n = depset(order='stable')");
    assertThat(lookup("n")).isInstanceOf(SkylarkNestedSet.class);
  }

  @Test
  public void testNsetOrder() throws Exception {
    eval("n = depset(['a', 'b'], order='compile')");
    assertThat(get("n").getSet(String.class).getOrder()).isEqualTo(Order.COMPILE_ORDER);
  }

  @Test
  public void testEmptyNsetGenericType() throws Exception {
    eval("n = depset()");
    assertThat(get("n").getContentType()).isEqualTo(SkylarkType.TOP);
  }

  @Test
  public void testFunctionReturnsNset() throws Exception {
    eval(
        "def func():",
        "  n = depset()",
        "  n += ['a']",
        "  return n",
        "s = func()");
    assertThat(get("s")).isInstanceOf(SkylarkNestedSet.class);
    assertThat(get("s").toCollection()).containsExactly("a");
  }

  @Test
  public void testNsetTwoReferences() throws Exception {
    eval(
        "def func():",
        "  n1 = depset()",
        "  n1 += ['a']",
        "  n2 = n1",
        "  n2 += ['b']",
        "  return n1",
        "n = func()");
    assertThat(get("n").toCollection()).containsExactly("a");
  }

  @Test
  public void testNsetUnionOrder() throws Exception {
    //    -----o----- []
    //   /           \
    //  o ['a', 'b']  o ['b', 'c']
    //  |             |
    //  o []          o []
    eval(
        "def func():",
        "  n1 = depset()",
        "  n2 = depset()",
        "  n1 += ['a', 'b']",
        "  n2 += ['b', 'c']",
        "  n1 += n2",
        "  return n1",
        "n = func()");
    // Under old behavior, the left operand was made a parent of the right operand. If that had
    // happened, then the post-order traversal would yield [b, c, a] rather than [a, b, c].
    assertThat(get("n").toCollection()).containsExactly("a", "b", "c").inOrder();
  }

  @Test
  public void testNsetMultiUnionOrder() throws Exception {
    //       --------o-------- []
    //      /                 \
    //   ---o--- []          ---o--- []
    //  /       \           /       \
    //  o ['a']  o ['b']   o ['c']   o ['d']
    eval(
        "def func():",
        "  na = depset(['a'], order='compile')",
        "  nb = depset(['b'], order='compile')",
        "  nc = depset(['c'], order='compile')",
        "  nd = depset(['d'], order='compile')",
        "  return na + nb + (nc + nd)",
        "n = func()");
    // Under old behavior, in a chain of three or more unions s1 + s2 + ... + sn (left-
    // associative), the post-order traversal yielded in order the elements of s2, s3, ..., sn, s1.
    // Now it should just be s1, ..., sn.
    assertThat(get("n").toCollection()).containsExactly("a", "b", "c", "d").inOrder();
  }
  
  @Test
  public void testNsetTransitiveOrder() throws Exception {
    //    ---o--- []
    //   /       \
    //  o ['a']   o ['b']
    //            |
    //            o ['c']
    eval(
        "def func():",
        "  na = depset(['a'])",
        "  nc = depset(['c'])",
        "  nb = nc + ['b']",
        "  return na + nb",
        "n = func()");
    assertThat(get("n").toCollection()).containsExactly("a", "c", "b").inOrder();
  }

  private Collection<Object> getDiamondTraversal(String order) throws Exception {
    //      o ['a']
    //      |
    //    --o--
    //   /     \
    //  o ['b'] o ['c']
    //   \     /
    //    --o-- ['d']
    initialize();
    eval(
        "def func():",
        "  nd = depset(['d'], order='" + order + "')",
        "  nb = nd + ['b']",
        "  nc = nd + ['c']",
        "  na = nb + nc + ['a']",
        "  return na",
        "n = func()");
    return get("n").toCollection();
  }

  @Test
  public void testNsetDiamond() throws Exception {
    assertThat(getDiamondTraversal("compile")).containsExactly("d", "b", "c", "a").inOrder();
    assertThat(getDiamondTraversal("naive_link")).containsExactly("a", "b", "d", "c").inOrder();
    assertThat(getDiamondTraversal("link")).containsExactly("a", "b", "c", "d").inOrder();
  }

  @Test
  public void testNsetNestedItemBadOrder() throws Exception {
    checkEvalError(
        "LINK_ORDER != COMPILE_ORDER",
        "depset(['a', 'b'], order='compile') + depset(['c', 'd'], order='link')");
  }

  @Test
  public void testNsetItemList() throws Exception {
    eval(
        "def func():",
        "  n = depset()",
        "  n += ['a', 'b']",
        "  return n",
        "n = func()");
    assertThat(get("n").toCollection()).containsExactly("a", "b").inOrder();
  }

  @Test
  public void testNsetFuncParamNoSideEffects() throws Exception {
    eval(
        "def func1(n):",
        "  n += ['b']",
        "def func2():",
        "  n = depset()",
        "  n += ['a']",
        "  func1(n)",
        "  return n",
        "n = func2()");
    assertThat(get("n").toCollection()).containsExactly("a");
  }

  @Test
  public void testNsetOrdering() throws Exception {
    eval(
        "def func():",
        "  na = depset()",
        "  na += [4]",
        "  na += [2, 4]",
        "  na += [3, 4, 5]",
        "  return na",
        "n = func()");
    // The iterator lists the Transitive sets first
    assertThat(get("n").toCollection()).containsExactly(4, 2, 3, 5).inOrder();
  }

  @Test
  public void testNsetBadOrder() throws Exception {
    checkEvalError("Invalid order: non_existing", "depset(order='non_existing')");
  }

  @Test
  public void testNsetBadRightOperand() throws Exception {
    checkEvalError("cannot add value of type 'string' to a depset", "l = ['a']", "depset() + l[0]");
  }

  @Test
  public void testNsetToString() throws Exception {
    //  o [3, 4, 5]
    //  |
    //  o [2, 4, 6]
    //  |
    //  o []
    eval(
        "s = depset() + [2, 4, 6] + [3, 4, 5]",
        "x = str(s)");
    assertThat(lookup("x")).isEqualTo("set([2, 4, 6, 3, 5])");
  }

  @Test
  public void testNsetToStringWithOrder() throws Exception {
    //  o [3, 4, 5]
    //  |
    //  o [2, 4, 6]
    //  |
    //  o []
    eval(
        "s = depset(order = 'link') + [2, 4, 6] + [3, 4, 5]",
        "x = str(s)");
    assertThat(lookup("x")).isEqualTo("set([3, 5, 2, 4, 6], order = \"link\")");
  }

  @SuppressWarnings("unchecked")
  private SkylarkNestedSet get(String varname) throws Exception {
    return (SkylarkNestedSet) lookup(varname);
  }

  @Test
  public void testSetOrderCompatibility() throws Exception {
    // Two sets are compatible if
    //  (a) both have the same order or
    //  (b) at least one order is "stable"

    for (Order first : Order.values()) {
      SkylarkNestedSet s1 = new SkylarkNestedSet(first, Tuple.of("1", "11"), null);

      for (Order second : Order.values()) {
        SkylarkNestedSet s2 = new SkylarkNestedSet(second, Tuple.of("2", "22"), null);

        boolean compatible = true;

        try {
          new SkylarkNestedSet(s1, s2, null);
        } catch (Exception ex) {
          compatible = false;
        }

        assertThat(compatible).isEqualTo(areOrdersCompatible(first, second));
      }
    }
  }

  private boolean areOrdersCompatible(Order first, Order second) {
    return first == Order.STABLE_ORDER || second == Order.STABLE_ORDER || first == second;
  }
}
