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
import static org.junit.Assert.assertEquals;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.syntax.SkylarkList.Tuple;
import com.google.devtools.build.lib.syntax.util.EvaluationTestCase;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
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
    assertEquals(Order.COMPILE_ORDER, get("n").getSet(String.class).getOrder());
  }

  @Test
  public void testEmptyNsetGenericType() throws Exception {
    eval("n = depset()");
    assertEquals(SkylarkType.TOP, get("n").getContentType());
  }

  @Test
  public void testFunctionReturnsNset() throws Exception {
    eval("def func():", "  n = depset()", "  n += ['a']", "  return n", "s = func()");
    assertEquals(ImmutableList.of("a"), get("s").toCollection());
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
    assertEquals(ImmutableList.of("a"), get("n").toCollection());
  }

  @Test
  public void testNsetNestedItem() throws Exception {
    eval(
        "def func():",
        "  n1 = depset()",
        "  n2 = depset()",
        "  n1 += ['a']",
        "  n2 += ['b']",
        "  n1 += n2",
        "  return n1",
        "n = func()");
    assertEquals(ImmutableList.of("b", "a"), get("n").toCollection());
  }

  @Test
  public void testNsetNestedItemBadOrder() throws Exception {
    checkEvalError(
        "Order mismatch: LINK_ORDER != COMPILE_ORDER",
        "depset(['a', 'b'], order='compile') + depset(['c', 'd'], order='link')");
  }

  @Test
  public void testNsetItemList() throws Exception {
    eval("def func():", "  n = depset()", "  n += ['a', 'b']", "  return n", "n = func()");
    assertEquals(ImmutableList.of("a", "b"), get("n").toCollection());
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
    assertEquals(ImmutableList.of("a"), get("n").toCollection());
  }

  @Test
  public void testNsetTransitiveOrdering() throws Exception {
    eval(
        "def func():",
        "  na = depset(['a'], order='compile')",
        "  nb = depset(['b'], order='compile')",
        "  nc = depset(['c'], order='compile') + na",
        "  return depset() + nb + nc",
        "n = func()");
    // The iterator lists the Transitive sets first
    assertEquals(ImmutableList.of("b", "a", "c"), get("n").toCollection());
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
    assertEquals(ImmutableList.of(4, 2, 3, 5), get("n").toCollection());
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
    eval("s = depset() + [2, 4, 6] + [3, 4, 5]", "x = str(s)");
    assertEquals("set([2, 4, 6, 3, 5])", lookup("x"));
  }

  @Test
  public void testNsetToStringWithOrder() throws Exception {
    eval("s = depset(order = 'link') + [2, 4, 6] + [3, 4, 5]", "x = str(s)");
    assertEquals("set([2, 4, 6, 3, 5], order = \"link\")", lookup("x"));
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

  @Test
  public void testSetOrderComplexUnion() throws Exception {
    // {1, 11, {2, 22}, {3, 33}, {4, 44}}
    List<String> preOrder = Arrays.asList("1", "11", "2", "22", "3", "33", "4", "44");
    List<String> postOrder = Arrays.asList("2", "22", "3", "33", "4", "44", "1", "11");

    MergeStrategy strategy = new MergeStrategy() {
      @Override
      public SkylarkNestedSet merge(SkylarkNestedSet[] sets) throws Exception {
        SkylarkNestedSet union = new SkylarkNestedSet(sets[0], sets[1], null);
        union = new SkylarkNestedSet(union, sets[2], null);
        union = new SkylarkNestedSet(union, sets[3], null);

        return union;
      }
    };

    runComplexOrderTest(strategy, preOrder, postOrder);
  }

  @Test
  public void testSetOrderBalancedTree() throws Exception {
    // {{1, 11, {2, 22}}, {3, 33, {4, 44}}}
    List<String> preOrder = Arrays.asList("1", "11", "2", "22", "3", "33", "4", "44");
    List<String> postOrder = Arrays.asList("2", "22", "4", "44", "3", "33", "1", "11");

    MergeStrategy strategy = new MergeStrategy() {
      @Override
      public SkylarkNestedSet merge(SkylarkNestedSet[] sets) throws Exception {
        SkylarkNestedSet leftUnion = new SkylarkNestedSet(sets[0], sets[1], null);
        SkylarkNestedSet rightUnion = new SkylarkNestedSet(sets[2], sets[3], null);
        SkylarkNestedSet union = new SkylarkNestedSet(leftUnion, rightUnion, null);

        return union;
      }
    };

    runComplexOrderTest(strategy, preOrder, postOrder);
  }

  @Test
  public void testSetOrderManyLevelsOfNesting() throws Exception {
    // {1, 11, {2, 22, {3, 33, {4, 44}}}}
    List<String> preOrder = Arrays.asList("1", "11", "2", "22", "3", "33", "4", "44");
    List<String> postOrder = Arrays.asList("4", "44", "3", "33", "2", "22", "1", "11");

    MergeStrategy strategy = new MergeStrategy() {
      @Override
      public SkylarkNestedSet merge(SkylarkNestedSet[] sets) throws Exception {
        SkylarkNestedSet union = new SkylarkNestedSet(sets[2], sets[3], null);
        union = new SkylarkNestedSet(sets[1], union, null);
        union = new SkylarkNestedSet(sets[0], union, null);

        return union;
      }
    };

    runComplexOrderTest(strategy, preOrder, postOrder);
  }

  private interface MergeStrategy {
    SkylarkNestedSet merge(SkylarkNestedSet[] sets) throws Exception;
  }

  private void runComplexOrderTest(
      MergeStrategy strategy, List<String> preOrder, List<String> postOrder) throws Exception {
    Map<Order, List<String>> expected = createExpectedMap(preOrder, postOrder);
    for (Order order : Order.values()) {
      SkylarkNestedSet union = strategy.merge(makeFourSets(order));
      assertThat(union.toCollection()).containsExactlyElementsIn(expected.get(order)).inOrder();
    }
  }

  private Map<Order, List<String>> createExpectedMap(
      List<String> preOrder, List<String> postOrder) {
    Map<Order, List<String>> expected = new HashMap<>();

    for (Order order : Order.values()) {
      expected.put(order, isPostOrder(order) ? postOrder : preOrder);
    }

    return expected;
  }

  private boolean isPostOrder(Order order) {
    return order == Order.STABLE_ORDER || order == Order.COMPILE_ORDER;
  }

  private SkylarkNestedSet[] makeFourSets(Order order) throws Exception {
    return new SkylarkNestedSet[] {
        new SkylarkNestedSet(order, Tuple.of("1", "11"), null),
        new SkylarkNestedSet(order, Tuple.of("2", "22"), null),
        new SkylarkNestedSet(order, Tuple.of("3", "33"), null),
        new SkylarkNestedSet(order, Tuple.of("4", "44"), null)};
  }
}
