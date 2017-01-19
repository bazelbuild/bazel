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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.syntax.SkylarkList.MutableList;
import com.google.devtools.build.lib.syntax.SkylarkList.Tuple;
import com.google.devtools.build.lib.syntax.util.EvaluationTestCase;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for SkylarkNestedSet.
 */
@RunWith(JUnit4.class)
public class SkylarkNestedSetTest extends EvaluationTestCase {

  @Test
  public void testLegacyConstructor() throws Exception {
    eval("s = set([1, 2, 3], order='compile')");
    SkylarkNestedSet s = get("s");
    assertThat(s.getOrder().getName()).isEqualTo("compile");
    assertThat(s.getSet(Object.class)).containsExactly(1, 2, 3);
  }

  @Test
  public void testConstructor() throws Exception {
    eval("s = depset(order='stable')");
    assertThat(lookup("s")).isInstanceOf(SkylarkNestedSet.class);
  }

  @Test
  public void testGetSet() throws Exception {
    eval("s = depset(['a', 'b'])");
    assertThat(get("s").getSet(String.class)).containsExactly("a", "b").inOrder();
    assertThat(get("s").getSet(Object.class)).containsExactly("a", "b").inOrder();
    try {
      get("s").getSet(Integer.class);
      Assert.fail("getSet() with wrong type should have raised IllegalArgumentException");
    } catch (IllegalArgumentException expected) {
    }
  }

  @Test
  public void testOrder() throws Exception {
    eval("s = depset(['a', 'b'], order='compile')");
    assertThat(get("s").getSet(String.class).getOrder()).isEqualTo(Order.COMPILE_ORDER);
  }

  @Test
  public void testBadOrder() throws Exception {
    new BothModesTest().testIfExactError(
        "Invalid order: non_existing",
        "depset(['a'], order='non_existing')");
  }

  @Test
  public void testEmptyGenericType() throws Exception {
    eval("s = depset()");
    assertThat(get("s").getContentType()).isEqualTo(SkylarkType.TOP);
  }

  @Test
  public void testHomogeneousGenericType() throws Exception {
    eval("s = depset(['a', 'b', 'c'])");
    assertThat(get("s").getContentType()).isEqualTo(SkylarkType.of(String.class));
  }

  @Test
  public void testBadGenericType() throws Exception {
    new BothModesTest()
        .testIfExactError(
            "cannot add an item of type 'int' to a depset of 'string'",
            "depset(['a', 5])")
        .testIfExactError(
            "cannot add value of type 'string' to a depset",
            "depset() + 'a'");
  }

  @Test
  public void testUnionWithList() throws Exception {
    assertContainsInOrder("depset([]).union(['a', 'b', 'c'])", "a", "b", "c");
    assertContainsInOrder("depset(['a']).union(['b', 'c'])", "a", "b", "c");
    assertContainsInOrder("depset(['a', 'b']).union(['c'])", "a", "b", "c");
    assertContainsInOrder("depset(['a', 'b', 'c']).union([])", "a", "b", "c");
  }

  @Test
  public void testUnionWithDepset() throws Exception {
    assertContainsInOrder("depset([]).union(depset(['a', 'b', 'c']))", "a", "b", "c");
    assertContainsInOrder("depset(['a']).union(depset(['b', 'c']))", "a", "b", "c");
    assertContainsInOrder("depset(['a', 'b']).union(depset(['c']))", "a", "b", "c");
    assertContainsInOrder("depset(['a', 'b', 'c']).union(depset([]))", "a", "b", "c");
  }

  @Test
  public void testUnionDuplicates() throws Exception {
    assertContainsInOrder("depset(['a', 'b', 'c']).union(['a', 'b', 'c'])", "a", "b", "c");
    assertContainsInOrder("depset(['a', 'a', 'a']).union(['a', 'a'])", "a");

    assertContainsInOrder("depset(['a', 'b', 'c']).union(depset(['a', 'b', 'c']))", "a", "b", "c");
    assertContainsInOrder("depset(['a', 'a', 'a']).union(depset(['a', 'a']))", "a");
  }

  private void assertContainsInOrder(String statement, Object... expectedElements)
      throws Exception {
    new BothModesTest().testCollection(statement, expectedElements);
  }

  @Test
  public void testUnionOrder() throws Exception {
    eval(
        "def func():",
        "  s1 = depset()",
        "  s2 = depset()",
        "  s1 += ['a']",
        "  s2 += ['b']",
        "  s1 += s2",
        "  return s1",
        "s = func()");
    assertThat(get("s").toCollection()).containsExactly("b", "a").inOrder();
  }

  @Test
  public void testUnionIncompatibleOrder() throws Exception {
    checkEvalError(
        "Order mismatch: LINK_ORDER != COMPILE_ORDER",
        "depset(['a', 'b'], order='compile') + depset(['c', 'd'], order='link')");
  }

  @Test
  public void testUnionWithNonsequence() throws Exception {
    new BothModesTest()
        .testIfExactError(
            "method depset.union(new_elements: Iterable) is not applicable for arguments (int): "
            + "'new_elements' is 'int', but should be 'Iterable'",
            "depset([]).union(5)")
        .testIfExactError(
            "method depset.union(new_elements: Iterable) is not applicable for arguments (string): "
            + "'new_elements' is 'string', but should be 'Iterable'",
            "depset(['a']).union('b')");
  }

  @Test
  public void testUnionWrongNumArgs() throws Exception {
    new BothModesTest()
        .testIfErrorContains("insufficient arguments received by union", "depset(['a']).union()");
  }

  @Test
  public void testUnionNoSideEffects() throws Exception {
    eval(
        "def func():",
        "  s1 = depset(['a'])",
        "  s2 = s1.union(['b'])",
        "  return s1",
        "s = func()");
    assertThat(((SkylarkNestedSet) lookup("s")).toCollection()).isEqualTo(ImmutableList.of("a"));
  }

  @Test
  public void testFunctionReturnsDepset() throws Exception {
    eval(
        "def func():",
        "  t = depset()",
        "  t += ['a']",
        "  return t",
        "s = func()");
    assertThat(get("s")).isInstanceOf(SkylarkNestedSet.class);
    assertThat(get("s").toCollection()).containsExactly("a");
  }

  @Test
  public void testPlusEqualsWithList() throws Exception {
    eval(
        "def func():",
        "  t = depset()",
        "  t += ['a', 'b']",
        "  return t",
        "s = func()");
    assertThat(get("s").toCollection()).containsExactly("a", "b").inOrder();
  }

  @Test
  public void testPlusEqualsNoSideEffects() throws Exception {
    eval(
        "def func():",
        "  s1 = depset()",
        "  s1 += ['a']",
        "  s2 = s1",
        "  s2 += ['b']",
        "  return s1",
        "s = func()");
    assertThat(get("s").toCollection()).containsExactly("a");
  }

  @Test
  public void testFuncParamNoSideEffects() throws Exception {
    eval(
        "def func1(t):",
        "  t += ['b']",
        "def func2():",
        "  u = depset()",
        "  u += ['a']",
        "  func1(u)",
        "  return u",
        "s = func2()");
    assertThat(get("s").toCollection()).containsExactly("a");
  }

  @Test
  public void testTransitiveOrdering() throws Exception {
    eval(
        "def func():",
        "  sa = depset(['a'], order='compile')",
        "  sb = depset(['b'], order='compile')",
        "  sc = depset(['c'], order='compile') + sa",
        "  return depset() + sb + sc",
        "s = func()");
    // The iterator lists the Transitive sets first
    assertThat(get("s").toCollection()).containsExactly("b", "a", "c").inOrder();
  }

  @Test
  public void testLeftRightDirectOrdering() throws Exception {
    eval(
        "def func():",
        "  t = depset()",
        "  t += [4]",
        "  t += [2, 4]",
        "  t += [3, 4, 5]",
        "  return t",
        "s = func()");
    // All elements are direct. The iterator lists them left-to-right.
    assertThat(get("s").toCollection()).containsExactly(4, 2, 3, 5).inOrder();
  }

  @Test
  public void testToString() throws Exception {
    eval(
        "s = depset() + [2, 4, 6] + [3, 4, 5]",
        "x = str(s)");
    assertThat(lookup("x")).isEqualTo("set([2, 4, 6, 3, 5])");
  }

  @Test
  public void testToStringWithOrder() throws Exception {
    eval(
        "s = depset(order = 'link') + [2, 4, 6] + [3, 4, 5]",
        "x = str(s)");
    assertThat(lookup("x")).isEqualTo("set([2, 4, 6, 3, 5], order = \"link\")");
  }

  @SuppressWarnings("unchecked")
  private SkylarkNestedSet get(String varname) throws Exception {
    return (SkylarkNestedSet) lookup(varname);
  }

  @Test
  public void testToList() throws Exception {
    eval(
        "s = depset() + [2, 4, 6] + [3, 4, 5]",
        "x = s.to_list()");
    Object value = lookup("x");
    assertThat(value).isInstanceOf(MutableList.class);
    assertThat((Iterable<?>) value).containsExactly(2, 4, 6, 3, 5).inOrder();
  }

  @Test
  public void testOrderCompatibility() throws Exception {
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
  public void testOrderComplexUnion() throws Exception {
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
  public void testOrderBalancedTree() throws Exception {
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
  public void testOrderManyLevelsOfNesting() throws Exception {
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
