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
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.syntax.SkylarkList.MutableList;
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
  public void testConstructor() throws Exception {
    eval("s = depset(order='default')");
    assertThat(lookup("s")).isInstanceOf(SkylarkNestedSet.class);
  }

  @Test
  public void testTuplePairs() throws Exception {
    eval(
        // Depsets with tuple-pairs
        "s_one = depset([('1', '2'), ('3', '4')])",
        "s_two = depset(direct = [('1', '2'), ('3', '4'), ('5', '6')])",
        "s_three = depset(transitive = [s_one, s_two])",
        "s_four = depset(direct = [('1', '3')], transitive = [s_one, s_two])",
        // Depsets with tuple-pairs and non-pair tuples are considered just tuple depsets.
        "s_five = depset(direct = [(1, 3, 5)], transitive = [s_one, s_two])",
        "s_six = depset(transitive = [s_one, s_five])",
        "s_seven = depset(direct = [('1', '3')], transitive = [s_one, s_five])",
        "s_eight = depset(direct = [(1, 3)], transitive = [s_one, s_two])");
    assertThat(get("s_one").getContentType()).isEqualTo(SkylarkType.STRING_PAIR);
    assertThat(get("s_two").getContentType()).isEqualTo(SkylarkType.STRING_PAIR);
    assertThat(get("s_three").getContentType()).isEqualTo(SkylarkType.STRING_PAIR);
    assertThat(get("s_four").getContentType()).isEqualTo(SkylarkType.STRING_PAIR);

    assertThat(get("s_five").getContentType()).isEqualTo(SkylarkType.TUPLE);
    assertThat(get("s_six").getContentType()).isEqualTo(SkylarkType.TUPLE);
    assertThat(get("s_seven").getContentType()).isEqualTo(SkylarkType.TUPLE);
    assertThat(get("s_eight").getContentType()).isEqualTo(SkylarkType.TUPLE);
  }

  @Test
  public void testGetSet() throws Exception {
    eval("s = depset(['a', 'b'])");
    assertThat(get("s").getSet(String.class)).containsExactly("a", "b").inOrder();
    assertThat(get("s").getSet(Object.class)).containsExactly("a", "b").inOrder();
    assertThrows(
        IllegalArgumentException.class,
        () -> get("s").getSet(Integer.class)
    );
  }

  @Test
  public void testGetSetDirect() throws Exception {
    eval("s = depset(direct = ['a', 'b'])");
    assertThat(get("s").getSet(String.class)).containsExactly("a", "b").inOrder();
    assertThat(get("s").getSet(Object.class)).containsExactly("a", "b").inOrder();
    assertThrows(
        IllegalArgumentException.class,
        () -> get("s").getSet(Integer.class)
    );
  }

  @Test
  public void testGetSetItems() throws Exception {
    eval("s = depset(items = ['a', 'b'])");
    assertThat(get("s").getSet(String.class)).containsExactly("a", "b").inOrder();
    assertThat(get("s").getSet(Object.class)).containsExactly("a", "b").inOrder();
    assertThrows(
        IllegalArgumentException.class,
        () -> get("s").getSet(Integer.class)
    );
  }


  @Test
  public void testToCollection() throws Exception {
    eval("s = depset(['a', 'b'])");
    assertThat(get("s").toCollection(String.class)).containsExactly("a", "b").inOrder();
    assertThat(get("s").toCollection(Object.class)).containsExactly("a", "b").inOrder();
    assertThat(get("s").toCollection()).containsExactly("a", "b").inOrder();
    assertThrows(
        IllegalArgumentException.class,
        () -> get("s").toCollection(Integer.class)
    );
  }

  @Test
  public void testToCollectionDirect() throws Exception {
    eval("s = depset(direct = ['a', 'b'])");
    assertThat(get("s").toCollection(String.class)).containsExactly("a", "b").inOrder();
    assertThat(get("s").toCollection(Object.class)).containsExactly("a", "b").inOrder();
    assertThat(get("s").toCollection()).containsExactly("a", "b").inOrder();
    assertThrows(
        IllegalArgumentException.class,
        () -> get("s").toCollection(Integer.class)
    );
  }

  @Test
  public void testToCollectionItems() throws Exception {
    eval("s = depset(items = ['a', 'b'])");
    assertThat(get("s").toCollection(String.class)).containsExactly("a", "b").inOrder();
    assertThat(get("s").toCollection(Object.class)).containsExactly("a", "b").inOrder();
    assertThat(get("s").toCollection()).containsExactly("a", "b").inOrder();
    assertThrows(
        IllegalArgumentException.class,
        () -> get("s").toCollection(Integer.class)
    );
  }

  @Test
  public void testOrder() throws Exception {
    eval("s = depset(['a', 'b'], order='postorder')");
    assertThat(get("s").getSet(String.class).getOrder()).isEqualTo(Order.COMPILE_ORDER);
  }

  @Test
  public void testOrderDirect() throws Exception {
    eval("s = depset(direct = ['a', 'b'], order='postorder')");
    assertThat(get("s").getSet(String.class).getOrder()).isEqualTo(Order.COMPILE_ORDER);
  }

  @Test
  public void testOrderItems() throws Exception {
    eval("s = depset(items = ['a', 'b'], order='postorder')");
    assertThat(get("s").getSet(String.class).getOrder()).isEqualTo(Order.COMPILE_ORDER);
  }

  @Test
  public void testBadOrder() throws Exception {
    new BothModesTest().testIfExactError(
        "Invalid order: non_existing",
        "depset(['a'], order='non_existing')");
  }

  @Test
  public void testBadOrderDirect() throws Exception {
    new BothModesTest().testIfExactError(
        "Invalid order: non_existing",
        "depset(direct = ['a'], order='non_existing')");
  }

  @Test
  public void testBadOrderItems() throws Exception {
    new BothModesTest().testIfExactError(
        "Invalid order: non_existing",
        "depset(items = ['a'], order='non_existing')");
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
  public void testHomogeneousGenericTypeDirect() throws Exception {
    eval("s = depset(['a', 'b', 'c'], transitive = [])");
    assertThat(get("s").getContentType()).isEqualTo(SkylarkType.of(String.class));
  }

  @Test
  public void testHomogeneousGenericTypeItems() throws Exception {
    eval("s = depset(items = ['a', 'b', 'c'], transitive = [])");
    assertThat(get("s").getContentType()).isEqualTo(SkylarkType.of(String.class));
  }

  @Test
  public void testHomogeneousGenericTypeTransitive() throws Exception {
    eval("s = depset(['a', 'b', 'c'], transitive = [depset(['x'])])");
    assertThat(get("s").getContentType()).isEqualTo(SkylarkType.of(String.class));
  }

  @Test
  public void testTransitiveIncompatibleOrder() throws Exception {
    checkEvalError(
        "Order 'postorder' is incompatible with order 'topological'",
        "depset(['a', 'b'], order='postorder',",
        "       transitive = [depset(['c', 'd'], order='topological')])");
  }

  @Test
  public void testBadGenericType() throws Exception {
    new BothModesTest().testIfExactError(
        "cannot add an item of type 'int' to a depset of 'string'",
        "depset(['a', 5])");
  }

  @Test
  public void testBadGenericTypeDirect() throws Exception {
    new BothModesTest().testIfExactError(
        "cannot add an item of type 'int' to a depset of 'string'",
        "depset(direct = ['a', 5])");
  }

  @Test
  public void testBadGenericTypeItems() throws Exception {
    new BothModesTest().testIfExactError(
        "cannot add an item of type 'int' to a depset of 'string'",
        "depset(items = ['a', 5])");
  }

  @Test
  public void testBadGenericTypeTransitive() throws Exception {
    new BothModesTest().testIfExactError(
        "cannot add an item of type 'int' to a depset of 'string'",
        "depset(['a', 'b'], transitive=[depset([1])])");
  }

  @Test
  public void testLegacyAndNewApi() throws Exception {
    new BothModesTest().testIfExactError(
        "Do not pass both 'direct' and 'items' argument to depset constructor.",
        "depset(['a', 'b'], direct = ['c', 'd'])");
  }

  @Test
  public void testItemsAndTransitive() throws Exception {
    new BothModesTest().testIfExactError(
        "expected type 'sequence' for items but got type 'depset' instead",
        "depset(items = depset(), transitive = [depset()])");
  }

  @Test
  public void testTooManyPositionals() throws Exception {
    new BothModesTest().testIfExactError(
        "too many (3) positional arguments in call to "
            + "depset(items = [], order: string = \"default\", *, "
            + "direct: sequence or NoneType = None, "
            + "transitive: sequence of depsets or NoneType = None)",
        "depset([], 'default', [])");
  }


  @Test
  public void testTransitiveOrder() throws Exception {
    assertContainsInOrder("depset([], transitive=[depset(['a', 'b', 'c'])])", "a", "b", "c");
    assertContainsInOrder("depset(['a'], transitive = [depset(['b', 'c'])])", "b", "c", "a");
    assertContainsInOrder("depset(['a', 'b'], transitive = [depset(['c'])])", "c", "a", "b");
    assertContainsInOrder("depset(['a', 'b', 'c'], transitive = [depset([])])", "a", "b", "c");
  }

  @Test
  public void testTransitiveOrderItems() throws Exception {
    assertContainsInOrder("depset(items=[], transitive=[depset(['a', 'b', 'c'])])", "a", "b", "c");
    assertContainsInOrder("depset(items=['a'], transitive = [depset(['b', 'c'])])", "b", "c", "a");
    assertContainsInOrder("depset(items=['a', 'b'], transitive = [depset(['c'])])", "c", "a", "b");
    assertContainsInOrder("depset(items=['a', 'b', 'c'], transitive = [depset([])])",
        "a", "b", "c");
  }

  @Test
  public void testTransitiveOrderDirect() throws Exception {
    assertContainsInOrder("depset(direct=[], transitive=[depset(['a', 'b', 'c'])])", "a", "b", "c");
    assertContainsInOrder("depset(direct=['a'], transitive = [depset(['b', 'c'])])", "b", "c", "a");
    assertContainsInOrder("depset(direct=['a', 'b'], transitive = [depset(['c'])])", "c", "a", "b");
    assertContainsInOrder("depset(direct=['a', 'b', 'c'], transitive = [depset([])])",
        "a", "b", "c");
  }

  @Test
  public void testIncompatibleUnion() throws Exception {
    new BothModesTest("--incompatible_depset_union=true")
        .testIfErrorContains(
            "depset method `.union` has been removed", "depset([]).union(['a', 'b', 'c'])");

    new BothModesTest("--incompatible_depset_union=true")
        .testIfErrorContains("`+` operator on a depset is forbidden", "depset([]) + ['a']");

    new BothModesTest("--incompatible_depset_union=true")
        .testIfErrorContains("`|` operator on a depset is forbidden", "depset([]) | ['a']");
  }

  @Test
  public void testUnionWithList() throws Exception {
    env = newEnvironmentWithSkylarkOptions("--incompatible_depset_union=false");
    assertContainsInOrder("depset([]).union(['a', 'b', 'c'])", "a", "b", "c");
    assertContainsInOrder("depset(['a']).union(['b', 'c'])", "a", "b", "c");
    assertContainsInOrder("depset(['a', 'b']).union(['c'])", "a", "b", "c");
    assertContainsInOrder("depset(['a', 'b', 'c']).union([])", "a", "b", "c");
  }

  @Test
  public void testUnionWithDepset() throws Exception {
    env = newEnvironmentWithSkylarkOptions("--incompatible_depset_union=false");
    assertContainsInOrder("depset([]).union(depset(['a', 'b', 'c']))", "a", "b", "c");
    assertContainsInOrder("depset(['a']).union(depset(['b', 'c']))", "b", "c", "a");
    assertContainsInOrder("depset(['a', 'b']).union(depset(['c']))", "c", "a", "b");
    assertContainsInOrder("depset(['a', 'b', 'c']).union(depset([]))", "a", "b", "c");
  }

  @Test
  public void testUnionDuplicates() throws Exception {
    env = newEnvironmentWithSkylarkOptions("--incompatible_depset_union=false");
    assertContainsInOrder("depset(['a', 'b', 'c']).union(['a', 'b', 'c'])", "a", "b", "c");
    assertContainsInOrder("depset(['a', 'a', 'a']).union(['a', 'a'])", "a");

    assertContainsInOrder("depset(['a', 'b', 'c']).union(depset(['a', 'b', 'c']))", "a", "b", "c");
    assertContainsInOrder("depset(['a', 'a', 'a']).union(depset(['a', 'a']))", "a");
  }


  private void assertContainsInOrder(String statement, Object... expectedElements)
      throws Exception {
    assertThat(((SkylarkNestedSet) eval(statement)).toCollection())
        .containsExactly(expectedElements)
        .inOrder();
  }

  @Test
  public void testUnionOrder() throws Exception {
    env = newEnvironmentWithSkylarkOptions("--incompatible_depset_union=false");
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
        "Order mismatch: topological != postorder",
        "depset(['a', 'b'], order='postorder') + depset(['c', 'd'], order='topological')");
  }

  @Test
  public void testUnionWithNonsequence() throws Exception {
    new BothModesTest()
        .testIfExactError(
            "cannot union value of type 'int' to a depset",
            "depset([]).union(5)")
        .testIfExactError(
            "cannot union value of type 'string' to a depset",
            "depset(['a']).union('b')");
  }

  @Test
  public void testUnionWrongNumArgs() throws Exception {
    new BothModesTest().testIfErrorContains(
        "parameter 'new_elements' has no default value, in method call union() of 'depset'",
        "depset(['a']).union()");
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
        "  sa = depset(['a'], order='postorder')",
        "  sb = depset(['b'], order='postorder')",
        "  sc = depset(['c'], order='postorder') + sa",
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
    env = newEnvironmentWithSkylarkOptions("--incompatible_depset_union=false");
    eval(
        "s = depset() + [2, 4, 6] + [3, 4, 5]",
        "x = str(s)");
    assertThat(lookup("x")).isEqualTo("depset([2, 4, 6, 3, 5])");
  }

  @Test
  public void testToStringWithOrder() throws Exception {
    env = newEnvironmentWithSkylarkOptions("--incompatible_depset_union=false");
    eval(
        "s = depset(order = 'topological') + [2, 4, 6] + [3, 4, 5]",
        "x = str(s)");
    assertThat(lookup("x")).isEqualTo("depset([2, 4, 6, 3, 5], order = \"topological\")");
  }

  @SuppressWarnings("unchecked")
  private SkylarkNestedSet get(String varname) throws Exception {
    return (SkylarkNestedSet) lookup(varname);
  }

  @Test
  public void testToList() throws Exception {
    env = newEnvironmentWithSkylarkOptions("--incompatible_depset_union=false");
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
    //  (b) at least one order is "default"

    for (Order first : Order.values()) {
      SkylarkNestedSet s1 = SkylarkNestedSet.of(first, Tuple.of("1", "11"), null);

      for (Order second : Order.values()) {
        SkylarkNestedSet s2 = SkylarkNestedSet.of(second, Tuple.of("2", "22"), null);

        boolean compatible = true;

        try {
          SkylarkNestedSet.of(s1, s2, null);
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

    MergeStrategy strategy =
        new MergeStrategy() {
          @Override
          public SkylarkNestedSet merge(SkylarkNestedSet[] sets) throws Exception {
            SkylarkNestedSet union = SkylarkNestedSet.of(sets[0], sets[1], null);
            union = SkylarkNestedSet.of(union, sets[2], null);
            union = SkylarkNestedSet.of(union, sets[3], null);

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

    MergeStrategy strategy =
        new MergeStrategy() {
          @Override
          public SkylarkNestedSet merge(SkylarkNestedSet[] sets) throws Exception {
            SkylarkNestedSet leftUnion = SkylarkNestedSet.of(sets[0], sets[1], null);
            SkylarkNestedSet rightUnion = SkylarkNestedSet.of(sets[2], sets[3], null);
            SkylarkNestedSet union = SkylarkNestedSet.of(leftUnion, rightUnion, null);

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

    MergeStrategy strategy =
        new MergeStrategy() {
          @Override
          public SkylarkNestedSet merge(SkylarkNestedSet[] sets) throws Exception {
            SkylarkNestedSet union = SkylarkNestedSet.of(sets[2], sets[3], null);
            union = SkylarkNestedSet.of(sets[1], union, null);
            union = SkylarkNestedSet.of(sets[0], union, null);

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
      SkylarkNestedSet.of(order, Tuple.of("1", "11"), null),
      SkylarkNestedSet.of(order, Tuple.of("2", "22"), null),
      SkylarkNestedSet.of(order, Tuple.of("3", "33"), null),
      SkylarkNestedSet.of(order, Tuple.of("4", "44"), null)
    };
  }
}
