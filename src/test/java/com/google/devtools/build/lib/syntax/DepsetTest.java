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

import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.syntax.util.EvaluationTestCase;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for Depset. */
@RunWith(JUnit4.class)
public class DepsetTest extends EvaluationTestCase {

  @Test
  public void testConstructor() throws Exception {
    exec("s = depset(order='default')");
    assertThat(lookup("s")).isInstanceOf(Depset.class);
  }

  @Test
  public void testTuples() throws Exception {
    exec(
        "s_one = depset([('1', '2'), ('3', '4')])",
        "s_two = depset(direct = [('1', '2'), ('3', '4'), ('5', '6')])",
        "s_three = depset(transitive = [s_one, s_two])",
        "s_four = depset(direct = [('1', '3')], transitive = [s_one, s_two])",
        "s_five = depset(direct = [('1', '3', '5')], transitive = [s_one, s_two])",
        "s_six = depset(transitive = [s_one, s_five])",
        "s_seven = depset(direct = [('1', '3')], transitive = [s_one, s_five])",
        "s_eight = depset(direct = [(1, 3)], transitive = [s_one, s_two])"); // note, tuple of int
    assertThat(get("s_one").getContentType()).isEqualTo(SkylarkType.TUPLE);
    assertThat(get("s_two").getContentType()).isEqualTo(SkylarkType.TUPLE);
    assertThat(get("s_three").getContentType()).isEqualTo(SkylarkType.TUPLE);
    assertThat(get("s_eight").getContentType()).isEqualTo(SkylarkType.TUPLE);

    assertThat(get("s_four").getSet(Tuple.class))
        .containsExactly(
            Tuple.of("1", "3"), Tuple.of("1", "2"), Tuple.of("3", "4"), Tuple.of("5", "6"));
    assertThat(get("s_five").getSet(Tuple.class))
        .containsExactly(
            Tuple.of("1", "3", "5"), Tuple.of("1", "2"), Tuple.of("3", "4"), Tuple.of("5", "6"));
    assertThat(get("s_eight").getSet(Tuple.class))
        .containsExactly(
            Tuple.of(1, 3), Tuple.of("1", "2"), Tuple.of("3", "4"), Tuple.of("5", "6"));
  }

  @Test
  public void testGetSet() throws Exception {
    exec("s = depset(['a', 'b'])");
    assertThat(get("s").getSet(String.class)).containsExactly("a", "b").inOrder();
    assertThat(get("s").getSet(Object.class)).containsExactly("a", "b").inOrder();
    assertThrows(Depset.TypeException.class, () -> get("s").getSet(Integer.class));
  }

  @Test
  public void testGetSetDirect() throws Exception {
    exec("s = depset(direct = ['a', 'b'])");
    assertThat(get("s").getSet(String.class)).containsExactly("a", "b").inOrder();
    assertThat(get("s").getSet(Object.class)).containsExactly("a", "b").inOrder();
    assertThrows(Depset.TypeException.class, () -> get("s").getSet(Integer.class));
  }

  @Test
  public void testGetSetItems() throws Exception {
    exec("s = depset(items = ['a', 'b'])");
    assertThat(get("s").getSet(String.class)).containsExactly("a", "b").inOrder();
    assertThat(get("s").getSet(Object.class)).containsExactly("a", "b").inOrder();
    assertThrows(Depset.TypeException.class, () -> get("s").getSet(Integer.class));
  }


  @Test
  public void testToCollection() throws Exception {
    exec("s = depset(['a', 'b'])");
    assertThat(get("s").toCollection(String.class)).containsExactly("a", "b").inOrder();
    assertThat(get("s").toCollection(Object.class)).containsExactly("a", "b").inOrder();
    assertThat(get("s").toCollection()).containsExactly("a", "b").inOrder();
    assertThrows(Depset.TypeException.class, () -> get("s").toCollection(Integer.class));
  }

  @Test
  public void testToCollectionDirect() throws Exception {
    exec("s = depset(direct = ['a', 'b'])");
    assertThat(get("s").toCollection(String.class)).containsExactly("a", "b").inOrder();
    assertThat(get("s").toCollection(Object.class)).containsExactly("a", "b").inOrder();
    assertThat(get("s").toCollection()).containsExactly("a", "b").inOrder();
    assertThrows(Depset.TypeException.class, () -> get("s").toCollection(Integer.class));
  }

  @Test
  public void testToCollectionItems() throws Exception {
    exec("s = depset(items = ['a', 'b'])");
    assertThat(get("s").toCollection(String.class)).containsExactly("a", "b").inOrder();
    assertThat(get("s").toCollection(Object.class)).containsExactly("a", "b").inOrder();
    assertThat(get("s").toCollection()).containsExactly("a", "b").inOrder();
    assertThrows(Depset.TypeException.class, () -> get("s").toCollection(Integer.class));
  }

  @Test
  public void testOrder() throws Exception {
    exec("s = depset(['a', 'b'], order='postorder')");
    assertThat(get("s").getSet(String.class).getOrder()).isEqualTo(Order.COMPILE_ORDER);
  }

  @Test
  public void testOrderDirect() throws Exception {
    exec("s = depset(direct = ['a', 'b'], order='postorder')");
    assertThat(get("s").getSet(String.class).getOrder()).isEqualTo(Order.COMPILE_ORDER);
  }

  @Test
  public void testOrderItems() throws Exception {
    exec("s = depset(items = ['a', 'b'], order='postorder')");
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
    exec("s = depset()");
    assertThat(get("s").getContentType()).isEqualTo(SkylarkType.TOP);
  }

  @Test
  public void testHomogeneousGenericType() throws Exception {
    exec("s = depset(['a', 'b', 'c'])");
    assertThat(get("s").getContentType()).isEqualTo(SkylarkType.of(String.class));
  }

  @Test
  public void testHomogeneousGenericTypeDirect() throws Exception {
    exec("s = depset(['a', 'b', 'c'], transitive = [])");
    assertThat(get("s").getContentType()).isEqualTo(SkylarkType.of(String.class));
  }

  @Test
  public void testHomogeneousGenericTypeItems() throws Exception {
    exec("s = depset(items = ['a', 'b', 'c'], transitive = [])");
    assertThat(get("s").getContentType()).isEqualTo(SkylarkType.of(String.class));
  }

  @Test
  public void testHomogeneousGenericTypeTransitive() throws Exception {
    exec("s = depset(['a', 'b', 'c'], transitive = [depset(['x'])])");
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
    new BothModesTest()
        .testIfErrorContains(
            "expected no more than 2 positional arguments, but got 3", "depset([], 'default', [])");
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
        .testIfErrorContains("`+` operator on a depset is forbidden", "depset([]) + ['a']");

    new BothModesTest("--incompatible_depset_union=true")
        .testIfErrorContains("`|` operator on a depset is forbidden", "depset([]) | ['a']");
  }

  private void assertContainsInOrder(String statement, Object... expectedElements)
      throws Exception {
    assertThat(((Depset) eval(statement)).toCollection())
        .containsExactly(expectedElements)
        .inOrder();
  }

  @Test
  public void testUnionOrder() throws Exception {
    thread = newStarlarkThreadWithSkylarkOptions("--incompatible_depset_union=false");
    exec(
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
    thread = newStarlarkThreadWithSkylarkOptions("--incompatible_depset_union=false");
    checkEvalError(
        "Order mismatch: topological != postorder",
        "depset(['a', 'b'], order='postorder') + depset(['c', 'd'], order='topological')");
  }

  @Test
  public void testFunctionReturnsDepset() throws Exception {
    thread = newStarlarkThreadWithSkylarkOptions("--incompatible_depset_union=false");
    exec(
        "def func():", //
        "  t = depset()",
        "  t += ['a']",
        "  return t",
        "s = func()");
    assertThat(get("s")).isInstanceOf(Depset.class);
    assertThat(get("s").toCollection()).containsExactly("a");
  }

  @Test
  public void testPlusEqualsWithList() throws Exception {
    thread = newStarlarkThreadWithSkylarkOptions("--incompatible_depset_union=false");
    exec(
        "def func():", //
        "  t = depset()",
        "  t += ['a', 'b']",
        "  return t",
        "s = func()");
    assertThat(get("s").toCollection()).containsExactly("a", "b").inOrder();
  }

  @Test
  public void testPlusEqualsNoSideEffects() throws Exception {
    thread = newStarlarkThreadWithSkylarkOptions("--incompatible_depset_union=false");
    exec(
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
    thread = newStarlarkThreadWithSkylarkOptions("--incompatible_depset_union=false");
    exec(
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
    thread = newStarlarkThreadWithSkylarkOptions("--incompatible_depset_union=false");
    exec(
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
    thread = newStarlarkThreadWithSkylarkOptions("--incompatible_depset_union=false");
    exec(
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
    thread = newStarlarkThreadWithSkylarkOptions("--incompatible_depset_union=false");
    exec(
        "s = depset() + [2, 4, 6] + [3, 4, 5]", //
        "x = str(s)");
    assertThat(lookup("x")).isEqualTo("depset([2, 4, 6, 3, 5])");
  }

  @Test
  public void testToStringWithOrder() throws Exception {
    thread = newStarlarkThreadWithSkylarkOptions("--incompatible_depset_union=false");
    exec(
        "s = depset(order = 'topological') + [2, 4, 6] + [3, 4, 5]", //
        "x = str(s)");
    assertThat(lookup("x")).isEqualTo("depset([2, 4, 6, 3, 5], order = \"topological\")");
  }

  private Depset get(String varname) throws Exception {
    return (Depset) lookup(varname);
  }

  @Test
  public void testToList() throws Exception {
    thread = newStarlarkThreadWithSkylarkOptions("--incompatible_depset_union=false");
    exec(
        "s = depset() + [2, 4, 6] + [3, 4, 5]", //
        "x = s.to_list()");
    Object value = lookup("x");
    assertThat(value).isInstanceOf(StarlarkList.class);
    assertThat((Iterable<?>) value).containsExactly(2, 4, 6, 3, 5).inOrder();
  }

  @Test
  public void testOrderCompatibility() throws Exception {
    // Two sets are compatible if
    //  (a) both have the same order or
    //  (b) at least one order is "default"

    for (Order first : Order.values()) {
      Depset s1 = Depset.legacyOf(first, Tuple.of("1", "11"));

      for (Order second : Order.values()) {
        Depset s2 = Depset.legacyOf(second, Tuple.of("2", "22"));

        boolean compatible = true;

        try {
          Depset.unionOf(s1, s2);
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
          public Depset merge(Depset[] sets) throws Exception {
            Depset union = Depset.unionOf(sets[0], sets[1]);
            union = Depset.unionOf(union, sets[2]);
            union = Depset.unionOf(union, sets[3]);

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
          public Depset merge(Depset[] sets) throws Exception {
            Depset leftUnion = Depset.unionOf(sets[0], sets[1]);
            Depset rightUnion = Depset.unionOf(sets[2], sets[3]);
            Depset union = Depset.unionOf(leftUnion, rightUnion);

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
          public Depset merge(Depset[] sets) throws Exception {
            Depset union = Depset.unionOf(sets[2], sets[3]);
            union = Depset.unionOf(sets[1], union);
            union = Depset.unionOf(sets[0], union);

            return union;
          }
        };

    runComplexOrderTest(strategy, preOrder, postOrder);
  }

  @Test
  public void testMutableDepsetElementsLegacyBehavior() throws Exception {
    // See b/144992997 and github.com/bazelbuild/bazel/issues/10313.
    thread =
        newStarlarkThreadWithSkylarkOptions("--incompatible_always_check_depset_elements=false");

    // Test legacy depset(...) and new depset(direct=...) constructors.

    // mutable list should be an error
    checkEvalError("depset elements must not be mutable values", "depset([[1,2,3]])");
    checkEvalError("depsets cannot contain items of type 'list'", "depset(direct=[[1,2,3]])");

    // struct containing mutable list should be an error
    checkEvalError("depset elements must not be mutable values", "depset([struct(a=[])])");
    eval("depset(direct=[struct(a=[])])"); // no error (!)

    // tuple of frozen list currently gives no error (this may change)
    update("x", StarlarkList.empty());
    eval("depset([(x,)])");
    eval("depset(direct=[(x,)])");

    // any list (even frozen) is an error, even with legacy constructor
    checkEvalError("depsets cannot contain items of type 'list'", "depset([x])");
    checkEvalError("depsets cannot contain items of type 'list'", "depset(direct=[x])");

    // toplevel dict is an error, even with legacy constructor
    checkEvalError("depset elements must not be mutable values", "depset([{}])");
    checkEvalError("depsets cannot contain items of type 'dict'", "depset(direct=[{}])");

    // struct containing dict should be an error
    checkEvalError("depset elements must not be mutable values", "depset([struct(a={})])");
    eval("depset(direct=[struct(a={})])"); // no error (!)
  }

  @Test
  public void testMutableDepsetElementsDesiredBehavior() throws Exception {
    // See b/144992997 and github.com/bazelbuild/bazel/issues/10313.
    thread =
        newStarlarkThreadWithSkylarkOptions("--incompatible_always_check_depset_elements=true");

    // Test legacy depset(...) and new depset(direct=...) constructors.

    // mutable list should be an error
    checkEvalError("depset elements must not be mutable values", "depset([[1,2,3]])");
    checkEvalError("depset elements must not be mutable values", "depset(direct=[[1,2,3]])");

    // struct containing mutable list should be an error
    checkEvalError("depset elements must not be mutable values", "depset([struct(a=[])])");
    checkEvalError("depset elements must not be mutable values", "depset(direct=[struct(a=[])])");

    // tuple of frozen list currently gives no error (this may change)
    update("x", StarlarkList.empty());
    eval("depset([(x,)])");
    eval("depset(direct=[(x,)])");

    // any list (even frozen) is an error, even with legacy constructor
    checkEvalError("depsets cannot contain items of type 'list'", "depset([x,])");
    checkEvalError("depsets cannot contain items of type 'list'", "depset(direct=[x,])");

    // toplevel dict is an error, even with legacy constructor
    checkEvalError("depset elements must not be mutable values", "depset([{}])");
    checkEvalError("depset elements must not be mutable values", "depset(direct=[{}])");

    // struct containing dict should be an error
    checkEvalError("depset elements must not be mutable values", "depset([struct(a={})])");
    checkEvalError("depset elements must not be mutable values", "depset(direct=[struct(a={})])");
  }

  @Test
  public void testDepthExceedsLimitDuringIteration() throws Exception {
    NestedSet.setApplicationDepthLimit(2000);
    new SkylarkTest()
        .setUp(
            "def create_depset(depth):",
            "  x = depset([0])",
            "  for i in range(1, depth):",
            "    x = depset([i], transitive = [x])",
            "  for element in x.to_list():",
            "    str(x)",
            "  return None")
        .testEval("create_depset(1000)", "None")
        .testIfErrorContains("depset exceeded maximum depth 2000", "create_depset(3000)");
  }

  private interface MergeStrategy {
    Depset merge(Depset[] sets) throws Exception;
  }

  private void runComplexOrderTest(
      MergeStrategy strategy, List<String> preOrder, List<String> postOrder) throws Exception {
    Map<Order, List<String>> expected = createExpectedMap(preOrder, postOrder);
    for (Order order : Order.values()) {
      Depset union = strategy.merge(makeFourSets(order));
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

  private Depset[] makeFourSets(Order order) throws Exception {
    return new Depset[] {
      Depset.legacyOf(order, Tuple.of("1", "11")),
      Depset.legacyOf(order, Tuple.of("2", "22")),
      Depset.legacyOf(order, Tuple.of("3", "33")),
      Depset.legacyOf(order, Tuple.of("4", "44"))
    };
  }
}
