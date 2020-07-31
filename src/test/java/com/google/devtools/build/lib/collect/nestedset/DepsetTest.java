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
package com.google.devtools.build.lib.collect.nestedset;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.collect.nestedset.Depset.ElementType;
import com.google.devtools.build.lib.syntax.Dict;
import com.google.devtools.build.lib.syntax.Sequence;
import com.google.devtools.build.lib.syntax.StarlarkCallable;
import com.google.devtools.build.lib.syntax.StarlarkIterable;
import com.google.devtools.build.lib.syntax.StarlarkList;
import com.google.devtools.build.lib.syntax.StarlarkValue;
import com.google.devtools.build.lib.syntax.Tuple;
import com.google.devtools.build.lib.syntax.util.EvaluationTestCase;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for Depset. */
@RunWith(JUnit4.class)
public final class DepsetTest {

  private final EvaluationTestCase ev = new EvaluationTestCase();

  @Test
  public void testConstructor() throws Exception {
    ev.exec("s = depset(order='default')");
    assertThat(ev.lookup("s")).isInstanceOf(Depset.class);
  }

  private static final ElementType TUPLE = ElementType.of(Tuple.class);

  @Test
  public void testTuples() throws Exception {
    ev.exec(
        "s_one = depset([('1', '2'), ('3', '4')])",
        "s_two = depset(direct = [('1', '2'), ('3', '4'), ('5', '6')])",
        "s_three = depset(transitive = [s_one, s_two])",
        "s_four = depset(direct = [('1', '3')], transitive = [s_one, s_two])",
        "s_five = depset(direct = [('1', '3', '5')], transitive = [s_one, s_two])",
        "s_six = depset(transitive = [s_one, s_five])",
        "s_seven = depset(direct = [('1', '3')], transitive = [s_one, s_five])",
        "s_eight = depset(direct = [(1, 3)], transitive = [s_one, s_two])"); // note, tuple of int
    assertThat(get("s_one").getElementType()).isEqualTo(TUPLE);
    assertThat(get("s_two").getElementType()).isEqualTo(TUPLE);
    assertThat(get("s_three").getElementType()).isEqualTo(TUPLE);
    assertThat(get("s_eight").getElementType()).isEqualTo(TUPLE);

    assertThat(get("s_four").getSet(Tuple.class).toList())
        .containsExactly(
            Tuple.of("1", "3"), Tuple.of("1", "2"), Tuple.of("3", "4"), Tuple.of("5", "6"));
    assertThat(get("s_five").getSet(Tuple.class).toList())
        .containsExactly(
            Tuple.of("1", "3", "5"), Tuple.of("1", "2"), Tuple.of("3", "4"), Tuple.of("5", "6"));
    assertThat(get("s_eight").getSet(Tuple.class).toList())
        .containsExactly(
            Tuple.of(1, 3), Tuple.of("1", "2"), Tuple.of("3", "4"), Tuple.of("5", "6"));
  }

  @Test
  public void testGetSet() throws Exception {
    ev.exec("s = depset(['a', 'b'])");
    assertThat(get("s").getSet(String.class).toList()).containsExactly("a", "b").inOrder();
    assertThat(get("s").getSet(Object.class).toList()).containsExactly("a", "b").inOrder();
    assertThrows(Depset.TypeException.class, () -> get("s").getSet(Integer.class));

    // getSet argument must be a legal Starlark value class, or Object,
    // but not some superclass that doesn't implement StarlarkValue.
    Depset ints = Depset.legacyOf(Order.STABLE_ORDER, Tuple.of(1, 2, 3));
    assertThat(ints.getSet(Integer.class).toString()).isEqualTo("[1, 2, 3]");
    IllegalArgumentException ex =
        assertThrows(IllegalArgumentException.class, () -> ints.getSet(Number.class));
    assertThat(ex.getMessage()).contains("Number is not a subclass of StarlarkValue");
  }

  @Test
  public void testGetSetDirect() throws Exception {
    ev.exec("s = depset(direct = ['a', 'b'])");
    assertThat(get("s").getSet(String.class).toList()).containsExactly("a", "b").inOrder();
    assertThat(get("s").getSet(Object.class).toList()).containsExactly("a", "b").inOrder();
    assertThrows(Depset.TypeException.class, () -> get("s").getSet(Integer.class));
  }

  @Test
  public void testGetSetItems() throws Exception {
    ev.exec("s = depset(items = ['a', 'b'])");
    assertThat(get("s").getSet(String.class).toList()).containsExactly("a", "b").inOrder();
    assertThat(get("s").getSet(Object.class).toList()).containsExactly("a", "b").inOrder();
    assertThrows(Depset.TypeException.class, () -> get("s").getSet(Integer.class));
  }

  @Test
  public void testToList() throws Exception {
    ev.exec("s = depset(['a', 'b'])");
    assertThat(get("s").toList(String.class)).containsExactly("a", "b").inOrder();
    assertThat(get("s").toList(Object.class)).containsExactly("a", "b").inOrder();
    assertThat(get("s").toList()).containsExactly("a", "b").inOrder();
    assertThrows(Depset.TypeException.class, () -> get("s").toList(Integer.class));
  }

  @Test
  public void testToListDirect() throws Exception {
    ev.exec("s = depset(direct = ['a', 'b'])");
    assertThat(get("s").toList(String.class)).containsExactly("a", "b").inOrder();
    assertThat(get("s").toList(Object.class)).containsExactly("a", "b").inOrder();
    assertThat(get("s").toList()).containsExactly("a", "b").inOrder();
    assertThrows(Depset.TypeException.class, () -> get("s").toList(Integer.class));
  }

  @Test
  public void testToListItems() throws Exception {
    ev.exec("s = depset(items = ['a', 'b'])");
    assertThat(get("s").toList(String.class)).containsExactly("a", "b").inOrder();
    assertThat(get("s").toList(Object.class)).containsExactly("a", "b").inOrder();
    assertThat(get("s").toList()).containsExactly("a", "b").inOrder();
    assertThrows(Depset.TypeException.class, () -> get("s").toList(Integer.class));
  }

  @Test
  public void testOrder() throws Exception {
    ev.exec("s = depset(['a', 'b'], order='postorder')");
    assertThat(get("s").getSet(String.class).getOrder()).isEqualTo(Order.COMPILE_ORDER);
  }

  @Test
  public void testOrderDirect() throws Exception {
    ev.exec("s = depset(direct = ['a', 'b'], order='postorder')");
    assertThat(get("s").getSet(String.class).getOrder()).isEqualTo(Order.COMPILE_ORDER);
  }

  @Test
  public void testOrderItems() throws Exception {
    ev.exec("s = depset(items = ['a', 'b'], order='postorder')");
    assertThat(get("s").getSet(String.class).getOrder()).isEqualTo(Order.COMPILE_ORDER);
  }

  @Test
  public void testBadOrder() throws Exception {
    ev.new Scenario()
        .testIfExactError("Invalid order: non_existing", "depset(['a'], order='non_existing')");
  }

  @Test
  public void testBadOrderDirect() throws Exception {
    ev.new Scenario()
        .testIfExactError(
            "Invalid order: non_existing", "depset(direct = ['a'], order='non_existing')");
  }

  @Test
  public void testBadOrderItems() throws Exception {
    ev.new Scenario()
        .testIfExactError(
            "Invalid order: non_existing", "depset(items = ['a'], order='non_existing')");
  }

  @Test
  public void testEmptyGenericType() throws Exception {
    ev.exec("s = depset()");
    assertThat(get("s").getElementType()).isEqualTo(ElementType.EMPTY);
  }

  @Test
  public void testHomogeneousGenericType() throws Exception {
    ev.exec("s = depset(['a', 'b', 'c'])");
    assertThat(get("s").getElementType()).isEqualTo(ElementType.STRING);
  }

  @Test
  public void testHomogeneousGenericTypeDirect() throws Exception {
    ev.exec("s = depset(['a', 'b', 'c'], transitive = [])");
    assertThat(get("s").getElementType()).isEqualTo(ElementType.STRING);
  }

  @Test
  public void testHomogeneousGenericTypeItems() throws Exception {
    ev.exec("s = depset(items = ['a', 'b', 'c'], transitive = [])");
    assertThat(get("s").getElementType()).isEqualTo(ElementType.STRING);
  }

  @Test
  public void testHomogeneousGenericTypeTransitive() throws Exception {
    ev.exec("s = depset(['a', 'b', 'c'], transitive = [depset(['x'])])");
    assertThat(get("s").getElementType()).isEqualTo(ElementType.STRING);
  }

  @Test
  public void testTransitiveIncompatibleOrder() throws Exception {
    ev.checkEvalError(
        "Order 'postorder' is incompatible with order 'topological'",
        "depset(['a', 'b'], order='postorder',",
        "       transitive = [depset(['c', 'd'], order='topological')])");
  }

  @Test
  public void testBadGenericType() throws Exception {
    ev.new Scenario()
        .testIfExactError(
            "cannot add an item of type 'int' to a depset of 'string'", "depset(['a', 5])");
  }

  @Test
  public void testBadGenericTypeDirect() throws Exception {
    ev.new Scenario()
        .testIfExactError(
            "cannot add an item of type 'int' to a depset of 'string'",
            "depset(direct = ['a', 5])");
  }

  @Test
  public void testBadGenericTypeItems() throws Exception {
    ev.new Scenario()
        .testIfExactError(
            "cannot add an item of type 'int' to a depset of 'string'", "depset(items = ['a', 5])");
  }

  @Test
  public void testBadGenericTypeTransitive() throws Exception {
    ev.new Scenario()
        .testIfExactError(
            "cannot add an item of type 'int' to a depset of 'string'",
            "depset(['a', 'b'], transitive=[depset([1])])");
  }

  @Test
  public void testLegacyAndNewApi() throws Exception {
    ev.new Scenario()
        .testIfExactError(
            "Do not pass both 'direct' and 'items' argument to depset constructor.",
            "depset(['a', 'b'], direct = ['c', 'd'])");
  }

  @Test
  public void testItemsAndTransitive() throws Exception {
    ev.new Scenario()
        .testIfExactError(
            "for items, got depset, want sequence",
            "depset(items = depset(), transitive = [depset()])");
  }

  @Test
  public void testTooManyPositionals() throws Exception {
    ev.new Scenario()
        .testIfErrorContains(
            "depset() accepts no more than 2 positional arguments but got 3",
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
    ev.new Scenario()
        .testIfErrorContains("unsupported binary operation: depset + list", "depset([]) + ['a']");

    ev.new Scenario()
        .testIfErrorContains("unsupported binary operation: depset | list", "depset([]) | ['a']");
  }

  private void assertContainsInOrder(String statement, Object... expectedElements)
      throws Exception {
    assertThat(((Depset) ev.eval(statement)).toList()).containsExactly(expectedElements).inOrder();
  }

  @Test
  public void testToString() throws Exception {
    ev.exec("s = depset([3, 4, 5], transitive = [depset([2, 4, 6])])", "x = str(s)");
    assertThat(ev.lookup("x")).isEqualTo("depset([2, 4, 6, 3, 5])");
  }

  @Test
  public void testToStringWithOrder() throws Exception {
    ev.exec(
        "s = depset([3, 4, 5], transitive = [depset([2, 4, 6])], ",
        "           order = 'topological')",
        "x = str(s)");
    assertThat(ev.lookup("x")).isEqualTo("depset([3, 5, 6, 4, 2], order = \"topological\")");
  }

  private Depset get(String varname) throws Exception {
    return (Depset) ev.lookup(varname);
  }

  @Test
  public void testToListForStarlark() throws Exception {
    ev.exec("s = depset([3, 4, 5], transitive = [depset([2, 4, 6])])", "x = s.to_list()");
    Object value = ev.lookup("x");
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
          // merge
          Depset.fromDirectAndTransitive(
              first,
              /*direct=*/ ImmutableList.of(),
              /*transitive=*/ ImmutableList.of(s1, s2),
              /*strict=*/ true);
        } catch (Exception ex) {
          compatible = false;
        }

        assertThat(compatible).isEqualTo(areOrdersCompatible(first, second));
      }
    }
  }

  private static boolean areOrdersCompatible(Order first, Order second) {
    return first == Order.STABLE_ORDER || second == Order.STABLE_ORDER || first == second;
  }

  @Test
  public void testMutableDepsetElementsLegacyBehavior() throws Exception {
    // See b/144992997 and github.com/bazelbuild/bazel/issues/10313.
    ev.setSemantics("--incompatible_always_check_depset_elements=false");

    // Test legacy depset(...) and new depset(direct=...) constructors.

    // mutable list should be an error
    ev.checkEvalError("depset elements must not be mutable values", "depset([[1,2,3]])");
    ev.checkEvalError("depsets cannot contain items of type 'list'", "depset(direct=[[1,2,3]])");

    // struct containing mutable list should be an error
    ev.checkEvalError("depset elements must not be mutable values", "depset([struct(a=[])])");
    ev.eval("depset(direct=[struct(a=[])])"); // no error (!)

    // tuple of frozen list currently gives no error (this may change)
    ev.update("x", StarlarkList.empty());
    ev.eval("depset([(x,)])");
    ev.eval("depset(direct=[(x,)])");

    // any list (even frozen) is an error, even with legacy constructor
    ev.checkEvalError("depsets cannot contain items of type 'list'", "depset([x])");
    ev.checkEvalError("depsets cannot contain items of type 'list'", "depset(direct=[x])");

    // toplevel dict is an error, even with legacy constructor
    ev.checkEvalError("depset elements must not be mutable values", "depset([{}])");
    ev.checkEvalError("depsets cannot contain items of type 'dict'", "depset(direct=[{}])");

    // struct containing dict should be an error
    ev.checkEvalError("depset elements must not be mutable values", "depset([struct(a={})])");
    ev.eval("depset(direct=[struct(a={})])"); // no error (!)
  }

  @Test
  public void testMutableDepsetElementsDesiredBehavior() throws Exception {
    // See b/144992997 and github.com/bazelbuild/bazel/issues/10313.
    ev.setSemantics("--incompatible_always_check_depset_elements=true");

    // Test legacy depset(...) and new depset(direct=...) constructors.

    // mutable list should be an error
    ev.checkEvalError("depset elements must not be mutable values", "depset([[1,2,3]])");
    ev.checkEvalError("depset elements must not be mutable values", "depset(direct=[[1,2,3]])");

    // struct containing mutable list should be an error
    ev.checkEvalError("depset elements must not be mutable values", "depset([struct(a=[])])");
    ev.checkEvalError(
        "depset elements must not be mutable values", "depset(direct=[struct(a=[])])");

    // tuple of frozen list currently gives no error (this may change)
    ev.update("x", StarlarkList.empty());
    ev.eval("depset([(x,)])");
    ev.eval("depset(direct=[(x,)])");

    // any list (even frozen) is an error, even with legacy constructor
    ev.checkEvalError("depsets cannot contain items of type 'list'", "depset([x,])");
    ev.checkEvalError("depsets cannot contain items of type 'list'", "depset(direct=[x,])");

    // toplevel dict is an error, even with legacy constructor
    ev.checkEvalError("depset elements must not be mutable values", "depset([{}])");
    ev.checkEvalError("depset elements must not be mutable values", "depset(direct=[{}])");

    // struct containing dict should be an error
    ev.checkEvalError("depset elements must not be mutable values", "depset([struct(a={})])");
    ev.checkEvalError(
        "depset elements must not be mutable values", "depset(direct=[struct(a={})])");
  }

  @Test
  public void testConstructorDepthLimit() throws Exception {
    ev.new Scenario()
        .setUp(
            "def create_depset(depth):",
            "  x = depset([0])",
            "  for i in range(1, depth):",
            "    x = depset([i], transitive = [x])")
        .testEval("create_depset(3000)", "None") // succeeds
        .testIfErrorContains("depset depth 3501 exceeds limit (3500)", "create_depset(4000)");

    Depset.setDepthLimit(100);
    ev.new Scenario()
        .setUp(
            "def create_depset(depth):",
            "  x = depset([0])",
            "  for i in range(1, depth):",
            "    x = depset([i], transitive = [x])")
        .testEval("create_depset(99)", "None") // succeeds
        .testIfErrorContains("depset depth 101 exceeds limit (100)", "create_depset(1000)");
  }

  @Test
  public void testElementTypeOf() {
    // legal values
    assertThat(ElementType.of(String.class).toString()).isEqualTo("string");
    assertThat(ElementType.of(Integer.class).toString()).isEqualTo("int");
    assertThat(ElementType.of(Boolean.class).toString()).isEqualTo("bool");

    // concrete non-values
    assertThrows(IllegalArgumentException.class, () -> ElementType.of(Float.class));

    // concrete classes that implement StarlarkValue
    assertThat(ElementType.of(StarlarkList.class).toString()).isEqualTo("list");
    assertThat(ElementType.of(Tuple.class).toString()).isEqualTo("tuple");
    assertThat(ElementType.of(Dict.class).toString()).isEqualTo("dict");
    class V implements StarlarkValue {} // no StarlarkModule annotation
    assertThat(ElementType.of(V.class).toString()).isEqualTo("V");

    // abstract classes that implement StarlarkValue
    assertThat(ElementType.of(Sequence.class).toString()).isEqualTo("sequence");
    assertThat(ElementType.of(StarlarkCallable.class).toString()).isEqualTo("function");
    assertThat(ElementType.of(StarlarkIterable.class).toString()).isEqualTo("StarlarkIterable");

    // superclasses of legal values that aren't values themselves
    assertThrows(IllegalArgumentException.class, () -> ElementType.of(Number.class));
    assertThrows(IllegalArgumentException.class, () -> ElementType.of(CharSequence.class));
    assertThrows(IllegalArgumentException.class, () -> ElementType.of(Object.class));
  }

  @Test
  public void testSetComparison() throws Exception {
    ev.new Scenario()
        .testIfExactError("Cannot compare depset with depset", "depset([1, 2]) < depset([3, 4])");
  }

  @Test
  public void testDepsetItemsKeywordAndPositional() throws Exception {
    ev.new Scenario("--incompatible_disable_depset_items=false")
        .testIfErrorContains(
            "parameter 'items' cannot be specified both positionally and by keyword",
            "depset([0, 1], 'default', items=[0,1])");
  }

  @Test
  public void testDepsetDirectInvalidType() throws Exception {
    ev.new Scenario()
        .testIfErrorContains("for direct, got string, want sequence", "depset(direct='hello')");
  }

  @Test
  public void testDisableDepsetItems() throws Exception {
    ev.new Scenario("--incompatible_disable_depset_items")
        .setUp("x = depset([0])", "y = depset(direct = [1])")
        .testEval("depset([2, 3], transitive = [x, y]).to_list()", "[0, 1, 2, 3]")
        .testIfErrorContains(
            "parameter 'direct' cannot be specified both positionally and by keyword",
            "depset([0, 1], 'default', direct=[0,1])")
        .testIfErrorContains(
            "in call to depset(), parameter 'items' is deprecated and will be removed soon. "
                + "It may be temporarily re-enabled by setting "
                + "--incompatible_disable_depset_items=false",
            "depset(items=[0,1])");
  }
}
