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

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Test of evaluation behavior.  (Implicitly uses lexer + parser.)
 */
@RunWith(JUnit4.class)
public class EvaluationTest extends EvaluationTestCase {

  @Override
  public EvaluationContext newEvaluationContext() {
    return EvaluationContext.newBuildContext(getEventHandler(),
        new PackageFactory(TestRuleClassProvider.getRuleClassProvider()).getEnvironment());
  }

  @Test
  public void testExprs() throws Exception {
    assertEquals("fooxbar", eval("'%sx' % 'foo' + 'bar'"));
    assertEquals("fooxbar", eval("('%sx' % 'foo') + 'bar'"));
    assertEquals("foobarx", eval("'%sx' % ('foo' + 'bar')"));
    assertEquals(579, eval("123 + 456"));
    assertEquals(333, eval("456 - 123"));
    assertEquals(2, eval("8 % 3"));

    checkEvalErrorContains("unsupported operand type(s) for %: 'int' and 'string'", "3 % 'foo'");
  }

  @SuppressWarnings("unchecked")
  @Test
  public void testListExprs() throws Exception {
    assertThat((Iterable<Object>) eval("[1, 2, 3]")).containsExactly(1, 2, 3).inOrder();
    assertThat((Iterable<Object>) eval("(1, 2, 3)")).containsExactly(1, 2, 3).inOrder();
  }

  @Test
  public void testStringFormatMultipleArgs() throws Exception {
    assertEquals("XYZ", eval("'%sY%s' % ('X', 'Z')"));
  }

  @Test
  public void testAndOr() throws Exception {
    assertEquals(8, eval("8 or 9"));
    assertEquals(9, eval("0 or 9"));
    assertEquals(9, eval("8 and 9"));
    assertEquals(0, eval("0 and 9"));

    assertEquals(2, eval("1 and 2 or 3"));
    assertEquals(3, eval("0 and 2 or 3"));
    assertEquals(3, eval("1 and 0 or 3"));

    assertEquals(1, eval("1 or 2 and 3"));
    assertEquals(3, eval("0 or 2 and 3"));
    assertEquals(0, eval("0 or 0 and 3"));
    assertEquals(1, eval("1 or 0 and 3"));
    assertEquals(1, eval("1 or 0 and 3"));

    assertEquals(Environment.NONE, eval("None and 1"));
    assertEquals(9, eval("\"\" or 9"));
    assertEquals("abc", eval("\"abc\" or 9"));

    if (isSkylark()) {
      checkEvalError("ERROR 1:6: name 'foo' is not defined", "8 or foo");
      checkEvalError("ERROR 1:7: name 'foo' is not defined", "0 and foo");
    } else {
      assertEquals(8, eval("8 or foo")); // check that 'foo' is not evaluated
      assertEquals(0, eval("0 and foo")); // check that 'foo' is not evaluated
    }
  }

  @Test
  public void testNot() throws Exception {
    assertEquals(false, eval("not 1"));
    assertEquals(true, eval("not ''"));
  }

  @Test
  public void testNotWithLogicOperators() throws Exception {
    assertEquals(true, eval("not (0 and 0)"));
    assertEquals(false, eval("not (1 or 0)"));

    assertEquals(0, eval("0 and not 0"));
    assertEquals(0, eval("not 0 and 0"));

    assertEquals(true, eval("1 and not 0"));
    assertEquals(true, eval("not 0 or 0"));

    assertEquals(0, eval("not 1 or 0"));
    assertEquals(1, eval("not 1 or 1"));
  }

  @Test
  public void testNotWithArithmeticOperators() throws Exception {
    assertEquals(true, eval("not 0 + 0"));
    assertEquals(false, eval("not 2 - 1"));
  }

  @Test
  public void testNotWithCollections() throws Exception {
    assertEquals(true, eval("not []"));
    assertEquals(false, eval("not {'a' : 1}"));
  }

  @Test
  public void testEquality() throws Exception {
    assertEquals(true, eval("1 == 1"));
    assertEquals(false, eval("1 == 2"));
    assertEquals(true, eval("'hello' == 'hel' + 'lo'"));
    assertEquals(false, eval("'hello' == 'bye'"));
    assertEquals(true, eval("None == None"));
    assertEquals(true, eval("[1, 2] == [1, 2]"));
    assertEquals(false, eval("[1, 2] == [2, 1]"));
  }

  @Test
  public void testInequality() throws Exception {
    assertEquals(false, eval("1 != 1"));
    assertEquals(true, eval("1 != 2"));
    assertEquals(false, eval("'hello' != 'hel' + 'lo'"));
    assertEquals(true, eval("'hello' != 'bye'"));
    assertEquals(false, eval("[1, 2] != [1, 2]"));
    assertEquals(true, eval("[1, 2] != [2, 1]"));
  }

  @Test
  public void testEqualityPrecedence() throws Exception {
    assertEquals(true, eval("1 + 3 == 2 + 2"));
    assertEquals(true, eval("not 1 == 2"));
    assertEquals(false, eval("not 1 != 2"));
    assertEquals(true, eval("2 and 3 == 3 or 1"));
    assertEquals(2, eval("2 or 3 == 3 and 1"));
  }

  @Test
  public void testLessThan() throws Exception {
    assertEquals(true, eval("1 <= 1"));
    assertEquals(false, eval("1 < 1"));
    assertEquals(true, eval("'a' <= 'b'"));
    assertEquals(false, eval("'c' < 'a'"));
  }

  @Test
  public void testGreaterThan() throws Exception {
    assertEquals(true, eval("1 >= 1"));
    assertEquals(false, eval("1 > 1"));
    assertEquals(false, eval("'a' >= 'b'"));
    assertEquals(true, eval("'c' > 'a'"));
  }

  @Test
  public void testConditionalExpressions() throws Exception {
    assertEquals(1, eval("1 if True else 2"));
    assertEquals(2, eval("1 if False else 2"));
    assertEquals(3, eval("1 + 2 if 3 + 4 else 5 + 6"));

    setFailFast(false);
    parseExpression("1 if 2");
    assertContainsEvent(
        "missing else clause in conditional expression or semicolon before if");
  }

  @Test
  public void testListComparison() throws Exception {
    assertThat(eval("[] < [1]")).isEqualTo(true);
    assertThat(eval("[1] < [1, 1]")).isEqualTo(true);
    assertThat(eval("[1, 1] < [1, 2]")).isEqualTo(true);
    assertThat(eval("[1, 2] < [1, 2, 3]")).isEqualTo(true);
    assertThat(eval("[1, 2, 3] <= [1, 2, 3]")).isEqualTo(true);

    assertThat(eval("['a', 'b'] > ['a']")).isEqualTo(true);
    assertThat(eval("['a', 'b'] >= ['a']")).isEqualTo(true);
    assertThat(eval("['a', 'b'] < ['a']")).isEqualTo(false);
    assertThat(eval("['a', 'b'] <= ['a']")).isEqualTo(false);

    assertThat(eval("('a', 'b') > ('a', 'b')")).isEqualTo(false);
    assertThat(eval("('a', 'b') >= ('a', 'b')")).isEqualTo(true);
    assertThat(eval("('a', 'b') < ('a', 'b')")).isEqualTo(false);
    assertThat(eval("('a', 'b') <= ('a', 'b')")).isEqualTo(true);

    assertThat(eval("[[1, 1]] > [[1, 1], []]")).isEqualTo(false);
    assertThat(eval("[[1, 1]] < [[1, 1], []]")).isEqualTo(true);

    checkEvalError("Cannot compare int with string", "[1] < ['a']");
    checkEvalError("Cannot compare list with int", "[1] < 1");
  }

  @Test
  public void testCompareStringInt() throws Exception {
    checkEvalError("Cannot compare string with int", "'a' >= 1");
  }

  @Test
  public void testSumFunction() throws Exception {
    BaseFunction sum = new BaseFunction("sum") {
      @Override
      public Object call(List<Object> args, Map<String, Object> kwargs,
          FuncallExpression ast, Environment env) {
        int sum = 0;
        for (Object arg : args) {
          sum += (Integer) arg;
        }
        return sum;
      }
    };

    update(sum.getName(), sum);
    assertEquals(21, eval("sum(1, 2, 3, 4, 5, 6)"));
    assertEquals(sum, eval("sum"));
    assertEquals(0, eval("sum(a=1, b=2)"));
  }

  @Test
  public void testNotCallInt() throws Exception {
    eval("sum = 123456");
    assertEquals(123456, lookup("sum"));
    checkEvalError("'int' object is not callable", "sum(1, 2, 3, 4, 5, 6)");
    assertEquals(123456, eval("sum"));
  }

  @Test
  public void testKeywordArgs() throws Exception {

    // This function returns the map of keyword arguments passed to it.
    BaseFunction kwargs = new BaseFunction("kwargs") {
      @Override
      public Object call(List<Object> args,
          final Map<String, Object> kwargs,
          FuncallExpression ast,
          Environment env) {
        return kwargs;
      }
    };

    update(kwargs.getName(), kwargs);

    assertEquals(eval("[('bar', 'bar'), ('foo', 1), ('wiz', [1, 2, 3])]"),
        eval("kwargs(foo=1, bar='bar', wiz=[1,2,3]).items()"));
  }

  @Test
  public void testModulo() throws Exception {
    assertThat(eval("6 % 2")).isEqualTo(0);
    assertThat(eval("6 % 4")).isEqualTo(2);
    assertThat(eval("3 % 6")).isEqualTo(3);
    assertThat(eval("7 % -4")).isEqualTo(-1);
    assertThat(eval("-7 % 4")).isEqualTo(1);
    assertThat(eval("-7 % -4")).isEqualTo(-3);
    checkEvalError("integer modulo by zero", "5 % 0");
  }

  @Test
  public void testMult() throws Exception {
    assertEquals(42, eval("6 * 7"));

    assertEquals("ababab", eval("3 * 'ab'"));
    assertEquals("", eval("0 * 'ab'"));
    assertEquals("100000", eval("'1' + '0' * 5"));
  }

  @Test
  public void testDivision() throws Exception {
    assertThat(eval("6 / 2")).isEqualTo(3);
    assertThat(eval("6 / 4")).isEqualTo(1);
    assertThat(eval("3 / 6")).isEqualTo(0);
    assertThat(eval("7 / -2")).isEqualTo(-4);
    assertThat(eval("-7 / 2")).isEqualTo(-4);
    assertThat(eval("-7 / -2")).isEqualTo(3);
    assertThat(eval("2147483647 / 2")).isEqualTo(1073741823);
    checkEvalError("integer division by zero", "5 / 0");
  }

  @Test
  public void testOperatorPrecedence() throws Exception {
    assertThat(eval("2 + 3 * 4")).isEqualTo(14);
    assertThat(eval("2 + 3 / 4")).isEqualTo(2);
    assertThat(eval("2 * 3 + 4 / -2")).isEqualTo(4);
  }

  @Test
  public void testConcatStrings() throws Exception {
    assertEquals("foobar", eval("'foo' + 'bar'"));
  }

  @SuppressWarnings("unchecked")
  @Test
  public void testConcatLists() throws Exception {
    // list
    Object x = eval("[1,2] + [3,4]");
    assertThat((Iterable<Object>) x).containsExactly(1, 2, 3, 4).inOrder();
    assertEquals(Arrays.asList(1, 2, 3, 4), x);
    assertFalse(EvalUtils.isImmutable(x));

    // tuple
    x = eval("(1,2) + (3,4)");
    assertEquals(Arrays.asList(1, 2, 3, 4), x);
    assertTrue(EvalUtils.isImmutable(x));

    checkEvalError("can only concatenate List (not \"Tuple\") to List",
        "(1,2) + [3,4]"); // list + tuple
  }

  @SuppressWarnings("unchecked")
  @Test
  public void testListComprehensions() throws Exception {
    assertThat((Iterable<Object>) eval("['foo/%s.java' % x for x in []]")).isEmpty();

    assertThat((Iterable<Object>) eval("['foo/%s.java' % y for y in ['bar', 'wiz', 'quux']]"))
        .containsExactly("foo/bar.java", "foo/wiz.java", "foo/quux.java").inOrder();

    assertThat((Iterable<Object>) eval(
        "['%s/%s.java' % (z, t) "
        + "for z in ['foo', 'bar'] "
        + "for t in ['baz', 'wiz', 'quux']]"))
        .containsExactly("foo/baz.java", "foo/wiz.java", "foo/quux.java",
            "bar/baz.java", "bar/wiz.java", "bar/quux.java").inOrder();

    assertThat((Iterable<Object>) eval(
        "['%s/%s.java' % (b, b) "
        + "for a in ['foo', 'bar'] "
        + "for b in ['baz', 'wiz', 'quux']]"))
        .containsExactly("baz/baz.java", "wiz/wiz.java", "quux/quux.java",
            "baz/baz.java", "wiz/wiz.java", "quux/quux.java").inOrder();

    assertThat((Iterable<Object>) eval(
        "['%s/%s.%s' % (c, d, e) "
        + "for c in ['foo', 'bar'] "
        + "for d in ['baz', 'wiz', 'quux'] "
        + "for e in ['java', 'cc']]"))
        .containsExactly("foo/baz.java", "foo/baz.cc", "foo/wiz.java", "foo/wiz.cc",
            "foo/quux.java", "foo/quux.cc", "bar/baz.java", "bar/baz.cc",
            "bar/wiz.java", "bar/wiz.cc", "bar/quux.java", "bar/quux.cc").inOrder();
  }

  @Test
  public void testNestedListComprehensions() throws Exception {
    assertThat((Iterable<?>) eval(
          "li = [[1, 2], [3, 4]]\n"
          + "[j for i in li for j in i]"))
        .containsExactly(1, 2, 3, 4).inOrder();

    assertThat((Iterable<?>) eval(
          "input = [['abc'], ['def', 'ghi']]\n"
          + "['%s %s' % (b, c) for a in input for b in a for c in b]"))
        .containsExactly(
            "abc a", "abc b", "abc c", "def d", "def e", "def f", "ghi g", "ghi h", "ghi i")
        .inOrder();
  }

  @Test
  public void testListComprehensionsMultipleVariables() throws Exception {
    assertThat(eval("[x + y for x, y in [(1, 2), (3, 4)]]").toString())
        .isEqualTo("[3, 7]");
    assertThat(eval("[z + t for (z, t) in [[1, 2], [3, 4]]]").toString())
        .isEqualTo("[3, 7]");
  }

  @Test
  public void testListComprehensionsMultipleVariablesFail() throws Exception {
    checkEvalError("lvalue has length 3, but rvalue has has length 2",
        "[x + y for x, y, z in [(1, 2), (3, 4)]]");

    checkEvalError("type 'int' is not a collection",
        "[x + y for x, y in (1, 2)]");
  }

  @Test
  public void testTupleDestructuring() throws Exception {
    eval("a, b = 1, 2");
    assertThat(lookup("a")).isEqualTo(1);
    assertThat(lookup("b")).isEqualTo(2);

    eval("c, d = {'key1':2, 'key2':3}");
    assertThat(lookup("c")).isEqualTo("key1");
    assertThat(lookup("d")).isEqualTo("key2");
  }

  @Test
  public void testHeterogeneousDict() throws Exception {
    eval("d = {'str': 1, 2: 3}\n"
         + "a = d['str']\n"
         + "b = d[2]");
    assertThat(lookup("a")).isEqualTo(1);
    assertThat(lookup("b")).isEqualTo(3);
  }

  @Test
  public void testRecursiveTupleDestructuring() throws Exception {
    eval("((a, b), (c, d)) = [(1, 2), (3, 4)]");
    assertThat(lookup("a")).isEqualTo(1);
    assertThat(lookup("b")).isEqualTo(2);
    assertThat(lookup("c")).isEqualTo(3);
    assertThat(lookup("d")).isEqualTo(4);
  }

  @SuppressWarnings("unchecked")
  @Test
  public void testListComprehensionModifiesGlobalEnv() throws Exception {
    update("x", 42);
    if (isSkylark()) {
      checkEvalError("ERROR 1:1: Variable x is read only", "[x + 1 for x in [1,2,3]]");
    } else {
      assertThat((Iterable<Object>) eval("[x + 1 for x in [1,2,3]]"))
          .containsExactly(2, 3, 4).inOrder();
      assertEquals(3, lookup("x")); // (x is global)
    }
  }

  @Test
  public void testDictComprehensions() throws Exception {
    assertEquals(Collections.emptyMap(), eval("{a : a for a in []}"));
    assertEquals(ImmutableMap.of(1, 1, 2, 2), eval("{b : b for b in [1, 2]}"));
    assertEquals(ImmutableMap.of("a", "v_a", "b", "v_b"),
        eval("{c : 'v_' + c for c in ['a', 'b']}"));
    assertEquals(ImmutableMap.of("k_a", "a", "k_b", "b"),
        eval("{'k_' + d : d for d in ['a', 'b']}"));
    assertEquals(ImmutableMap.of("k_a", "v_a", "k_b", "v_b"),
        eval("{'k_' + e : 'v_' + e for e in ['a', 'b']}"));
    assertEquals(ImmutableMap.of(5, 6), eval("{x+y : x*y for x, y in [[2, 3]]}"));
  }

  @Test
  public void testDictComprehensionOnNonIterable() throws Exception {
    checkEvalError(
        "type 'int' is not iterable",
        "{k : k for k in 3}");
  }

  @Test
  public void testDictComprehensions_MultipleKey() throws Exception {
    assertEquals(ImmutableMap.of(1, 1, 2, 2), eval("{x : x for x in [1, 2, 1]}"));
    assertEquals(ImmutableMap.of("ab", "ab", "c", "c"),
        eval("{y : y for y in ['ab', 'c', 'a' + 'b']}"));
  }

  @Test
  public void testDictComprehensions_ToString() throws Exception {
    assertEquals("{x: x for x in [1, 2]}",
        evaluationContext.parseExpression("{x : x for x in [1, 2]}").toString());
    assertEquals("{x + 'a': x for x in [1, 2]}",
        evaluationContext.parseExpression("{x + 'a' : x for x in [1, 2]}").toString());
  }

  @Test
  public void testListConcatenation() throws Exception {
    assertEquals(Arrays.asList(1, 2, 3, 4), eval("[1, 2] + [3, 4]"));
    assertEquals(ImmutableList.of(1, 2, 3, 4), eval("(1, 2) + (3, 4)"));
    checkEvalError("can only concatenate Tuple (not \"List\") to Tuple",
        "[1, 2] + (3, 4)");
    checkEvalError("can only concatenate List (not \"Tuple\") to List",
        "(1, 2) + [3, 4]");
  }

  @SuppressWarnings("unchecked")
  @Test
  public void testSelectorListConcatenation() throws Exception {
    SelectorList x = (SelectorList) eval("select({'foo': ['FOO'], 'bar': ['BAR']}) + []");
    List<Object> elements = x.getElements();
    assertThat(elements.size()).isEqualTo(2);
    assertThat(elements.get(0)).isInstanceOf(SelectorValue.class);
    assertThat((Iterable) elements.get(1)).isEmpty();
  }

  @Test
  public void testListComprehensionFailsOnNonSequence() throws Exception {
    checkEvalErrorContains("type 'int' is not iterable", "[x + 1 for x in 123]");
  }

  @SuppressWarnings("unchecked")
  @Test
  public void testListComprehensionOnString() throws Exception {
    assertThat((Iterable<Object>) eval("[x for x in 'abc']"))
        .containsExactly("a", "b", "c").inOrder();
  }

  @Test
  public void testInvalidAssignment() throws Exception {
    update("x", 1);
    checkEvalErrorContains("can only assign to variables and tuples, not to 'x + 1'",
        "x + 1 = 2");
  }

  @SuppressWarnings("unchecked")
  @Test
  public void testListComprehensionOnDictionary() throws Exception {
    assertThat((Iterable<Object>) eval("val = ['var_' + n for n in {'a':1,'b':2}] ; val"))
        .containsExactly("var_a", "var_b").inOrder();
  }

  @Test
  public void testListComprehensionOnDictionaryCompositeExpression() throws Exception {
    eval("d = {1:'a',2:'b'}\n" + "l = [d[x] for x in d]");
    assertEquals("[\"a\", \"b\"]", EvalUtils.prettyPrintValue(lookup("l")));
  }

  @Test
  public void testInOperator() throws Exception {
    assertEquals(Boolean.TRUE, eval("'b' in ['a', 'b']"));
    assertEquals(Boolean.FALSE, eval("'c' in ['a', 'b']"));
    assertEquals(Boolean.TRUE, eval("'b' in ('a', 'b')"));
    assertEquals(Boolean.FALSE, eval("'c' in ('a', 'b')"));
    assertEquals(Boolean.TRUE, eval("'b' in {'a' : 1, 'b' : 2}"));
    assertEquals(Boolean.FALSE, eval("'c' in {'a' : 1, 'b' : 2}"));
    assertEquals(Boolean.FALSE, eval("1 in {'a' : 1, 'b' : 2}"));
    assertEquals(Boolean.TRUE, eval("'b' in 'abc'"));
    assertEquals(Boolean.FALSE, eval("'d' in 'abc'"));
  }

  @Test
  public void testNotInOperator() throws Exception {
    assertEquals(Boolean.FALSE, eval("'b' not in ['a', 'b']"));
    assertEquals(Boolean.TRUE, eval("'c' not in ['a', 'b']"));
    assertEquals(Boolean.FALSE, eval("'b' not in ('a', 'b')"));
    assertEquals(Boolean.TRUE, eval("'c' not in ('a', 'b')"));
    assertEquals(Boolean.FALSE, eval("'b' not in {'a' : 1, 'b' : 2}"));
    assertEquals(Boolean.TRUE, eval("'c' not in {'a' : 1, 'b' : 2}"));
    assertEquals(Boolean.TRUE, eval("1 not in {'a' : 1, 'b' : 2}"));
    assertEquals(Boolean.FALSE, eval("'b' not in 'abc'"));
    assertEquals(Boolean.TRUE, eval("'d' not in 'abc'"));
  }

  @Test
  public void testInFail() throws Exception {
    checkEvalError("in operator only works on strings if the left operand is also a string",
        "1 in '123'");
    checkEvalError("in operator only works on lists, tuples, sets, dicts and strings", "'a' in 1");
  }

  @Test
  public void testInCompositeForPrecedence() throws Exception {
    assertEquals(0, eval("not 'a' in ['a'] or 0"));
  }

  private Object createObjWithStr() {
    return new Object() {
      @Override
      public String toString() {
        return "str marker";
      }
    };
  }

  @Test
  public void testPercOnObject() throws Exception {
    update("obj", createObjWithStr());
    assertEquals("str marker", eval("'%s' % obj"));
  }

  @Test
  public void testPercOnObjectList() throws Exception {
    update("obj", createObjWithStr());
    assertEquals("str marker str marker", eval("'%s %s' % (obj, obj)"));
  }

  @Test
  public void testPercOnObjectInvalidFormat() throws Exception {
    update("obj", createObjWithStr());
    checkEvalError("invalid arguments for format string", "'%d' % obj");
  }

  @SuppressWarnings("unchecked")
  @Test
  public void testDictKeys() throws Exception {
    assertThat((Iterable<Object>) eval("v = {'a': 1}.keys() + ['b', 'c'] ; v"))
        .containsExactly("a", "b", "c").inOrder();
  }

  @Test
  public void testDictKeysTooManyArgs() throws Exception {
    checkEvalError("too many (2) positional arguments in call to keys(self: dict)",
        "{'a': 1}.keys('abc')");
  }

  @Test
  public void testDictKeysTooManyKeyArgs() throws Exception {
    checkEvalError("unexpected keyword 'arg' in call to keys(self: dict)",
        "{'a': 1}.keys(arg='abc')");
  }

  @Test
  public void testDictKeysDuplicateKeyArgs() throws Exception {
    checkEvalError("duplicate keywords 'arg', 'k' in call to keys",
        "{'a': 1}.keys(arg='abc', arg='def', k=1, k=2)");
  }

  @Test
  public void testArgBothPosKey() throws Exception {
    checkEvalErrorStartsWith("arguments 'old', 'new' passed both by position and by name "
        + "in call to replace(self: string, ",
        "'banana'.replace('a', 'o', 3, old='a', new=4)");
  }
}
