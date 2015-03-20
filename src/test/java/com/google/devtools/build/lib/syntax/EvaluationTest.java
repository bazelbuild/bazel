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
import static org.junit.Assert.fail;

import com.google.common.base.Functions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Ordering;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Test of evaluation behavior.  (Implicitly uses lexer + parser.)
 */
@RunWith(JUnit4.class)
public class EvaluationTest extends AbstractEvaluationTestCase {

  protected Environment env;

  @Before
  public void setUp() throws Exception {

    PackageFactory factory = new PackageFactory(TestRuleClassProvider.getRuleClassProvider());
    env = factory.getEnvironment();
  }

  public Environment singletonEnv(String id, Object value) {
    Environment env = new Environment();
    env.update(id, value);
    return env;
  }

  @Override
  public Object eval(String input) throws Exception {
    return eval(parseExpr(input), env);
  }

  @Test
  public void testExprs() throws Exception {
    assertEquals("fooxbar",
                 eval("'%sx' % 'foo' + 'bar'"));
    assertEquals("fooxbar",
                 eval("('%sx' % 'foo') + 'bar'"));
    assertEquals("foobarx",
                 eval("'%sx' % ('foo' + 'bar')"));
    assertEquals(579,
                 eval("123 + 456"));
    assertEquals(333,
                 eval("456 - 123"));
    assertEquals(2,
                 eval("8 % 3"));

    checkEvalError("3 % 'foo'", "unsupported operand type(s) for %: 'int' and 'string'");
  }

  @Test
  public void testListExprs() throws Exception {
    assertEquals(Arrays.asList(1, 2, 3),
        eval("[1, 2, 3]"));
    assertEquals(Arrays.asList(1, 2, 3),
        eval("(1, 2, 3)"));
  }

  @Test
  public void testStringFormatMultipleArgs() throws Exception {
    assertEquals("XYZ", eval("'%sY%s' % ('X', 'Z')"));
  }

  @Test
  public void testAndOr() throws Exception {
    assertEquals(8, eval("8 or 9"));
    assertEquals(8, eval("8 or foo")); // check that 'foo' is not evaluated
    assertEquals(9, eval("0 or 9"));
    assertEquals(9, eval("8 and 9"));
    assertEquals(0, eval("0 and 9"));
    assertEquals(0, eval("0 and foo")); // check that 'foo' is not evaluated

    assertEquals(2, eval("1 and 2 or 3"));
    assertEquals(3, eval("0 and 2 or 3"));
    assertEquals(3, eval("1 and 0 or 3"));

    assertEquals(1, eval("1 or 2 and 3"));
    assertEquals(3, eval("0 or 2 and 3"));
    assertEquals(0, eval("0 or 0 and 3"));
    assertEquals(1, eval("1 or 0 and 3"));
    assertEquals(1, eval("1 or 0 and 3"));

    assertEquals(9, eval("\"\" or 9"));
    assertEquals("abc", eval("\"abc\" or 9"));
    assertEquals(Environment.NONE, eval("None and 1"));
  }

  @Test
  public void testNot() throws Exception {
    assertEquals(false, eval("not 1"));
    assertEquals(true, eval("not ''"));
  }

  @Test
  public void testNotWithLogicOperators() throws Exception {
    assertEquals(0, eval("0 and not 0"));
    assertEquals(0, eval("not 0 and 0"));

    assertEquals(true, eval("1 and not 0"));
    assertEquals(true, eval("not 0 or 0"));

    assertEquals(0, eval("not 1 or 0"));
    assertEquals(1, eval("not 1 or 1"));

    assertEquals(true, eval("not (0 and 0)"));
    assertEquals(false, eval("not (1 or 0)"));
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
    assertEquals(true, eval("[1, 2] == [1, 2]"));
    assertEquals(false, eval("[1, 2] == [2, 1]"));
    assertEquals(true, eval("None == None"));
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

    syntaxEvents.setFailFast(false);
    parseExpr("1 if 2");
    syntaxEvents.assertContainsEvent(
        "missing else clause in conditional expression or semicolon before if");
    syntaxEvents.collector().clear();
  }

  @Test
  public void testCompareStringInt() throws Exception {
    checkEvalError("'a' >= 1", "Cannot compare string with int");
  }

  @Test
  public void testNotComparable() throws Exception {
    checkEvalError("[1, 2] < [1, 3]", "[1, 2] is not comparable");
  }

  @Test
  public void testSumFunction() throws Exception {
    Function sum = new AbstractFunction("sum") {
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

    Environment env = singletonEnv(sum.getName(), sum);

    String callExpr = "sum(1, 2, 3, 4, 5, 6)";
    assertEquals(21, eval(callExpr, env));

    assertEquals(sum, eval("sum", env));

    assertEquals(0, eval("sum(a=1, b=2)", env));

    // rebind 'sum' in a new environment:
    env = new Environment();
    exec(parseStmt("sum = 123456"), env);

    assertEquals(123456, env.lookup("sum"));

    // now we can't call it any more:
    checkEvalError(callExpr, env, "'int' object is not callable");

    assertEquals(123456, eval("sum", env));
  }

  @Test
  public void testKeywordArgs() throws Exception {

    // This function returns the list of keyword-argument keys or values,
    // depending on whether its first (integer) parameter is zero.
    Function keyval = new AbstractFunction("keyval") {
        @Override
        public Object call(List<Object> args,
                           final Map<String, Object> kwargs,
                           FuncallExpression ast,
                           Environment env) {
          List<String> keys = Ordering.natural().sortedCopy(new ArrayList<String>(kwargs.keySet()));
          if ((Integer) args.get(0) == 0) {
            return keys;
          } else {
            return Lists.transform(keys, Functions.forMap(kwargs, null));
          }
        }
      };

    Environment env = singletonEnv(keyval.getName(), keyval);

    assertEquals(eval("['bar', 'foo', 'wiz']"),
                 eval("keyval(0, foo=1, bar='bar', wiz=[1,2,3])", env));

    assertEquals(eval("['bar', 1, [1,2,3]]"),
                 eval("keyval(1, foo=1, bar='bar', wiz=[1,2,3])", env));
  }

  @Test
  public void testMult() throws Exception {
    assertEquals(42, eval("6 * 7"));

    assertEquals("ababab", eval("3 * 'ab'"));
    assertEquals("", eval("0 * 'ab'"));
    assertEquals("100000", eval("'1' + '0' * 5"));
  }

  @Test
  public void testConcatStrings() throws Exception {
    assertEquals("foobar", eval("'foo' + 'bar'"));
  }

  @Test
  public void testConcatLists() throws Exception {
    // list
    Object x = eval("[1,2] + [3,4]");
    assertEquals(Arrays.asList(1, 2, 3, 4), x);
    assertFalse(EvalUtils.isImmutable(x));

    // tuple
    x = eval("(1,2) + (3,4)");
    assertEquals(Arrays.asList(1, 2, 3, 4), x);
    assertTrue(EvalUtils.isImmutable(x));

    checkEvalError("(1,2) + [3,4]", // list + tuple
        "can only concatenate List (not \"Tuple\") to List");
  }

  @SuppressWarnings("unchecked")
  @Test
  public void testListComprehensions() throws Exception {
    Iterable<Object> eval = (Iterable<Object>) eval(
        "['foo/%s.java' % x for x in []]");
    assertThat(eval).isEmpty();

    eval = (Iterable<Object>) eval(
        "['foo/%s.java' % x for x in ['bar', 'wiz', 'quux']]");
    assertThat(eval).containsExactly("foo/bar.java", "foo/wiz.java", "foo/quux.java").inOrder();

    eval = (Iterable<Object>) eval(
        "['%s/%s.java' % (x, y) "
        + "for x in ['foo', 'bar'] "
        + "for y in ['baz', 'wiz', 'quux']]");
    assertThat(eval).containsExactly("foo/baz.java", "foo/wiz.java", "foo/quux.java",
        "bar/baz.java", "bar/wiz.java", "bar/quux.java").inOrder();

    eval = (Iterable<Object>) eval(
        "['%s/%s.java' % (x, x) "
        + "for x in ['foo', 'bar'] "
        + "for x in ['baz', 'wiz', 'quux']]");
    assertThat(eval).containsExactly("baz/baz.java", "wiz/wiz.java", "quux/quux.java",
        "baz/baz.java", "wiz/wiz.java", "quux/quux.java").inOrder();

    eval = (Iterable<Object>) eval(
        "['%s/%s.%s' % (x, y, z) "
        + "for x in ['foo', 'bar'] "
        + "for y in ['baz', 'wiz', 'quux'] "
        + "for z in ['java', 'cc']]");
    assertThat(eval).containsExactly("foo/baz.java", "foo/baz.cc", "foo/wiz.java", "foo/wiz.cc",
        "foo/quux.java", "foo/quux.cc", "bar/baz.java", "bar/baz.cc", "bar/wiz.java", "bar/wiz.cc",
        "bar/quux.java", "bar/quux.cc").inOrder();
  }

  @Test
  public void testListComprehensionsMultipleVariables() throws Exception {
    assertThat(eval("[x + y for x, y in [(1, 2), (3, 4)]]").toString())
        .isEqualTo("[3, 7]");
    assertThat(eval("[x + y for (x, y) in [[1, 2], [3, 4]]]").toString())
        .isEqualTo("[3, 7]");
  }

  @Test
  public void testListComprehensionsMultipleVariablesFail() throws Exception {
    checkEvalError("[x + y for x, y, z in [(1, 2), (3, 4)]]",
        "lvalue has length 3, but rvalue has has length 2");

    checkEvalError("[x + y for x, y in (1, 2)]",
        "type 'int' is not a collection");
  }

  @Test
  public void testTupleDestructuring() throws Exception {
    exec(parseFile("a, b = 1, 2"), env);
    assertThat(env.lookup("a")).isEqualTo(1);
    assertThat(env.lookup("b")).isEqualTo(2);

    exec(parseFile("c, d = {'key1':2, 'key2':3}"), env);
    assertThat(env.lookup("c")).isEqualTo("key1");
    assertThat(env.lookup("d")).isEqualTo("key2");
  }

  @Test
  public void testRecursiveTupleDestructuring() throws Exception {
    List<Statement> input = parseFile("((a, b), (c, d)) = [(1, 2), (3, 4)]");
    exec(input, env);
    assertThat(env.lookup("a")).isEqualTo(1);
    assertThat(env.lookup("b")).isEqualTo(2);
    assertThat(env.lookup("c")).isEqualTo(3);
    assertThat(env.lookup("d")).isEqualTo(4);
  }

  // TODO(bazel-team): should this test work in Skylark?
  @SuppressWarnings("unchecked")
  @Test
  public void testListComprehensionModifiesGlobalEnv() throws Exception {
    Environment env = singletonEnv("x", 42);
    assertThat((Iterable<Object>) eval(parseExpr("[x + 1 for x in [1,2,3]]"), env))
        .containsExactly(2, 3, 4).inOrder();
    assertEquals(3, env.lookup("x")); // (x is global)
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
  public void testDictComprehensions_MultipleKey() throws Exception {
    assertEquals(ImmutableMap.of(1, 1, 2, 2), eval("{x : x for x in [1, 2, 1]}"));
    assertEquals(ImmutableMap.of("ab", "ab", "c", "c"),
        eval("{y : y for y in ['ab', 'c', 'a' + 'b']}"));
  }

  @Test
  public void testDictComprehensions_ToString() throws Exception {
    assertEquals("{x: x for x in [1, 2]}", parseExpr("{x : x for x in [1, 2]}").toString());
    assertEquals("{x + 'a': x for x in [1, 2]}",
        parseExpr("{x + 'a' : x for x in [1, 2]}").toString());
  }

  @Test
  public void testListConcatenation() throws Exception {
    assertEquals(Arrays.asList(1, 2, 3, 4), eval("[1, 2] + [3, 4]", env));
    assertEquals(ImmutableList.of(1, 2, 3, 4), eval("(1, 2) + (3, 4)", env));
    checkEvalError("[1, 2] + (3, 4)", "can only concatenate Tuple (not \"List\") to Tuple");
    checkEvalError("(1, 2) + [3, 4]", "can only concatenate List (not \"Tuple\") to List");
  }

  @Test
  public void testListComprehensionFailsOnNonSequence() throws Exception {
    checkEvalError("[x + 1 for x in 123]", "type 'int' is not an iterable");
  }

  @SuppressWarnings("unchecked")
  @Test
  public void testListComprehensionOnString() throws Exception {
    assertThat((Iterable<Object>) eval("[x for x in 'abc']")).containsExactly("a", "b", "c")
        .inOrder();
  }

  @Test
  public void testInvalidAssignment() throws Exception {
    Environment env = singletonEnv("x", 1);
    checkEvalError(parseStmt("x + 1 = 2"), env,
        "can only assign to variables and tuples, not to 'x + 1'");
  }

  @Test
  public void testListComprehensionOnDictionary() throws Exception {
    List<Statement> input = parseFile("val = ['var_' + n for n in {'a':1,'b':2}]");
    exec(input, env);
    Iterable<?> result = (Iterable<?>) env.lookup("val");
    assertThat(result).hasSize(2);
    assertEquals("var_a", Iterables.get(result, 0));
    assertEquals("var_b", Iterables.get(result, 1));
  }

  @Test
  public void testListComprehensionOnDictionaryCompositeExpression() throws Exception {
    exec(parseFile("d = {1:'a',2:'b'}\n"
                  + "l = [d[x] for x in d]"), env);
    assertEquals("[\"a\", \"b\"]", EvalUtils.prettyPrintValue(env.lookup("l")));
  }

  @Test
  public void testInOnListContains() throws Exception {
    assertEquals(Boolean.TRUE, eval("'b' in ['a', 'b']"));
  }

  @Test
  public void testInOnListDoesNotContain() throws Exception {
    assertEquals(Boolean.FALSE, eval("'c' in ['a', 'b']"));
  }

  @Test
  public void testInOnTupleContains() throws Exception {
    assertEquals(Boolean.TRUE, eval("'b' in ('a', 'b')"));
  }

  @Test
  public void testInOnTupleDoesNotContain() throws Exception {
    assertEquals(Boolean.FALSE, eval("'c' in ('a', 'b')"));
  }

  @Test
  public void testInOnDictContains() throws Exception {
    assertEquals(Boolean.TRUE, eval("'b' in {'a' : 1, 'b' : 2}"));
  }

  @Test
  public void testInOnDictDoesNotContainKey() throws Exception {
    assertEquals(Boolean.FALSE, eval("'c' in {'a' : 1, 'b' : 2}"));
  }

  @Test
  public void testInOnDictDoesNotContainVal() throws Exception {
    assertEquals(Boolean.FALSE, eval("1 in {'a' : 1, 'b' : 2}"));
  }

  @Test
  public void testInOnStringContains() throws Exception {
    assertEquals(Boolean.TRUE, eval("'b' in 'abc'"));
  }

  @Test
  public void testInOnStringDoesNotContain() throws Exception {
    assertEquals(Boolean.FALSE, eval("'d' in 'abc'"));
  }

  @Test
  public void testInOnStringLeftNotString() throws Exception {
    checkEvalError("1 in '123'",
        "in operator only works on strings if the left operand is also a string");
  }

  @Test
  public void testInFailsOnNonIterable() throws Exception {
    checkEvalError("'a' in 1",
        "in operator only works on lists, tuples, dictionaries and strings");
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
    env.update("obj", createObjWithStr());
    assertEquals("str marker", eval("'%s' % obj", env));
  }

  @Test
  public void testPercOnObjectList() throws Exception {
    env.update("obj", createObjWithStr());
    assertEquals("str marker str marker", eval("'%s %s' % (obj, obj)", env));
  }

  @Test
  public void testPercOnObjectInvalidFormat() throws Exception {
    env.update("obj", createObjWithStr());
    checkEvalError("'%d' % obj", env, "invalid arguments for format string");
  }

  @SuppressWarnings("unchecked")
  @Test
  public void testDictKeys() throws Exception {
    exec("v = {'a': 1}.keys() + ['b', 'c']", env);
    assertThat((Iterable<Object>) env.lookup("v")).containsExactly("a", "b", "c").inOrder();
  }

  @Test
  public void testDictKeysTooManyArgs() throws Exception {
    checkEvalError("{'a': 1}.keys('abc')", env, "Invalid number of arguments (expected 0)");
    checkEvalError("{'a': 1}.keys(arg='abc')", env, "Invalid number of arguments (expected 0)");
  }

  protected void checkEvalError(String input, String msg) throws Exception {
    checkEvalError(input, env, msg);
  }

  protected void checkEvalError(String input, Environment env, String msg) throws Exception {
    try {
      eval(input, env);
      fail();
    } catch (EvalException e) {
      assertThat(e).hasMessage(msg);
    }
  }

  protected void checkEvalError(Statement input, Environment env, String msg) throws Exception {
    checkEvalError(ImmutableList.of(input), env, msg);
  }

  protected void checkEvalError(List<Statement> input, Environment env, String msg)
      throws Exception {
    try {
      exec(input, env);
      fail();
    } catch (EvalException e) {
      assertThat(e).hasMessage(msg);
    }
  }
}
