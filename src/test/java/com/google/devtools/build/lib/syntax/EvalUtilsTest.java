// Copyright 2006 The Bazel Authors. All Rights Reserved.
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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.syntax.ClassObject.SkylarkClassObject;
import com.google.devtools.build.lib.syntax.SkylarkList.MutableList;
import com.google.devtools.build.lib.syntax.SkylarkList.Tuple;
import com.google.devtools.build.lib.syntax.util.EvaluationTestCase;
import java.util.TreeMap;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 *  Test properties of the evaluator's datatypes and utility functions
 *  without actually creating any parse trees.
 */
@RunWith(JUnit4.class)
public class EvalUtilsTest extends EvaluationTestCase {

  private static MutableList<Object> makeList(Environment env) {
    return MutableList.<Object>of(env, 1, 2, 3);
  }

  private static SkylarkDict<Object, Object> makeDict(Environment env) {
    return SkylarkDict.<Object, Object>of(env, 1, 1, 2, 2);
  }

  private static SkylarkClassObject makeStruct(String field, Object value) {
    return new SkylarkClassObject(ImmutableMap.of(field, value));
  }

  private static SkylarkClassObject makeBigStruct(Environment env) {
    // struct(a=[struct(x={1:1}), ()], b=(), c={2:2})
    return new SkylarkClassObject(ImmutableMap.<String, Object>of(
        "a", MutableList.<Object>of(env,
            new SkylarkClassObject(ImmutableMap.<String, Object>of(
                "x", SkylarkDict.<Object, Object>of(env, 1, 1))),
            Tuple.of()),
        "b", Tuple.of(),
        "c", SkylarkDict.<Object, Object>of(env, 2, 2)));
  }

  @Test
  public void testEmptyStringToIterable() throws Exception {
    assertThat(EvalUtils.toIterable("", null)).isEmpty();
  }

  @Test
  public void testStringToIterable() throws Exception {
    assertThat(EvalUtils.toIterable("abc", null)).hasSize(3);
  }

  @Test
  public void testDataTypeNames() throws Exception {
    assertEquals("string", EvalUtils.getDataTypeName("foo"));
    assertEquals("int", EvalUtils.getDataTypeName(3));
    assertEquals("tuple", EvalUtils.getDataTypeName(Tuple.of(1, 2, 3)));
    assertEquals("list",  EvalUtils.getDataTypeName(makeList(null)));
    assertEquals("dict",  EvalUtils.getDataTypeName(makeDict(null)));
    assertEquals("NoneType", EvalUtils.getDataTypeName(Runtime.NONE));
  }

  @Test
  public void testDatatypeMutabilityPrimitive() throws Exception {
    assertTrue(EvalUtils.isImmutable("foo"));
    assertTrue(EvalUtils.isImmutable(3));
  }

  @Test
  public void testDatatypeMutabilityShallow() throws Exception {
    assertTrue(EvalUtils.isImmutable(Tuple.of(1, 2, 3)));
    assertTrue(EvalUtils.isImmutable(makeStruct("a", 1)));

    // Mutability depends on the environment.
    assertTrue(EvalUtils.isImmutable(makeList(null)));
    assertTrue(EvalUtils.isImmutable(makeDict(null)));
    assertFalse(EvalUtils.isImmutable(makeList(env)));
    assertFalse(EvalUtils.isImmutable(makeDict(env)));
  }

  @Test
  public void testDatatypeMutabilityDeep() throws Exception {
    assertTrue(EvalUtils.isImmutable(Tuple.<Object>of(makeList(null))));
    assertTrue(EvalUtils.isImmutable(makeStruct("a", makeList(null))));
    assertTrue(EvalUtils.isImmutable(makeBigStruct(null)));

    assertFalse(EvalUtils.isImmutable(Tuple.<Object>of(makeList(env))));
    assertFalse(EvalUtils.isImmutable(makeStruct("a", makeList(env))));
    assertFalse(EvalUtils.isImmutable(makeBigStruct(env)));
  }

  @Test
  public void testComparatorWithDifferentTypes() throws Exception {
    TreeMap<Object, Object> map = new TreeMap<>(EvalUtils.SKYLARK_COMPARATOR);
    map.put(2, 3);
    map.put("1", 5);
    map.put(42, 4);
    map.put("test", 7);
    map.put(-1, 2);
    map.put("4", 6);
    map.put(true, 1);
    map.put(Runtime.NONE, 0);

    int expected = 0;
    // Expected order of keys is NoneType -> Double -> Integers -> Strings
    for (Object obj : map.values()) {
      assertThat(obj).isEqualTo(expected);
      ++expected;
    }
  }
}
