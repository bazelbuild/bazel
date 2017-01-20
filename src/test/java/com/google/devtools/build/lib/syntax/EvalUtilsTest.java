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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
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

  @Test
  public void testEmptyStringToIterable() throws Exception {
    assertThat(EvalUtils.toIterable("", null)).isEmpty();
  }

  @Test
  public void testStringToIterable() throws Exception {
    assertThat(EvalUtils.toIterable("abc", null)).hasSize(3);
  }

  @Test
  public void testSize() throws Exception {
    assertThat(EvalUtils.size("abc")).isEqualTo(3);
    assertThat(EvalUtils.size(ImmutableMap.of(1, 2, 3, 4))).isEqualTo(2);
    assertThat(EvalUtils.size(SkylarkList.Tuple.of(1, 2, 3))).isEqualTo(3);
    SkylarkNestedSet set = SkylarkNestedSet.of(
        Object.class,
        NestedSetBuilder.stableOrder().add(1).add(2).add(3).build());
    assertThat(EvalUtils.size(set)).isEqualTo(3);
    assertThat(EvalUtils.size(ImmutableList.of(1, 2, 3))).isEqualTo(3);
  }

  /** MockClassA */
  @SkylarkModule(name = "MockClassA", doc = "MockClassA")
  public static class MockClassA {
  }

  /** MockClassB */
  public static class MockClassB extends MockClassA {
  }

  @Test
  public void testDataTypeNames() throws Exception {
    assertEquals("string", EvalUtils.getDataTypeName("foo"));
    assertEquals("int", EvalUtils.getDataTypeName(3));
    assertEquals("tuple", EvalUtils.getDataTypeName(Tuple.of(1, 2, 3)));
    assertEquals("list",  EvalUtils.getDataTypeName(makeList(null)));
    assertEquals("dict",  EvalUtils.getDataTypeName(makeDict(null)));
    assertEquals("NoneType", EvalUtils.getDataTypeName(Runtime.NONE));
    assertEquals("MockClassA", EvalUtils.getDataTypeName(new MockClassA()));
    assertEquals("MockClassA", EvalUtils.getDataTypeName(new MockClassB()));
  }

  @Test
  public void testDatatypeMutabilityPrimitive() throws Exception {
    assertTrue(EvalUtils.isImmutable("foo"));
    assertTrue(EvalUtils.isImmutable(3));
  }

  @Test
  public void testDatatypeMutabilityShallow() throws Exception {
    assertTrue(EvalUtils.isImmutable(Tuple.of(1, 2, 3)));

    // Mutability depends on the environment.
    assertTrue(EvalUtils.isImmutable(makeList(null)));
    assertTrue(EvalUtils.isImmutable(makeDict(null)));
    assertFalse(EvalUtils.isImmutable(makeList(env)));
    assertFalse(EvalUtils.isImmutable(makeDict(env)));
  }

  @Test
  public void testDatatypeMutabilityDeep() throws Exception {
    assertTrue(EvalUtils.isImmutable(Tuple.<Object>of(makeList(null))));

    assertFalse(EvalUtils.isImmutable(Tuple.<Object>of(makeList(env))));
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
