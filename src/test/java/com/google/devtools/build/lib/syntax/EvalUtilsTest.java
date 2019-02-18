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
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.StructProvider;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrinter;
import com.google.devtools.build.lib.skylarkinterface.SkylarkValue;
import com.google.devtools.build.lib.syntax.EvalUtils.ComparisonException;
import com.google.devtools.build.lib.syntax.SkylarkList.MutableList;
import com.google.devtools.build.lib.syntax.SkylarkList.Tuple;
import com.google.devtools.build.lib.syntax.util.EvaluationTestCase;
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
    return MutableList.of(env, 1, 2, 3);
  }

  private static SkylarkDict<Object, Object> makeDict(Environment env) {
    return SkylarkDict.of(env, 1, 1, 2, 2);
  }

  @Test
  public void testEmptyStringToIterable() throws Exception {
    assertThat(EvalUtils.toIterable("", null, null)).isEmpty();
  }

  @Test
  public void testStringToIterable() throws Exception {
    assertThat(EvalUtils.toIterable("abc", null, null)).hasSize(3);
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
    assertThat(EvalUtils.getDataTypeName("foo")).isEqualTo("string");
    assertThat(EvalUtils.getDataTypeName(3)).isEqualTo("int");
    assertThat(EvalUtils.getDataTypeName(Tuple.of(1, 2, 3))).isEqualTo("tuple");
    assertThat(EvalUtils.getDataTypeName(makeList(null))).isEqualTo("list");
    assertThat(EvalUtils.getDataTypeName(makeDict(null))).isEqualTo("dict");
    assertThat(EvalUtils.getDataTypeName(Runtime.NONE)).isEqualTo("NoneType");
    assertThat(EvalUtils.getDataTypeName(new MockClassA())).isEqualTo("MockClassA");
    assertThat(EvalUtils.getDataTypeName(new MockClassB())).isEqualTo("MockClassA");
  }

  @Test
  public void testDatatypeMutabilityPrimitive() throws Exception {
    assertThat(EvalUtils.isImmutable("foo")).isTrue();
    assertThat(EvalUtils.isImmutable(3)).isTrue();
  }

  @Test
  public void testDatatypeMutabilityShallow() throws Exception {
    assertThat(EvalUtils.isImmutable(Tuple.of(1, 2, 3))).isTrue();

    // Mutability depends on the environment.
    assertThat(EvalUtils.isImmutable(makeList(null))).isTrue();
    assertThat(EvalUtils.isImmutable(makeDict(null))).isTrue();
    assertThat(EvalUtils.isImmutable(makeList(env))).isFalse();
    assertThat(EvalUtils.isImmutable(makeDict(env))).isFalse();
  }

  @Test
  public void testDatatypeMutabilityDeep() throws Exception {
    assertThat(EvalUtils.isImmutable(Tuple.<Object>of(makeList(null)))).isTrue();

    assertThat(EvalUtils.isImmutable(Tuple.<Object>of(makeList(env)))).isFalse();
  }

  @Test
  public void testComparatorWithDifferentTypes() throws Exception {
    Object[] objects = {
      "1",
      2,
      true,
      Runtime.NONE,
      SkylarkList.Tuple.of(1, 2, 3),
      SkylarkList.Tuple.of("1", "2", "3"),
      SkylarkList.MutableList.of(env, 1, 2, 3),
      SkylarkList.MutableList.of(env, "1", "2", "3"),
      SkylarkDict.of(env, "key", 123),
      SkylarkDict.of(env, 123, "value"),
      NestedSetBuilder.stableOrder().add(1).add(2).add(3).build(),
      StructProvider.STRUCT.create(ImmutableMap.of("key", (Object) "value"), "no field %s"),
    };

    for (int i = 0; i < objects.length; ++i) {
      for (int j = 0; j < objects.length; ++j) {
        if (i != j) {
          try {
            EvalUtils.SKYLARK_COMPARATOR.compare(objects[i], objects[j]);
            fail("Shouldn't have compared different types");
          } catch (ComparisonException e) {
            // expected
          }
        }
      }
    }
  }

  @Test
  public void testComparatorWithNones() throws Exception {
    try {
      EvalUtils.SKYLARK_COMPARATOR.compare(Runtime.NONE, Runtime.NONE);
      fail("Shouldn't have compared nones");
    } catch (ComparisonException e) {
      // expected
    }
  }

  @SkylarkModule(
      name = "ParentType",
      doc = "A parent class annotated with @SkylarkModule."
  )
  private static class ParentClassWithSkylarkModule {}

  private static class ChildClass extends ParentClassWithSkylarkModule {}

  private static class SkylarkValueSubclass implements SkylarkValue {
    @Override
    public void repr(SkylarkPrinter printer) {
      printer.append("SkylarkValueSubclass");
    }
  }

  private static class NonSkylarkValueSubclass {}

  @Test
  public void testGetSkylarkType() {
    assertThat(EvalUtils.getSkylarkType(ParentClassWithSkylarkModule.class))
        .isEqualTo(ParentClassWithSkylarkModule.class);
    assertThat(EvalUtils.getSkylarkType(ChildClass.class))
        .isEqualTo(ParentClassWithSkylarkModule.class);
    assertThat(EvalUtils.getSkylarkType(SkylarkValueSubclass.class))
        .isEqualTo(SkylarkValueSubclass.class);

    IllegalArgumentException expected =
        assertThrows(
            IllegalArgumentException.class,
            () -> EvalUtils.getSkylarkType(NonSkylarkValueSubclass.class));
    assertThat(expected)
        .hasMessageThat()
        .contains(
            "class com.google.devtools.build.lib.syntax.EvalUtilsTest$NonSkylarkValueSubclass "
                + "is not allowed as a Starlark value");
  }
}
