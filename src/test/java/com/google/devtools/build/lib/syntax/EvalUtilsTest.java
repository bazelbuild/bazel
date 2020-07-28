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
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.syntax.EvalUtils.ComparisonException;
import javax.annotation.Nullable;
import net.starlark.java.annot.StarlarkBuiltin;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Test properties of the evaluator's datatypes and utility functions without actually creating any
 * parse trees.
 */
@RunWith(JUnit4.class)
public final class EvalUtilsTest {

  private static StarlarkList<Object> makeList(@Nullable Mutability mu) {
    return StarlarkList.of(mu, 1, 2, 3);
  }

  private static Dict<Object, Object> makeDict(@Nullable Mutability mu) {
    return Dict.of(mu, 1, 1, 2, 2);
  }

  /** MockClassA */
  @StarlarkBuiltin(name = "MockClassA", doc = "MockClassA")
  public static class MockClassA implements StarlarkValue {}

  /** MockClassB */
  public static class MockClassB extends MockClassA {
  }

  @Test
  public void testDataTypeNames() throws Exception {
    assertThat(Starlark.type("foo")).isEqualTo("string");
    assertThat(Starlark.type(3)).isEqualTo("int");
    assertThat(Starlark.type(Tuple.of(1, 2, 3))).isEqualTo("tuple");
    assertThat(Starlark.type(makeList(null))).isEqualTo("list");
    assertThat(Starlark.type(makeDict(null))).isEqualTo("dict");
    assertThat(Starlark.type(Starlark.NONE)).isEqualTo("NoneType");
    assertThat(Starlark.type(new MockClassA())).isEqualTo("MockClassA");
    assertThat(Starlark.type(new MockClassB())).isEqualTo("MockClassA");
  }

  @Test
  public void testDatatypeMutabilityPrimitive() throws Exception {
    assertThat(EvalUtils.isImmutable("foo")).isTrue();
    assertThat(EvalUtils.isImmutable(3)).isTrue();
  }

  @Test
  public void testDatatypeMutabilityShallow() throws Exception {
    assertThat(EvalUtils.isImmutable(Tuple.of(1, 2, 3))).isTrue();

    assertThat(EvalUtils.isImmutable(makeList(null))).isTrue();
    assertThat(EvalUtils.isImmutable(makeDict(null))).isTrue();

    Mutability mu = Mutability.create("test");
    assertThat(EvalUtils.isImmutable(makeList(mu))).isFalse();
    assertThat(EvalUtils.isImmutable(makeDict(mu))).isFalse();
  }

  @Test
  public void testDatatypeMutabilityDeep() throws Exception {
    Mutability mu = Mutability.create("test");
    assertThat(EvalUtils.isImmutable(Tuple.of(makeList(null)))).isTrue();
    assertThat(EvalUtils.isImmutable(Tuple.of(makeList(mu)))).isFalse();
  }

  @Test
  public void testComparatorWithDifferentTypes() throws Exception {
    Mutability mu = Mutability.create("test");

    StarlarkValue myValue = new StarlarkValue() {};

    Object[] objects = {
      "1",
      2,
      true,
      Starlark.NONE,
      Tuple.of(1, 2, 3),
      Tuple.of("1", "2", "3"),
      StarlarkList.of(mu, 1, 2, 3),
      StarlarkList.of(mu, "1", "2", "3"),
      Dict.of(mu, "key", 123),
      Dict.of(mu, 123, "value"),
      myValue,
    };

    for (int i = 0; i < objects.length; ++i) {
      for (int j = 0; j < objects.length; ++j) {
        if (i != j) {
          Object first = objects[i];
          Object second = objects[j];
          assertThrows(
              ComparisonException.class,
              () -> EvalUtils.STARLARK_COMPARATOR.compare(first, second));
        }
      }
    }
  }

  @Test
  public void testComparatorWithNones() throws Exception {
    assertThrows(
        ComparisonException.class,
        () -> EvalUtils.STARLARK_COMPARATOR.compare(Starlark.NONE, Starlark.NONE));
  }

  @Test
  public void testLen() {
    assertThat(Starlark.len("abc")).isEqualTo(3);
    assertThat(Starlark.len(Tuple.of(1, 2, 3))).isEqualTo(3);
    assertThat(Starlark.len(StarlarkList.of(null, 1, 2, 3))).isEqualTo(3);
    assertThat(Starlark.len(Dict.of(null, "one", 1, "two", 2))).isEqualTo(2);
    assertThat(Starlark.len(true)).isEqualTo(-1);
    assertThrows(IllegalArgumentException.class, () -> Starlark.len(this));
  }
}
