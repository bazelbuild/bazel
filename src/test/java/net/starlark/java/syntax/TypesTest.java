// Copyright 2025 The Bazel Authors. All Rights Reserved.
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

package net.starlark.java.syntax;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests of built-in type objects. */
// TODO: #27370 - Move this to match whichever package Types.java is going to live in.
@RunWith(JUnit4.class)
public class TypesTest {

  /** Asserts {@code t1} is assignable to {@code t2}. */
  private static void assertLt(StarlarkType t1, StarlarkType t2) {
    assertWithMessage("%s is expected to be assignable to %s", t1, t2)
        .that(StarlarkType.assignableFrom(t2, t1))
        .isTrue();
  }

  /** Asserts {@code t1} is *not* assignable to {@code t2}. */
  private static void assertNotLt(StarlarkType t1, StarlarkType t2) {
    assertWithMessage("%s is expected to be *not* assignable to %s", t1, t2)
        .that(StarlarkType.assignableFrom(t2, t1))
        .isFalse();
  }

  /**
   * Asserts that the given types form a strict chain of assignability, with the ith element being
   * assignable to all jth elements where i < j, but not vice versa.
   */
  private static void assertStrictLtChain(StarlarkType... types) {
    for (int i = 0; i < types.length - 1; i++) {
      for (int j = i + 1; j < types.length; j++) {
        assertLt(types[i], types[j]);
        assertNotLt(types[j], types[i]);
      }
    }
  }

  @Test
  public void assignability_reflexivity() {
    assertLt(Types.INT, Types.INT);
    assertLt(Types.ANY, Types.ANY);
    assertLt(Types.OBJECT, Types.OBJECT);
  }

  @Test
  public void assignability_anyPassesEitherDirection() {
    assertLt(Types.INT, Types.ANY);
    assertLt(Types.ANY, Types.INT);
  }

  @Test
  public void assignability_objectIsTop() {
    assertLt(Types.INT, Types.OBJECT);
    assertNotLt(Types.OBJECT, Types.INT);
    assertLt(Types.ANY, Types.OBJECT);
    assertLt(Types.OBJECT, Types.ANY);
  }

  @Test
  public void assignability_primitiveTypesAreIncompatible() {
    assertNotLt(Types.INT, Types.STR);
    assertNotLt(Types.INT, Types.FLOAT); // unlike Python
    assertNotLt(Types.STR, Types.FLOAT);
    assertNotLt(Types.BOOL, Types.INT); // unlike Python
    assertNotLt(Types.NONE, Types.STR);
  }

  @Test
  public void assignability_union() {
    StarlarkType intOrStr = Types.union(Types.INT, Types.STR);
    StarlarkType intOrFloatOrStr = Types.union(Types.INT, Types.FLOAT, Types.STR);
    StarlarkType floatOrStr = Types.union(Types.FLOAT, Types.STR);
    // Assignability of a primitive type to a union
    assertLt(Types.INT, intOrStr);
    assertLt(Types.STR, intOrStr);
    assertLt(Types.ANY, intOrStr);
    assertNotLt(Types.FLOAT, intOrStr);
    assertNotLt(Types.OBJECT, intOrStr);

    // Assignability of a union to a primitive type
    assertLt(intOrStr, Types.ANY);
    assertLt(intOrStr, Types.OBJECT);
    assertNotLt(intOrStr, Types.INT);
    assertNotLt(intOrStr, Types.STR);

    // Assignability between unions
    assertLt(intOrStr, intOrStr);
    assertLt(intOrStr, intOrFloatOrStr);
    assertNotLt(intOrFloatOrStr, intOrStr);
    assertNotLt(intOrStr, floatOrStr);
    assertNotLt(floatOrStr, intOrStr);
  }

  @Test
  public void assignability_collection_subtypes() {
    StarlarkType intOrStr = Types.union(Types.INT, Types.STR);

    assertStrictLtChain(
        Types.list(Types.INT), Types.sequence(Types.INT), Types.collection(Types.INT));

    assertStrictLtChain(
        Types.tuple(Types.INT, Types.STR, Types.INT, Types.STR),
        Types.homogeneousTuple(intOrStr),
        Types.sequence(intOrStr),
        Types.collection(intOrStr));

    assertStrictLtChain(Types.set(Types.STR), Types.collection(Types.STR));
    assertNotLt(Types.set(Types.STR), Types.sequence(Types.STR));

    assertStrictLtChain(
        Types.dict(Types.STR, Types.INT),
        Types.mapping(Types.STR, Types.INT),
        Types.collection(Types.STR));
    assertNotLt(Types.dict(Types.STR, Types.INT), Types.sequence(Types.STR));

    // Works with unions too.
    assertStrictLtChain(
        Types.union(Types.dict(Types.STR, Types.INT), Types.list(Types.STR)),
        Types.union(Types.collection(Types.STR), Types.collection(Types.BOOL)));
  }

  @Test
  public void callable_toSignatureString() {
    // ordinary only
    assertThat(
            Types.callable(
                    /* parameterNames= */ ImmutableList.of("a"),
                    /* parameterTypes= */ ImmutableList.of(Types.INT),
                    /* numPositionalOnlyParameters= */ 0,
                    /* numPositionalParameters= */ 1,
                    /* mandatoryParams= */ ImmutableSet.of("a"),
                    /* varargsType= */ null,
                    /* kwargsType= */ null,
                    Types.BOOL)
                .toSignatureString())
        .isEqualTo("(a: int) -> bool");
    // kwonly only
    assertThat(
            Types.callable(
                    /* parameterNames= */ ImmutableList.of("a"),
                    /* parameterTypes= */ ImmutableList.of(Types.INT),
                    /* numPositionalOnlyParameters= */ 0,
                    /* numPositionalParameters= */ 0,
                    /* mandatoryParams= */ ImmutableSet.of("a"),
                    /* varargsType= */ null,
                    /* kwargsType= */ null,
                    Types.BOOL)
                .toSignatureString())
        .isEqualTo("(*, a: int) -> bool");
    // positional-only only
    assertThat(
            Types.callable(
                    /* parameterNames= */ ImmutableList.of("x"),
                    /* parameterTypes= */ ImmutableList.of(Types.INT),
                    /* numPositionalOnlyParameters= */ 1,
                    /* numPositionalParameters= */ 0,
                    /* mandatoryParams= */ ImmutableSet.of(),
                    /* varargsType= */ null,
                    /* kwargsType= */ null,
                    Types.BOOL)
                .toSignatureString())
        .isEqualTo("([int], /) -> bool");
    // no params
    assertThat(
            Types.callable(
                    /* parameterNames= */ ImmutableList.of(),
                    /* parameterTypes= */ ImmutableList.of(),
                    /* numPositionalOnlyParameters= */ 0,
                    /* numPositionalParameters= */ 0,
                    /* mandatoryParams= */ ImmutableSet.of(),
                    /* varargsType= */ null,
                    /* kwargsType= */ null,
                    Types.BOOL)
                .toSignatureString())
        .isEqualTo("() -> bool");
    // all kinds of params
    assertThat(
            Types.callable(
                    /* parameterNames= */ ImmutableList.of("x", "a", "b", "c", "d"),
                    /* parameterTypes= */ ImmutableList.of(
                        Types.BOOL, Types.INT, Types.FLOAT, Types.NONE, Types.ANY),
                    /* numPositionalOnlyParameters= */ 1,
                    /* numPositionalParameters= */ 3,
                    /* mandatoryParams= */ ImmutableSet.of("a", "c", "x"),
                    /* varargsType= */ Types.INT,
                    /* kwargsType= */ Types.INT,
                    Types.BOOL)
                .toSignatureString())
        .isEqualTo(
            "(bool, /, a: int, b: [float], *args: int, c: None, d: [Any], **kwargs: int) -> bool");
  }
}
