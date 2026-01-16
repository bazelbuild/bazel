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
    assertThat(StarlarkType.assignableFrom(t2, t1)).isTrue();
  }

  /** Asserts {@code t1} is *not* assignable to {@code t2}. */
  private static void assertNotLt(StarlarkType t1, StarlarkType t2) {
    assertThat(StarlarkType.assignableFrom(t2, t1)).isFalse();
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
