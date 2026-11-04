// Copyright 2026 The Bazel Authors. All rights reserved.
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

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests of the kinds of expressions and statements. */
@RunWith(JUnit4.class)
public final class NodeKindTest {

  @Test
  public void expressionKindOrdinalConstants() throws Exception {
    assertThat(Expression.Kind.values().length).isAtMost(1 << Node.KIND_BITS);
    for (Expression.Kind kind : Expression.Kind.values()) {
      assertWithMessage(kind.name())
          .that(Expression.class.getField(kind.name()).getInt(null))
          .isEqualTo(kind.ordinal());
    }
  }

  @Test
  public void statementKindOrdinalConstants() throws Exception {
    assertThat(Statement.Kind.values().length).isAtMost(1 << Node.KIND_BITS);
    for (Statement.Kind kind : Statement.Kind.values()) {
      assertWithMessage(kind.name())
          .that(Statement.class.getField(kind.name()).getInt(null))
          .isEqualTo(kind.ordinal());
    }
  }
}
