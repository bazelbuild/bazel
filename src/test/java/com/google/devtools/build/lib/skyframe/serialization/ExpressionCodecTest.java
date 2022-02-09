// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.serialization;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import net.starlark.java.syntax.Expression;
import net.starlark.java.syntax.ParserInput;
import net.starlark.java.syntax.SyntaxError;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link Expression_AutoCodec}. */
@RunWith(JUnit4.class)
public class ExpressionCodecTest {

  @Test
  public void testCodec() throws Exception {
    new SerializationTester(
            parseExpression("1 + 1"),
            parseExpression("1 if True else 2"),
            parseExpression("{x: 2*x for x in [1, 2, 3] if x > 0}"),
            parseExpression("{}"),
            parseExpression("{1:2, 3:4}"),
            parseExpression("a.b.c"),
            parseExpression("f()"),
            parseExpression("(-b + sqrt(b*b - 4*a*c)) / (2*a)"),
            parseExpression("f(w, x=5, *y, **z)"),
            parseExpression("var"),
            parseExpression("a[i]"),
            parseExpression("5"),
            parseExpression("[2*x for x in [1, 2, 3] if x > 0]"),
            parseExpression("[1, 2, 3]"),
            parseExpression("not True"),
            parseExpression("a[x:y:z]"),
            parseExpression("'foo'"),
            // This is optimized by the parser, so it's pretty much equivalent to the string literal
            // case above.
            parseExpression("'cat' + 'dog'"))
        .setVerificationFunction(
            (original, deserialized) -> {
              Expression x = (Expression) original;
              Expression y = (Expression) deserialized;
              assertThat(x.prettyPrint()).isEqualTo(y.prettyPrint());
              assertThat(x.getStartLocation()).isEqualTo(y.getStartLocation());
              assertThat(x.getEndLocation()).isEqualTo(y.getEndLocation());
            })
        .runTests();
  }

  private static Expression parseExpression(String input) throws SyntaxError.Exception {
    return Expression.parse(ParserInput.fromString(input, "<file>"));
  }
}
