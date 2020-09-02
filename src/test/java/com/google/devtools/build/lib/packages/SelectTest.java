// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.packages;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.FileOptions;
import com.google.devtools.build.lib.syntax.Module;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.syntax.ParserInput;
import com.google.devtools.build.lib.syntax.Starlark;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import com.google.devtools.build.lib.syntax.StarlarkThread;
import com.google.devtools.build.lib.syntax.SyntaxError;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests of @{code select} function and data type. */
@RunWith(JUnit4.class)
public class SelectTest {

  private static Object eval(String expr)
      throws SyntaxError.Exception, EvalException, InterruptedException {
    ParserInput input = ParserInput.fromLines(expr);
    Module module =
        Module.withPredeclared(StarlarkSemantics.DEFAULT, /*predeclared=*/ StarlarkLibrary.COMMON);
    try (Mutability mu = Mutability.create()) {
      StarlarkThread thread = new StarlarkThread(mu, StarlarkSemantics.DEFAULT);
      return Starlark.eval(input, FileOptions.DEFAULT, module, thread);
    }
  }

  private static void assertFails(String expr, String wantError) {
    EvalException ex = assertThrows(EvalException.class, () -> eval(expr));
    assertThat(ex).hasMessageThat().contains(wantError);
  }

  @Test
  public void testSelect() throws Exception {
    SelectorList result = (SelectorList) eval("select({'a': 1})");
    assertThat(((SelectorValue) Iterables.getOnlyElement(result.getElements())).getDictionary())
        .containsExactly("a", 1);
  }

  @Test
  public void testPlus() throws Exception {
    SelectorList x = (SelectorList) eval("select({'foo': ['FOO'], 'bar': ['BAR']}) + []");
    List<Object> elements = x.getElements();
    assertThat(elements).hasSize(2);
    assertThat(elements.get(0)).isInstanceOf(SelectorValue.class);
    assertThat((Iterable<?>) elements.get(1)).isEmpty();
  }

  @Test
  public void testPlusIncompatibleType() throws Exception {
    assertFails(
        "select({'foo': ['FOO'], 'bar': ['BAR']}) + 1",
        "'+' operator applied to incompatible types (select of list, int)");
    assertFails(
        "select({'foo': ['FOO']}) + select({'bar': 2})",
        "'+' operator applied to incompatible types (select of list, select of int)");
  }
}
