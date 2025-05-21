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

package net.starlark.java.eval;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.truth.StringSubject;
import net.starlark.java.syntax.FileOptions;
import net.starlark.java.types.StarlarkType;
import net.starlark.java.types.Types.CallableType;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for Starlark types. */
@RunWith(JUnit4.class)
public class TypeCheckTest {

  private EvaluationTestCase ev;

  @Before
  public void setup() {
    ev = new EvaluationTestCase();
    ev.setFileOptions(FileOptions.builder().allowTypeAnnotations(true).build());
  }

  @Test
  public void runtimeTypecheck() throws Exception {
    ev.exec("def f(a: None): pass", "f(None)");
    ev.exec("def f(a: bool): pass", "f(True)");
    ev.exec("def f(a: int): pass", "f(1)");
    ev.exec("def f(a: float): pass", "f(1.1)");
    ev.exec("def f(a: str): pass", "f('abc')");
    ev.exec("def f(a): pass", "f('abc')");
    ev.exec("def f(a): pass", "f(['abc'])");
    ev.exec("def f(x): pass", "def g(x): pass", "f(g)");
    // int is not below float
    assertExecThrows(EvalException.class, "def f(a: float): pass", "f(1)")
        .isEqualTo("in call to f(), parameter 'a' got value of type 'int', want 'float'");

    assertExecThrows(EvalException.class, "def f(a: int): pass", "f('abc')")
        .isEqualTo("in call to f(), parameter 'a' got value of type 'str', want 'int'");

    assertExecThrows(EvalException.class, "def f(a: int = 'abc'): pass")
        .isEqualTo("f(): parameter 'a' has default value of type 'str', declares 'int'");

    assertExecThrows(EvalException.class, "def f() -> int: return 'abc'", "f()")
        .isEqualTo("f(): returns value of type 'str', declares 'int'");
  }

  @Test
  public void lambdaDoesntFail() throws Exception {
    // Lambda has functionType set to null
    ev.exec(
        "def f(a: None):", //
        "  x = lambda y: 1",
        "  x(1)",
        "  y = lambda y = 1: 1",
        "  y(1)",
        "f(None)");
  }

  @Test
  public void testStarlarkUniverseTypes() {
    ImmutableList.Builder<String> builder = ImmutableList.builder();
    for (var entry : Starlark.UNIVERSE.entrySet()) {
      StarlarkType type = TypeChecker.type(entry.getValue());
      if (type instanceof CallableType callable) {
        builder.add(entry.getKey() + ": " + callable.toSignatureString());
      } else {
        builder.add(entry.getKey() + ": " + type);
      }
    }

    assertThat(builder.build())
        .containsAtLeast(
            "False: bool", //
            "True: bool",
            "None: None",
            "hash: (str, /) -> int",
            "bool: (object, /) -> bool",
            "getattr: (object, str, object, /) -> Any",
            "hasattr: (object, str, /) -> bool",
            "repr: (object, /) -> str",
            "str: (object, /) -> str",
            "type: (object, /) -> str");
  }

  private <T extends Throwable> StringSubject assertExecThrows(
      Class<T> expectedThrowable, String... lines) {
    T evalException = assertThrows(expectedThrowable, () -> ev.exec(lines));
    return assertThat(evalException).hasMessageThat();
  }
}
