// Copyright 2024 The Bazel Authors. All rights reserved.
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
//
package com.google.devtools.build.lib.bazel.bzlmod;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.bazel.bzlmod.CompiledModuleFile.IncludeStatement;
import net.starlark.java.syntax.Location;
import net.starlark.java.syntax.ParserInput;
import net.starlark.java.syntax.StarlarkFile;
import net.starlark.java.syntax.SyntaxError;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class CompiledModuleFileTest {

  private static ImmutableList<IncludeStatement> checkSyntax(String str) throws Exception {
    return CompiledModuleFile.checkModuleFileSyntax(
        StarlarkFile.parse(ParserInput.fromString(str, "test file")));
  }

  @Test
  public void checkSyntax_good() throws Exception {
    String program =
        """
        abc()
        include("hullo")
        foo = bar
        """;
    assertThat(checkSyntax(program))
        .containsExactly(
            new IncludeStatement("hullo", Location.fromFileLineColumn("test file", 2, 1)));
  }

  @Test
  public void checkSyntax_good_multiple() throws Exception {
    String program =
        """
        abc()
        include("hullo")
        foo = bar
        include('world')
        """;
    assertThat(checkSyntax(program))
        .containsExactly(
            new IncludeStatement("hullo", Location.fromFileLineColumn("test file", 2, 1)),
            new IncludeStatement("world", Location.fromFileLineColumn("test file", 4, 1)));
  }

  @Test
  public void checkSyntax_good_multilineLiteral() throws Exception {
    String program =
        """
        abc()
        # Ludicrous as this may be, it's still valid syntax. Your funeral, etc...
        include(\"""hullo
        world\""")
        """;
    assertThat(checkSyntax(program))
        .containsExactly(
            new IncludeStatement("hullo\nworld", Location.fromFileLineColumn("test file", 3, 1)));
  }

  @Test
  public void checkSyntax_good_benignUsageOfInclude() throws Exception {
    String program =
        """
        myext = use_extension('whatever')
        myext.include(include="hullo")
        """;
    assertThat(checkSyntax(program)).isEmpty();
  }

  @Test
  public void checkSyntax_good_includeIdentifierReassigned() throws Exception {
    String program =
        """
        include('world')
        include = print
        # from this point on, we no longer check anything about `include` usage.
        include('hello')
        str(include)
        exclude = include
        """;
    assertThat(checkSyntax(program))
        .containsExactly(
            new IncludeStatement("world", Location.fromFileLineColumn("test file", 1, 1)));
  }

  @Test
  public void checkSyntax_bad_if() throws Exception {
    String program =
        """
        abc()
        if d > 3:
          pass
        """;
    var ex = assertThrows(SyntaxError.Exception.class, () -> checkSyntax(program));
    assertThat(ex)
        .hasMessageThat()
        .contains("`if` statements are not allowed in MODULE.bazel files");
  }

  @Test
  public void checkSyntax_bad_assignIncludeResult() throws Exception {
    String program =
        """
        foo = include('hello')
        """;
    var ex = assertThrows(SyntaxError.Exception.class, () -> checkSyntax(program));
    assertThat(ex)
        .hasMessageThat()
        .contains("the `include` directive MUST be called directly at the top-level");
  }

  @Test
  public void checkSyntax_bad_assignIncludeIdentifier() throws Exception {
    String program =
        """
        foo = include
        foo('hello')
        """;
    var ex = assertThrows(SyntaxError.Exception.class, () -> checkSyntax(program));
    assertThat(ex)
        .hasMessageThat()
        .contains("the `include` directive MUST be called directly at the top-level");
  }

  @Test
  public void checkSyntax_bad_multipleArgumentsToInclude() throws Exception {
    String program =
        """
        include('hello', 'world')
        """;
    var ex = assertThrows(SyntaxError.Exception.class, () -> checkSyntax(program));
    assertThat(ex)
        .hasMessageThat()
        .contains("the `include` directive MUST be called with exactly one positional");
  }

  @Test
  public void checkSyntax_bad_keywordArgumentToInclude() throws Exception {
    String program =
        """
        include(label='hello')
        """;
    var ex = assertThrows(SyntaxError.Exception.class, () -> checkSyntax(program));
    assertThat(ex)
        .hasMessageThat()
        .contains("the `include` directive MUST be called with exactly one positional");
  }

  @Test
  public void checkSyntax_bad_nonLiteralArgumentToInclude() throws Exception {
    String program =
        """
        foo = 'hello'
        include(foo)
        """;
    var ex = assertThrows(SyntaxError.Exception.class, () -> checkSyntax(program));
    assertThat(ex)
        .hasMessageThat()
        .contains("the `include` directive MUST be called with exactly one positional");
  }
}
