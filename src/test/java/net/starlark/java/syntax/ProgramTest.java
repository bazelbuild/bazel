// Copyright 2025 The Bazel Authors. All rights reserved.
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

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests of the Starlark {@link Program}. */
@RunWith(JUnit4.class)
public final class ProgramTest {

  private Program compileFile(String... lines) throws SyntaxError.Exception {
    ParserInput input = ParserInput.fromLines(lines);
    StarlarkFile file = StarlarkFile.parse(input, FileOptions.DEFAULT);
    return Program.compileFile(file, TestUtils.Module.withPredeclared("pre"));
  }

  @Test
  public void docComments_basicFunctionality() throws Exception {
    Program program =
        compileFile(
            """
            #: Doc comment for A
            #: multiline
            FOO = 1
            BAR, BAZ = (2, 3)  #: Applies to LHS list
            """);
    assertThat(program.getDocCommentsMap().keySet()).containsExactly("FOO", "BAR", "BAZ").inOrder();
    assertThat(program.getDocCommentsMap().get("FOO").getText())
        .isEqualTo("Doc comment for A\nmultiline");
    assertThat(program.getDocCommentsMap().get("BAR").getText()).isEqualTo("Applies to LHS list");
    assertThat(program.getDocCommentsMap().get("BAZ").getText()).isEqualTo("Applies to LHS list");
  }

  @Test
  public void docComments_unused() throws Exception {
    Program program =
        compileFile(
            """
            #: Unused - separated by a non-doc comment line
            # Non-doc comment line - not in module.unassignedDocComments()
            A = 1

            #: Unused - overridden by trailing doc comment
            B = 2  #: Trailing doc comment for B overrides preceding doc comment block

            def func():
                #: Unused - not a global assignment
                C = 3
            # Another non-doc comment line - not in module.unassignedDocComments()
            """);

    assertThat(program.getDocCommentsMap().keySet()).containsExactly("B");
    assertThat(program.getDocCommentsMap().get("B").getText())
        .isEqualTo("Trailing doc comment for B overrides preceding doc comment block");
    assertThat(program.getUnusedDocCommentLines().stream().map(Comment::getDocCommentText))
        .containsExactly(
            "Unused - separated by a non-doc comment line",
            "Unused - overridden by trailing doc comment",
            "Unused - not a global assignment")
        .inOrder();
  }
}
