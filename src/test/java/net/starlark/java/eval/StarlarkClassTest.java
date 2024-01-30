// Copyright 2023 The Bazel Authors. All rights reserved.
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

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for static utility methods in the {@link Starlark} class. */
@RunWith(JUnit4.class)
public final class StarlarkClassTest {

  @Test
  public void trimDocString() {
    // See https://peps.python.org/pep-0257/#handling-docstring-indentation
    // Single line
    assertThat(Starlark.trimDocString("")).isEmpty();
    assertThat(Starlark.trimDocString("   ")).isEmpty();
    assertThat(Starlark.trimDocString("\t\t\t")).isEmpty();
    assertThat(Starlark.trimDocString("Hello world")).isEqualTo("Hello world");
    assertThat(Starlark.trimDocString("   Hello world   ")).isEqualTo("Hello world");
    // First line is always fully trimmed, regardless of subsequent indentation levels.
    assertThat(Starlark.trimDocString("   Hello\t\nworld")).isEqualTo("Hello\nworld");
    assertThat(Starlark.trimDocString("   Hello    \n  world")).isEqualTo("Hello\nworld");
    assertThat(Starlark.trimDocString(" Hello  \n    world")).isEqualTo("Hello\nworld");
    // Subsequent lines are dedented to their minimal indentation level and fully right-trimmed
    assertThat(Starlark.trimDocString("   Hello    \n   world \n  and  \n     good-bye     "))
        .isEqualTo("Hello\n world\nand\n   good-bye");
    // ... and the first line's indentation does not affect minimal indentation level computation.
    assertThat(Starlark.trimDocString(" Hello\n    world\n    and\n  good-bye"))
        .isEqualTo("Hello\n  world\n  and\ngood-bye");
    // Blank lines are trimmed and do not affect indentation level computation
    assertThat(
            Starlark.trimDocString(
                "   Hello    \n\n   world \n\n\n \n  and  \n   \n     good-bye     "))
        .isEqualTo("Hello\n\n world\n\n\n\nand\n\n   good-bye");
    // Windows-style \r\n is simplified to \n
    assertThat(Starlark.trimDocString("Hello\r\nworld")).isEqualTo("Hello\nworld");
    assertThat(Starlark.trimDocString("Hello\r\n\r\nworld")).isEqualTo("Hello\n\nworld");
    // Leading and trailing blank lines are removed
    assertThat(Starlark.trimDocString("\n\n\n")).isEmpty();
    assertThat(Starlark.trimDocString("\r\n\r\n\r\n")).isEmpty();
    assertThat(Starlark.trimDocString("\n \n  \n   ")).isEmpty();
    assertThat(Starlark.trimDocString("\n \r\n  \r\n   ")).isEmpty();
    assertThat(Starlark.trimDocString("\n\r\nHello world\n\r\n")).isEqualTo("Hello world");
    assertThat(Starlark.trimDocString("\n\n  \nHello\nworld\n\n \n\t")).isEqualTo("Hello\nworld");
    assertThat(Starlark.trimDocString("\n\n  \t\nHello\n  world\n \n")).isEqualTo("Hello\n  world");
    // Tabs are expanded to size 8 (following Python convention)
    assertThat(Starlark.trimDocString("Hello\tworld")).isEqualTo("Hello   world");
    assertThat(Starlark.trimDocString("\n\tHello\n\t\tworld")).isEqualTo("Hello\n        world");
    assertThat(Starlark.trimDocString("\n   \tHello\n\t\tworld")).isEqualTo("Hello\n        world");
    assertThat(Starlark.trimDocString("\n   Hello\n\tworld")).isEqualTo("Hello\n     world");
  }

  @Test
  public void expandTabs() {
    assertThat(Starlark.expandTabs("", 8)).isEmpty();
    assertThat(Starlark.expandTabs("Hello\nworld", 8)).isEqualTo("Hello\nworld");

    assertThat(Starlark.expandTabs("\t", 1)).isEqualTo(" ");
    assertThat(Starlark.expandTabs("\t", 2)).isEqualTo("  ");
    assertThat(Starlark.expandTabs(" \t", 2)).isEqualTo("  ");
    assertThat(Starlark.expandTabs("\t", 8)).isEqualTo("        ");
    assertThat(Starlark.expandTabs("     \t", 8)).isEqualTo("        ");

    assertThat(Starlark.expandTabs("01\t012\t0123\t01234", 4)).isEqualTo("01  012 0123    01234");
    assertThat(Starlark.expandTabs("01\t012\t0123\t01234", 8))
        .isEqualTo("01      012     0123    01234");

    assertThat(Starlark.expandTabs("01\t\n\t012\t0123\t01234", 4))
        .isEqualTo("01  \n    012 0123    01234");
    assertThat(Starlark.expandTabs("\r01\r\n\t012\r\n\t0123\t01234\n", 8))
        .isEqualTo("\r01\r\n        012\r\n        0123    01234\n");

    assertThrows(IllegalArgumentException.class, () -> Starlark.expandTabs("\t", 0));
    assertThrows(IllegalArgumentException.class, () -> Starlark.expandTabs("\t", -1));
  }
}
