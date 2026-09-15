// Copyright 2015 The Bazel Authors. All rights reserved.
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

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@code NodeVisitor} */
@RunWith(JUnit4.class)
public final class NodeVisitorTest {

  Supplier<IdentGatherer> gathererFactory = IdentGatherer::new;

  FileOptions fileOptions = FileOptions.builder().allowTypeSyntax(true).build();

  private StarlarkFile parse(String... lines) throws SyntaxError.Exception {
    ParserInput input = ParserInput.fromLines(lines);
    StarlarkFile file = StarlarkFile.parse(input, fileOptions);
    if (!file.ok()) {
      throw new SyntaxError.Exception(file.errors());
    }
    return file;
  }

  /** Records all identifiers in the order they were seen, including duplicates. */
  private static class IdentGatherer extends NodeVisitor {
    final List<String> idents = new ArrayList<>();

    static IdentGatherer skippingNonSymbolIdentifiers() {
      IdentGatherer gatherer = new IdentGatherer();
      gatherer.skipNonSymbolIdentifiers = true;
      return gatherer;
    }

    @Override
    public void visit(Identifier node) {
      idents.add(node.getName());
    }
  }

  /**
   * Asserts that the traversed identifiers (in order, including duplicates) of the given source
   * code match the expected identifiers, which is supplied as a space-delimited string.
   */
  public void assertIdentsAre(String src, String expectedIdents) throws Exception {
    StarlarkFile file = parse(src);
    IdentGatherer visitor = gathererFactory.get();
    visitor.visit(file);
    assertThat(visitor.idents).containsExactlyElementsIn(expectedIdents.split(" ")).inOrder();
  }

  @Test
  public void blockAndSimpleStatements() throws Exception {
    assertIdentsAre(
        """
        load("...", "a", b="c")
        d = e
        f
        pass
        g, h[i] = j + 1 + "xyz" + 0.0
        """,
        // "c" is omitted because we don't currently visit the original name in a load binding.
        "a b d e f g h i j");
  }

  @Test
  public void controlStatements() throws Exception {
    assertIdentsAre(
        """
        for a in b:
          if c:
            break
          else:
            continue
        """,
        "a b c");
  }

  @Test
  public void simpleExpressions() throws Exception {
    assertIdentsAre(
        """
        a + b if c else d.e
        {f: g, h: [i, j]}
        not k[l:m]
        """,
        "a b c d e f g h i j k l m");
  }

  @Test
  public void comprehensions() throws Exception {
    assertIdentsAre(
        """
        [a for b, c in d if e for f in {g: h for i in j}]
        """,
        "a b c d e f g h i j");
  }

  @Test
  public void calls() throws Exception {
    assertIdentsAre(
        """
        a(b, c=d, *e, **f)
        """,
        "a b c d e f");
  }

  @Test
  public void functionDefs() throws Exception {
    assertIdentsAre(
        """
        def a(b, c=d, *e, **f):
          g
        """,
        "a b c d e f g");
    assertIdentsAre(
        """
        def a(*, b, c=d):
          return
        """,
        "a b c d");
  }

  @Test
  public void typeAnnotations() throws Exception {
    assertIdentsAre(
        """
        def a[b, c](d : e[f], g: h) -> i:
          pass
        """,
        "a b c d e f g h i");

    assertIdentsAre(
        """
        a : b
        c : d = e
        """,
        "a b c d e");
  }

  @Test
  public void typeDeclarations() throws Exception {
    assertIdentsAre(
        """
        type a[b, c] = d
        """,
        "a b c d");
  }

  @Test
  public void ellipsis() throws Exception {
    // Really, this is just a test that we defined NodeVisitor#visit(Ellipsis) to exist...
    fileOptions =
        FileOptions.builder().allowTypeSyntax(true).tolerateInvalidTypeExpressions(true).build();
    assertIdentsAre(
        """
        def a() -> ...:
          pass
        """,
        "a");
  }

  @Test
  public void typeOperations() throws Exception {
    assertIdentsAre(
        """
        cast(a, b)
        isinstance(c, d)
        """,
        "a b c d");
  }

  @Test
  public void skipNonSymbolIdentifiers() throws Exception {
    gathererFactory = IdentGatherer::skippingNonSymbolIdentifiers;

    assertIdentsAre(
        """
        load("...", a="b")
        c(d=e.f)
        """,
        // No b, no d, no f.
        "a c e");
    assertIdentsAre(
        """
        def a(b, c=d):
          pass
        """,
        // Keyword param identifiers ("c") are still visited.
        "a b c d");
  }
}
