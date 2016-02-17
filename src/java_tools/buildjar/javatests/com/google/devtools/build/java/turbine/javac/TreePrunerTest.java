// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.java.turbine.javac;

import static com.google.common.collect.Iterables.getOnlyElement;
import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.base.Joiner;

import com.sun.tools.javac.file.JavacFileManager;
import com.sun.tools.javac.parser.JavacParser;
import com.sun.tools.javac.parser.ParserFactory;
import com.sun.tools.javac.tree.JCTree.JCClassDecl;
import com.sun.tools.javac.tree.JCTree.JCCompilationUnit;
import com.sun.tools.javac.tree.JCTree.JCVariableDecl;
import com.sun.tools.javac.util.Context;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link TreePruner}. */
@RunWith(JUnit4.class)
public class TreePrunerTest {

  static JCCompilationUnit parseLines(String... lines) {
    Context context = new Context();
    try (JavacFileManager fm = new JavacFileManager(context, true, UTF_8)) {
      ParserFactory parserFactory = ParserFactory.instance(context);
      String input = Joiner.on('\n').join(lines);
      JavacParser parser =
          parserFactory.newParser(
              input, /*keepDocComments=*/ false, /*keepEndPos=*/ false, /*keepLineMap=*/ false);
      return parser.parseCompilationUnit();
    }
  }

  static JCVariableDecl parseField(String line) {
    JCCompilationUnit unit = parseLines("class T {", line, "}");
    JCClassDecl classDecl = (JCClassDecl) getOnlyElement(unit.defs);
    return (JCVariableDecl) getOnlyElement(classDecl.defs);
  }

  static String printPruned(String line) {
    JCVariableDecl tree = parseField(line);
    TreePruner.prune(tree);
    return tree.toString();
  }

  @Test
  public void hello() {
    String[] lines = {
      "class Test {",
      // TODO(cushon): fix google-java-format's handling of lists of string literals
      "  void f() {",
      "    System.err.println(\"hello\");",
      "  }",
      "}",
    };
    JCCompilationUnit tree = parseLines(lines);
    TreePruner.prune(tree);
    String[] expected = {
      "class Test {",
      // TODO(cushon): fix google-java-format's handling of lists of string literals
      "    ",
      "    void f() {",
      "    }",
      "}",
    };
    assertThat(prettyPrint(tree)).isEqualTo(Joiner.on('\n').join(expected));
  }

  @Test
  public void initalizerBlocks() {
    String[] lines = {
      "class Test {",
      "  {",
      "    System.err.println(\"hello\");",
      "  }",
      "  static {",
      "    System.err.println(\"hello\");",
      "  }",
      "}",
    };
    JCCompilationUnit tree = parseLines(lines);
    TreePruner.prune(tree);
    String[] expected = {
      "class Test {",
      // TODO(cushon): fix google-java-format's handling of lists of string literals
      "    {",
      "    }",
      "    static {",
      "    }",
      "}",
    };
    assertThat(prettyPrint(tree)).isEqualTo(Joiner.on('\n').join(expected));
  }

  @Test
  public void nonConstantFieldInitializers() {
    String[] lines = {
      "class Test {",
      // TODO(cushon): fix google-java-format's handling of lists of string literals
      "  Object a = null;",
      "  int[] b = {1};",
      "  int c = g();",
      "  String d = \"\" + h();",
      "}",
    };
    JCCompilationUnit tree = parseLines(lines);
    TreePruner.prune(tree);
    String[] expected = {
      "class Test {",
      // TODO(cushon): fix google-java-format's handling of lists of string literals
      "    Object a;",
      "    int[] b;",
      "    int c;",
      "    String d;",
      "}"
    };
    assertThat(prettyPrint(tree)).isEqualTo(Joiner.on('\n').join(expected));
  }

  // JLS §4.12.4
  @Test
  public void constantVariable() {
    // A constant variable is a final variable
    assertThat(printPruned("final int x = 42;")).isEqualTo("final int x = 42");
    assertThat(printPruned("int x = 42;")).isEqualTo("int x");
    // ... of primitive type or type String
    assertThat(printPruned("final byte x = 0;")).isEqualTo("final byte x = 0");
    assertThat(printPruned("final char x = 0;")).isEqualTo("final char x = 0");
    assertThat(printPruned("final short x = 0;")).isEqualTo("final short x = 0");
    assertThat(printPruned("final int x = 0;")).isEqualTo("final int x = 0");
    assertThat(printPruned("final long x = 0;")).isEqualTo("final long x = 0");
    assertThat(printPruned("final float x = 0;")).isEqualTo("final float x = 0");
    assertThat(printPruned("final double x = 0;")).isEqualTo("final double x = 0");
    assertThat(printPruned("final boolean x = false;")).isEqualTo("final boolean x = false");
    assertThat(printPruned("final String x = \"\";")).isEqualTo("final String x = \"\"");
    assertThat(printPruned("final Object x = 42;")).isEqualTo("final Object x");
    // ... that is initialized with a constant expression (§15.28)
  }

  // JLS §15.28
  @Test
  public void constantExpression() {
    // A constant expression is an expression ... composed using only the following:
    // Literals of primitive type and literals of type String
    // Casts to primitive types and casts to type String
    assertThat(printPruned("final int x = (int)42;")).isEqualTo("final int x = (int)42");
    assertThat(printPruned("final int x = (Integer)42;")).isEqualTo("final int x");
    // The unary operators +, -, ~, and ! (but not ++ or --)
    assertThat(printPruned("final int x = +42;")).isEqualTo("final int x = +42");
    assertThat(printPruned("final int x = -42;")).isEqualTo("final int x = -42");
    assertThat(printPruned("final int x = ~42;")).isEqualTo("final int x = ~42");
    assertThat(printPruned("final boolean x = !false;")).isEqualTo("final boolean x = !false");
    assertThat(printPruned("final int x = ++y;")).isEqualTo("final int x");
    assertThat(printPruned("final int x = --y;")).isEqualTo("final int x");
    assertThat(printPruned("final int x = y++;")).isEqualTo("final int x");
    assertThat(printPruned("final int x = y--;")).isEqualTo("final int x");
    // The multiplicative operators *, /, and %
    assertThat(printPruned("final int x = 1 * 0;")).isEqualTo("final int x = 1 * 0");
    assertThat(printPruned("final int x = 1 / 0;")).isEqualTo("final int x = 1 / 0");
    assertThat(printPruned("final int x = 1 % 0;")).isEqualTo("final int x = 1 % 0");
    // The additive operators + and -
    assertThat(printPruned("final int x = 1 + 0;")).isEqualTo("final int x = 1 + 0");
    assertThat(printPruned("final int x = 1 - 0;")).isEqualTo("final int x = 1 - 0");
    // The shift operators <<, >>, and >>>
    assertThat(printPruned("final int x = 1 << 1;")).isEqualTo("final int x = 1 << 1");
    assertThat(printPruned("final int x = 1 >> 1;")).isEqualTo("final int x = 1 >> 1");
    assertThat(printPruned("final int x = 1 >>> 1;")).isEqualTo("final int x = 1 >>> 1");
    // The relational operators <, <=, >, and >= (but not instanceof)
    assertThat(printPruned("final boolean x = 1 < 1;")).isEqualTo("final boolean x = 1 < 1");
    assertThat(printPruned("final boolean x = 1 <= 1;")).isEqualTo("final boolean x = 1 <= 1");
    assertThat(printPruned("final boolean x = 1 > 1;")).isEqualTo("final boolean x = 1 > 1");
    assertThat(printPruned("final boolean x = 1 >= 1;")).isEqualTo("final boolean x = 1 >= 1");
    assertThat(printPruned("final boolean x = 1 instanceof Integer;")).isEqualTo("final boolean x");
    // The equality operators == and !=
    assertThat(printPruned("final boolean x = 1 == 1;")).isEqualTo("final boolean x = 1 == 1");
    assertThat(printPruned("final boolean x = 1 != 1;")).isEqualTo("final boolean x = 1 != 1");
    // The bitwise and logical operators &, ^, and |
    assertThat(printPruned("final boolean x = 1 & 1;")).isEqualTo("final boolean x = 1 & 1");
    assertThat(printPruned("final boolean x = 1 ^ 1;")).isEqualTo("final boolean x = 1 ^ 1");
    assertThat(printPruned("final boolean x = 1 | 1;")).isEqualTo("final boolean x = 1 | 1");
    // The conditional-and operator && and the conditional-or operator ||
    assertThat(printPruned("final boolean x = true && false;"))
        .isEqualTo("final boolean x = true && false");
    assertThat(printPruned("final boolean x = true || false;"))
        .isEqualTo("final boolean x = true || false");
    // The ternary conditional operator ? :
    assertThat(printPruned("final int x = true ? 1 : 0;")).isEqualTo("final int x = true ? 1 : 0");
    // Parenthesized expressions  whose contained expression is a constant expression
    assertThat(printPruned("final int x = (1);")).isEqualTo("final int x = (1)");
    // Simple names that refer to constant variables
    assertThat(printPruned("final int x = CONST;")).isEqualTo("final int x = CONST");
    // Qualified names of the form TypeName . Identifier that refer to constant variables
    assertThat(printPruned("final int x = TypeName.Identifier;"))
        .isEqualTo("final int x = TypeName.Identifier");
  }

  @Test
  public void nonConstantExpression() {
    // definitely not a constant
    assertThat(printPruned("final int x = g();")).isEqualTo("final int x");
    // assume all qualified identifiers refer to constant variables, but not that all select
    // expressions are qualified identifiers
    assertThat(printPruned("final int x = new X().y;")).isEqualTo("final int x");
  }

  @Test
  public void constructorChaining() {
    String[] lines = {
      "class Test {",
      "  Test() {",
      "    this(42);",
      "    process();",
      "  }",
      "  Test() {",
      "    this(42);",
      "  }",
      "  Test() {",
      "    super(42);",
      "  }",
      "  Test() {}",
      "}",
    };
    JCCompilationUnit tree = parseLines(lines);
    TreePruner.prune(tree);
    String[] expected = {
      "class Test {",
      "    ",
      "    Test() {",
      "        this(42);",
      "    }",
      "    ",
      "    Test() {",
      "        this(42);",
      "    }",
      "    ",
      "    Test() {",
      "        super(42);",
      "    }",
      "    ",
      "    Test() {",
      "    }",
      "}",
    };
    assertThat(prettyPrint(tree)).isEqualTo(Joiner.on('\n').join(expected));
  }

  private String prettyPrint(JCCompilationUnit tree) {
    return tree.toString().trim();
  }
}
