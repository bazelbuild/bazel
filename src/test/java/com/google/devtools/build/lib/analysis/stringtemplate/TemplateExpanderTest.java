// Copyright 2006 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.stringtemplate;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableSet;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Unit tests for the {@link TemplateExpander}.
 */
@RunWith(JUnit4.class)
public class TemplateExpanderTest {
  private static final class TemplateContextImpl implements TemplateContext {
    private final Map<String, String> vars = new HashMap<>();
    private final Map<String, Function<String, String>> functions = new HashMap<>();

    @Override
    public String lookupVariable(String name)
        throws ExpansionException {
      // Not a Make variable. Let the shell handle the expansion.
      if (name.startsWith("$")) {
        return name;
      }
      if (!vars.containsKey(name)) {
        throw new ExpansionException(String.format("$(%s) not defined", name));
      }
      return vars.get(name);
    }

    @Override
    public String lookupFunction(String name, String param) throws ExpansionException {
      if (!functions.containsKey(name)) {
        throw new ExpansionException(String.format("$(%s) not defined", name));
      }
      return functions.get(name).apply(param);
    }
  }

  private TemplateContextImpl context;

  @Before
  public final void createContext() throws Exception  {
    context = new TemplateContextImpl();
  }

  private Expansion expand(String value) throws ExpansionException, InterruptedException {
    return TemplateExpander.expand(value, context);
  }

  private String expandSingleVariable(String value)
      throws ExpansionException, InterruptedException {
    return TemplateExpander.expandSingleVariable(value, context);
  }

  private ExpansionException expansionFailure(String cmd) throws InterruptedException {
    try {
      expand(cmd);
      fail("Expansion of " + cmd + " didn't fail as expected");
      throw new AssertionError();
    } catch (ExpansionException e) {
      return e;
    }
  }

  @Test
  public void testVariableExpansion() throws Exception {
    context.vars.put("SRCS", "src1 src2");
    context.vars.put("<", "src1");
    context.vars.put("OUTS", "out1 out2");
    context.vars.put("@", "out1");
    context.vars.put("^", "src1 src2 dep1 dep2");
    context.vars.put("@D", "outdir");
    context.vars.put("BINDIR", "bindir");
    context.vars.put("CUSTOMVAR", "custom val");

    assertThat(expand("$(SRCS)")).isEqualTo(Expansion.create("src1 src2", ImmutableSet.of("SRCS")));
    assertThat(expand("$<")).isEqualTo(Expansion.create("src1", ImmutableSet.of("<")));
    assertThat(expand("$(OUTS)")).isEqualTo(Expansion.create("out1 out2", ImmutableSet.of("OUTS")));
    assertThat(expand("$(@)")).isEqualTo(Expansion.create("out1", ImmutableSet.of("@")));
    assertThat(expand("$@")).isEqualTo(Expansion.create("out1", ImmutableSet.of("@")));
    assertThat(expand("$@,")).isEqualTo(Expansion.create("out1,", ImmutableSet.of("@")));
    assertThat(expand("$(CUSTOMVAR)"))
        .isEqualTo(Expansion.create("custom val", ImmutableSet.of("CUSTOMVAR")));

    assertThat(expand("$(SRCS) $(OUTS)"))
        .isEqualTo(Expansion.create("src1 src2 out1 out2", ImmutableSet.of("SRCS", "OUTS")));

    assertThat(expand("cmd")).isEqualTo(Expansion.create("cmd", ImmutableSet.of()));
    assertThat(expand("cmd $(SRCS),"))
        .isEqualTo(Expansion.create("cmd src1 src2,", ImmutableSet.of("SRCS")));
    assertThat(expand("label1 $(SRCS),"))
        .isEqualTo(Expansion.create("label1 src1 src2,", ImmutableSet.of("SRCS")));
    assertThat(expand(":label1 $(SRCS),"))
        .isEqualTo(Expansion.create(":label1 src1 src2,", ImmutableSet.of("SRCS")));
  }

  @Test
  public void testUndefinedVariableExpansion() throws Exception {
    assertThat(expansionFailure("$(foo)"))
        .hasMessageThat().isEqualTo("$(foo) not defined");
  }

  @Test
  public void testFunctionExpansion() throws Exception {
    context.functions.put("foo", (String p) -> "FOO(" + p + ")");
    context.vars.put("bar", "bar");

    assertThat(expand("$(foo baz)"))
        .isEqualTo(Expansion.create("FOO(baz)", ImmutableSet.of("foo")));
    assertThat(expand("$(bar) $(foo baz)"))
        .isEqualTo(Expansion.create("bar FOO(baz)", ImmutableSet.of("bar", "foo")));
    assertThat(expand("xyz$(foo baz)zyx"))
        .isEqualTo(Expansion.create("xyzFOO(baz)zyx", ImmutableSet.of("foo")));
  }

  @Test
  public void testFunctionExpansionThrows() throws Exception {
    ExpansionException e =
        assertThrows(
            ExpansionException.class,
            () ->
                TemplateExpander.expand(
                    "$(foo baz)",
                    new TemplateContext() {
                      @Override
                      public String lookupVariable(String name) throws ExpansionException {
                        throw new ExpansionException(name);
                      }

                      @Override
                      public String lookupFunction(String name, String param)
                          throws ExpansionException {
                        throw new ExpansionException(name + "(" + param + ")");
                      }
                    }));
    assertThat(e).hasMessageThat().isEqualTo("foo(baz)");
  }

  @Test
  public void testUndefinedFunctionExpansion() throws Exception {
    // Note: $(location x) is considered an undefined variable;
    assertThat(expansionFailure("$(location label1), $(SRCS),"))
        .hasMessageThat().isEqualTo("$(location) not defined");
    assertThat(expansionFailure("$(basename file)"))
        .hasMessageThat().isEqualTo("$(basename) not defined");
  }

  @Test
  public void testRecursiveExpansion() throws Exception {
    // Expansion is recursive: $(recursive) -> $(SRCS) -> "src1 src2"
    context.vars.put("SRCS", "src1 src2");
    context.vars.put("recursive", "$(SRCS)");
    assertThat(expand("$(recursive)"))
        .isEqualTo(Expansion.create("src1 src2", ImmutableSet.of("recursive", "SRCS")));
  }

  @Test
  public void testRecursiveExpansionDoesNotSpanExpansionBoundaries() throws Exception {
    // Recursion does not span expansion boundaries:
    // $(recur2a)$(recur2b) --> "$" + "(SRCS)"  --/--> "src1 src2"
    context.vars.put("SRCS", "src1 src2");
    context.vars.put("recur2a", "$$");
    context.vars.put("recur2b", "(SRCS)");
    assertThat(expand("$(recur2a)$(recur2b)"))
        .isEqualTo(Expansion.create("$(SRCS)", ImmutableSet.of("recur2a", "recur2b")));
  }

  @Test
  public void testSelfInfiniteExpansionFailsGracefully() throws Exception {
    context.vars.put("infinite", "$(infinite)");
    assertThat(expansionFailure("$(infinite)")).hasMessageThat()
        .isEqualTo("potentially unbounded recursion during expansion of '$(infinite)'");
  }

  @Test
  public void testMutuallyInfiniteExpansionFailsGracefully() throws Exception {
    context.vars.put("black", "$(white)");
    context.vars.put("white", "$(black)");
    assertThat(expansionFailure("$(white) is the new $(black)")).hasMessageThat()
        .isEqualTo("potentially unbounded recursion during expansion of '$(black)'");
  }

  @Test
  public void testErrors() throws Exception {
    assertThat(expansionFailure("$(SRCS")).hasMessageThat()
        .isEqualTo("unterminated variable reference");
    assertThat(expansionFailure("$")).hasMessageThat().isEqualTo("unterminated $");

    String suffix = "instead for \"Make\" variables, or escape the '$' as '$$' if you intended "
        + "this for the shell";
    assertThat(expansionFailure("for file in a b c;do echo $file;done")).hasMessageThat()
        .isEqualTo("'$file' syntax is not supported; use '$(file)' " + suffix);
    assertThat(expansionFailure("${file%:.*8}")).hasMessageThat()
        .isEqualTo("'${file%:.*8}' syntax is not supported; use '$(file%:.*8)' " + suffix);
  }

  @Test
  public void testDollarDollar() throws Exception {
    assertThat(expand("for file in a b c;do echo $$file;done"))
        .isEqualTo(Expansion.create("for file in a b c;do echo $file;done", ImmutableSet.of()));
    assertThat(expand("$${file%:.*8}"))
        .isEqualTo(Expansion.create("${file%:.*8}", ImmutableSet.of()));
    assertThat(expand("$$(basename file)"))
        .isEqualTo(Expansion.create("$(basename file)", ImmutableSet.of()));
  }

  // Regression test: check that the parameter is trimmed before expanding.
  @Test
  public void testFunctionExpansionIsTrimmed() throws Exception {
    context.functions.put("foo", (String p) -> "FOO(" + p + ")");
    assertThat(expand("$(foo  baz )"))
        .isEqualTo(Expansion.create("FOO(baz)", ImmutableSet.of("foo")));
  }

  @Test
  public void testExpandSingleVariable() throws Exception {
    context.vars.put("SINGLE", "val1 val2");
    assertThat(expandSingleVariable("$(SINGLE)")).isEqualTo("val1 val2");
    assertThat(expandSingleVariable("foo $(SINGLE)")).isNull();
  }
}
