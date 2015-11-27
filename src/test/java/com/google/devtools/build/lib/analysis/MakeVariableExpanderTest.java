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
package com.google.devtools.build.lib.analysis;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.HashMap;
import java.util.Map;

/**
 * Unit tests for the {@link MakeVariableExpander}, which expands variable references of the form
 * <code>"$x"</code> and <code>"$(foo)"</code> into their corresponding values.
 */
@RunWith(JUnit4.class)
public class MakeVariableExpanderTest {

  private MakeVariableExpander.Context context;

  private Map<String, String> vars = new HashMap<>();

  @Before
  public final void createContext() throws Exception  {
    context = new MakeVariableExpander.Context() {
        @Override
        public String lookupMakeVariable(String name)
            throws MakeVariableExpander.ExpansionException {
          // Not a Make variable. Let the shell handle the expansion.
          if (name.startsWith("$")) {
            return name;
          }
          if (!vars.containsKey(name)) {
            throw new MakeVariableExpander.ExpansionException("$(" + name + ") not defined");
          }
          return vars.get(name);
        }
      };

    vars.put("SRCS", "src1 src2");
  }

  private void assertExpansionEquals(String expected, String cmd)
      throws MakeVariableExpander.ExpansionException {
    assertEquals(expected, MakeVariableExpander.expand(cmd, context));
  }

  private void assertExpansionFails(String expectedErrorSuffix, String cmd) {
    try {
      MakeVariableExpander.expand(cmd, context);
      fail("Expansion of " + cmd + " didn't fail as expected");
    } catch (Exception e) {
      assertThat(e).hasMessage(expectedErrorSuffix);
    }
  }

  @Test
  public void testExpansion() throws Exception {
    vars.put("<", "src1");
    vars.put("OUTS", "out1 out2");
    vars.put("@", "out1");
    vars.put("^", "src1 src2 dep1 dep2");
    vars.put("@D", "outdir");
    vars.put("BINDIR", "bindir");

    assertExpansionEquals("src1 src2", "$(SRCS)");
    assertExpansionEquals("src1", "$<");
    assertExpansionEquals("out1 out2", "$(OUTS)");
    assertExpansionEquals("out1", "$(@)");
    assertExpansionEquals("out1", "$@");
    assertExpansionEquals("out1,", "$@,");

    assertExpansionEquals("src1 src2 out1 out2", "$(SRCS) $(OUTS)");

    assertExpansionEquals("cmd", "cmd");
    assertExpansionEquals("cmd src1 src2,", "cmd $(SRCS),");
    assertExpansionEquals("label1 src1 src2,", "label1 $(SRCS),");
    assertExpansionEquals(":label1 src1 src2,", ":label1 $(SRCS),");

    // Note: $(location x) is considered an undefined variable;
    assertExpansionFails("$(location label1) not defined",
                         "$(location label1), $(SRCS),");
  }

  @Test
  public void testRecursiveExpansion() throws Exception {
    // Expansion is recursive: $(recursive) -> $(SRCS) -> "src1 src2"
    vars.put("recursive", "$(SRCS)");
    assertExpansionEquals("src1 src2", "$(recursive)");

    // Recursion does not span expansion boundaries:
    // $(recur2a)$(recur2b) --> "$" + "(SRCS)"  --/--> "src1 src2"
    vars.put("recur2a", "$$");
    vars.put("recur2b", "(SRCS)");
    assertExpansionEquals("$(SRCS)", "$(recur2a)$(recur2b)");
  }

  @Test
  public void testInfiniteRecursionFailsGracefully() throws Exception {
    vars.put("infinite", "$(infinite)");
    assertExpansionFails("potentially unbounded recursion during expansion "
                         + "of '$(infinite)'",
                         "$(infinite)");

    vars.put("black", "$(white)");
    vars.put("white", "$(black)");
    assertExpansionFails("potentially unbounded recursion during expansion "
                         + "of '$(black)'",
                         "$(white) is the new $(black)");
  }

  @Test
  public void testErrors() throws Exception {
    assertExpansionFails("unterminated variable reference", "$(SRCS");
    assertExpansionFails("unterminated $", "$");

    String suffix = "instead for \"Make\" variables, or escape the '$' as '$$' if you intended "
        + "this for the shell";
    assertExpansionFails("'$file' syntax is not supported; use '$(file)' " + suffix,
                         "for file in a b c;do echo $file;done");
    assertExpansionFails("'${file%:.*8}' syntax is not supported; use '$(file%:.*8)' " + suffix,
                         "${file%:.*8}");
  }

  @Test
  public void testShellVariables() throws Exception {
    assertExpansionEquals("for file in a b c;do echo $file;done",
        "for file in a b c;do echo $$file;done");
    assertExpansionEquals("${file%:.*8}", "$${file%:.*8}");
    assertExpansionFails("$(basename file) not defined", "$(basename file)");
    assertExpansionEquals("$(basename file)", "$$(basename file)");
  }
}
