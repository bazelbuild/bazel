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

package com.google.devtools.build.lib.starlarkdocextract;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.extensions.proto.ProtoTruth.assertThat;
import static com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.FunctionParamRole.PARAM_ROLE_KEYWORD_ONLY;
import static com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.FunctionParamRole.PARAM_ROLE_KWARGS;
import static com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.FunctionParamRole.PARAM_ROLE_ORDINARY;
import static com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.FunctionParamRole.PARAM_ROLE_VARARGS;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.cmdline.BazelModuleContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.BzlLoadFunction;
import com.google.devtools.build.lib.starlark.util.BazelEvaluationTestCase;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.FunctionDeprecationInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.FunctionParamInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.FunctionReturnInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.OriginKey;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.StarlarkFunctionInfo;
import net.starlark.java.eval.Module;
import net.starlark.java.eval.StarlarkFunction;
import net.starlark.java.syntax.FileOptions;
import net.starlark.java.syntax.ParserInput;
import net.starlark.java.syntax.Program;
import net.starlark.java.syntax.StarlarkFile;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class StarlarkFunctionInfoExtractorTest {

  private String fakeLabelString = null; // set by exec()

  /**
   * Executes the given Starlark code and returns the value of the first global variable, which must
   * be a function.
   */
  private StarlarkFunction exec(String... lines) throws Exception {
    BazelEvaluationTestCase ev = new BazelEvaluationTestCase();
    Module module = ev.getModule();
    Label fakeLabel = BazelModuleContext.of(module).label();
    fakeLabelString = fakeLabel.getCanonicalForm();
    ParserInput input = ParserInput.fromLines(lines);
    StarlarkFile file = StarlarkFile.parse(input, FileOptions.DEFAULT);
    Program program = Program.compileFile(file, module);
    BzlLoadFunction.execAndExport(
        program, fakeLabel, ev.getEventHandler(), module, ev.getStarlarkThread());
    return (StarlarkFunction) ev.getModule().getGlobals().values().stream().findFirst().get();
  }

  @Test
  public void basicFunctionality() throws Exception {
    StarlarkFunction fn =
        exec(
            """
            def fn(x):
                pass
            """);
    StarlarkFunctionInfo info =
        StarlarkFunctionInfoExtractor.fromNameAndFunction(
            "namespace.fn", fn, LabelRenderer.DEFAULT);

    assertThat(info)
        .isEqualTo(
            StarlarkFunctionInfo.newBuilder()
                .setFunctionName("namespace.fn")
                .addParameter(
                    FunctionParamInfo.newBuilder()
                        .setName("x")
                        .setRole(PARAM_ROLE_ORDINARY)
                        .setMandatory(true)
                        .build())
                .setOriginKey(OriginKey.newBuilder().setName("fn").setFile(fakeLabelString).build())
                .build());
  }

  @Test
  public void summary_canStartOnFirstOrSecondLine() throws Exception {
    StarlarkFunction fn1 =
        exec(
            """
            def fn1(x):
                "Summary."
                pass
            """);
    StarlarkFunction fn2 =
        exec(
            """
            def fn2(x):
                '''
                Summary.
                '''
                pass
            """);
    StarlarkFunctionInfo info1 =
        StarlarkFunctionInfoExtractor.fromNameAndFunction("fn", fn1, LabelRenderer.DEFAULT);
    StarlarkFunctionInfo info2 =
        StarlarkFunctionInfoExtractor.fromNameAndFunction("fn", fn2, LabelRenderer.DEFAULT);

    assertThat(info1.getDocString()).isEqualTo("Summary.");
    assertThat(info2.getDocString()).isEqualTo("Summary.");
  }

  @Test
  public void summary_mustBeFollowedByBlankLine() throws Exception {
    StarlarkFunction good =
        exec(
            """
            def good(x):
                '''
                Summary.

                Details.'''
                pass
            """);
    StarlarkFunction badNoBlankLine =
        exec(
            """
            def bad_no_blank_line(x):
                '''
                Summary.
                Details.
                '''
                pass
            """);

    assertThat(
            StarlarkFunctionInfoExtractor.fromNameAndFunction("good", good, LabelRenderer.DEFAULT)
                .getDocString())
        .isEqualTo("Summary.\n\nDetails.");
    ExtractionException noBlankLineException =
        assertThrows(
            ExtractionException.class,
            () ->
                StarlarkFunctionInfoExtractor.fromNameAndFunction(
                    "bad_no_blank_line", badNoBlankLine, LabelRenderer.DEFAULT));
    assertThat(noBlankLineException)
        .hasMessageThat()
        .contains("the one-line summary should be followed by a blank line");
  }

  @Test
  public void keywordOnly() throws Exception {
    StarlarkFunction fn =
        exec(
            """
            def fn(a, b=1, *, c, d=2):
                '''This function does stuff.

                Args:
                  a: A value.
                  b: B value
                  c: C value.
                  d: D value.
                '''
                pass
            """);
    StarlarkFunctionInfo info =
        StarlarkFunctionInfoExtractor.fromNameAndFunction("fn", fn, LabelRenderer.DEFAULT);
    assertThat(info.getParameterList())
        .containsExactly(
            FunctionParamInfo.newBuilder()
                .setName("a")
                .setRole(PARAM_ROLE_ORDINARY)
                .setDocString("A value.")
                .setMandatory(true)
                .build(),
            FunctionParamInfo.newBuilder()
                .setName("b")
                .setRole(PARAM_ROLE_ORDINARY)
                .setDocString("B value")
                .setMandatory(false)
                .setDefaultValue("1")
                .build(),
            FunctionParamInfo.newBuilder()
                .setName("c")
                .setRole(PARAM_ROLE_KEYWORD_ONLY)
                .setDocString("C value.")
                .setMandatory(true)
                .build(),
            FunctionParamInfo.newBuilder()
                .setName("d")
                .setRole(PARAM_ROLE_KEYWORD_ONLY)
                .setDocString("D value.")
                .setMandatory(false)
                .setDefaultValue("2")
                .build())
        .inOrder();
  }

  @Test
  public void keywordOnly_withVarargs() throws Exception {
    StarlarkFunction fn =
        exec(
            """
            def fn(a, b=1, *args, c, d=2):
                '''This function does stuff.

                Args:
                  a: A value.
                  b: B value
                  c: C value.
                  d: D value.
                  *args: Remaining positional arguments.
                '''
                pass
            """);
    StarlarkFunctionInfo info =
        StarlarkFunctionInfoExtractor.fromNameAndFunction("fn", fn, LabelRenderer.DEFAULT);
    assertThat(info.getParameterList())
        .containsExactly(
            FunctionParamInfo.newBuilder()
                .setName("a")
                .setRole(PARAM_ROLE_ORDINARY)
                .setDocString("A value.")
                .setMandatory(true)
                .build(),
            FunctionParamInfo.newBuilder()
                .setName("b")
                .setRole(PARAM_ROLE_ORDINARY)
                .setDocString("B value")
                .setMandatory(false)
                .setDefaultValue("1")
                .build(),
            FunctionParamInfo.newBuilder()
                .setName("c")
                .setRole(PARAM_ROLE_KEYWORD_ONLY)
                .setDocString("C value.")
                .setMandatory(true)
                .build(),
            FunctionParamInfo.newBuilder()
                .setName("d")
                .setRole(PARAM_ROLE_KEYWORD_ONLY)
                .setDocString("D value.")
                .setMandatory(false)
                .setDefaultValue("2")
                .build(),
            FunctionParamInfo.newBuilder()
                .setName("args")
                .setRole(PARAM_ROLE_VARARGS)
                .setDocString("Remaining positional arguments.")
                .setMandatory(false)
                .build())
        .inOrder();
  }

  @Test
  public void keywordOnly_withVarargsAndKwargs() throws Exception {
    StarlarkFunction fn =
        exec(
            """
            def fn(a, b=1, *args, c, d=2, **kwargs):
                '''This function does stuff.

                Args:
                  a: A value.
                  b: B value
                  c: C value.
                  d: D value.
                '''
                pass
            """);
    StarlarkFunctionInfo info =
        StarlarkFunctionInfoExtractor.fromNameAndFunction("fn", fn, LabelRenderer.DEFAULT);
    assertThat(info.getParameterList())
        .containsExactly(
            FunctionParamInfo.newBuilder()
                .setName("a")
                .setRole(PARAM_ROLE_ORDINARY)
                .setDocString("A value.")
                .setMandatory(true)
                .build(),
            FunctionParamInfo.newBuilder()
                .setName("b")
                .setRole(PARAM_ROLE_ORDINARY)
                .setDocString("B value")
                .setMandatory(false)
                .setDefaultValue("1")
                .build(),
            FunctionParamInfo.newBuilder()
                .setName("c")
                .setRole(PARAM_ROLE_KEYWORD_ONLY)
                .setDocString("C value.")
                .setMandatory(true)
                .build(),
            FunctionParamInfo.newBuilder()
                .setName("d")
                .setRole(PARAM_ROLE_KEYWORD_ONLY)
                .setDocString("D value.")
                .setMandatory(false)
                .setDefaultValue("2")
                .build(),
            FunctionParamInfo.newBuilder()
                .setName("args")
                .setRole(PARAM_ROLE_VARARGS)
                .setMandatory(false)
                .build(),
            FunctionParamInfo.newBuilder()
                .setName("kwargs")
                .setRole(PARAM_ROLE_KWARGS)
                .setMandatory(false)
                .build())
        .inOrder();
  }

  @Test
  public void returns() throws Exception {
    StarlarkFunction fn =
        exec(
            """
            def fn(x):
                '''
                My function.

                Returns:
                  The value of x.
                '''
                return x
            """);
    StarlarkFunctionInfo info =
        StarlarkFunctionInfoExtractor.fromNameAndFunction("fn", fn, LabelRenderer.DEFAULT);
    assertThat(info.getReturn())
        .isEqualTo(FunctionReturnInfo.newBuilder().setDocString("The value of x.").build());
  }

  @Test
  public void deprecation() throws Exception {
    StarlarkFunction fn =
        exec(
            """
            def fn(x):
                '''
                My function.

                Deprecated:
                  Do not use.
                  Use something else instead.
                '''
                pass
            """);
    StarlarkFunctionInfo info =
        StarlarkFunctionInfoExtractor.fromNameAndFunction("fn", fn, LabelRenderer.DEFAULT);
    assertThat(info.getDeprecated())
        .isEqualTo(
            FunctionDeprecationInfo.newBuilder()
                .setDocString("Do not use.\nUse something else instead.")
                .build());
  }

  @Test
  public void specialSections_canBeSeparatedByAnyNumberOfBlankLines() throws Exception {
    String extraBlankLines = "";
    for (int i = 0; i < 2; i++, extraBlankLines += "\n") {
      StarlarkFunction fn =
          exec(
              String.format(
                  """
                  def fn%d(x):
                      '''
                      My function.

                      Args:
                        x: X value.%s
                      Returns:
                        The value of x.%s
                      Deprecated:
                        Do not use.
                      '''
                      return x
                  """,
                  i, extraBlankLines, extraBlankLines));
      StarlarkFunctionInfo info =
          StarlarkFunctionInfoExtractor.fromNameAndFunction("fn" + i, fn, LabelRenderer.DEFAULT);
      assertThat(info)
          .isEqualTo(
              StarlarkFunctionInfo.newBuilder()
                  .setFunctionName("fn" + i)
                  .setDocString("My function.")
                  .addParameter(
                      FunctionParamInfo.newBuilder()
                          .setName("x")
                          .setRole(PARAM_ROLE_ORDINARY)
                          .setDocString("X value.")
                          .setMandatory(true)
                          .build())
                  .setReturn(
                      FunctionReturnInfo.newBuilder().setDocString("The value of x.").build())
                  .setDeprecated(
                      FunctionDeprecationInfo.newBuilder().setDocString("Do not use.").build())
                  .setOriginKey(
                      OriginKey.newBuilder().setName("fn" + i).setFile(fakeLabelString).build())
                  .build());
    }
  }
}
