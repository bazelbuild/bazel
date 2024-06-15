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

import static com.google.common.truth.extensions.proto.ProtoTruth.assertThat;
import static com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.FunctionParamRole.PARAM_ROLE_KEYWORD_ONLY;
import static com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.FunctionParamRole.PARAM_ROLE_KWARGS;
import static com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.FunctionParamRole.PARAM_ROLE_ORDINARY;
import static com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.FunctionParamRole.PARAM_ROLE_VARARGS;

import com.google.devtools.build.lib.cmdline.BazelModuleContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.BzlLoadFunction;
import com.google.devtools.build.lib.starlark.util.BazelEvaluationTestCase;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.FunctionParamInfo;
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
}
