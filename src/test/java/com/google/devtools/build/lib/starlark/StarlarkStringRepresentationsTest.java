// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.starlark;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.packages.StarlarkInfo;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for string representations of Starlark objects. */
@RunWith(JUnit4.class)
public class StarlarkStringRepresentationsTest extends BuildViewTestCase {

  // Different ways to format objects, these suffixes are used in the `prepare_params` function
  private static final ImmutableList<String> SUFFIXES =
      ImmutableList.of("_str", "_repr", "_format", "_str_perc", "_repr_perc");

  private Object starlarkLoadingEval(String code) throws Exception {
    return starlarkLoadingEval(code, "");
  }

  /**
   * Evaluates {@code code} in the loading phase in a .bzl file
   *
   * @param code The code to execute
   * @param definition Additional code to define necessary variables
   */
  private Object starlarkLoadingEval(String code, String definition) throws Exception {
    scratch.overwriteFile(
        "eval/BUILD",
        """
        load(":eval.bzl", "eval")

        eval(name = "eval")
        """);
    scratch.overwriteFile(
        "eval/eval.bzl",
        definition,
        String.format("x = %s", code), // Should be placed here to execute during the loading phase
        "Info = provider()",
        "def _impl(ctx):",
        "  return Info(result = x)",
        "eval = rule(implementation = _impl)");
    skyframeExecutor.invalidateFilesUnderPathForTesting(
        reporter,
        new ModifiedFileSet.Builder()
            .modify(PathFragment.create("eval/BUILD"))
            .modify(PathFragment.create("eval/eval.bzl"))
            .build(),
        Root.fromPath(rootDirectory));

    ConfiguredTarget target = getConfiguredTarget("//eval");
    return getStarlarkProvider(target, "Info").getValue("result");
  }

  /**
   * Evaluates {@code code} in the loading phase in a BUILD file. {@code code} must return a string.
   *
   * @param code The code to execute
   */
  private Object starlarkLoadingEvalInBuildFile(String code) throws Exception {
    scratch.overwriteFile("eval/BUILD",
        "load(':eval.bzl', 'eval')",
        String.format("eval(name='eval', param = %s)", code));
    scratch.overwriteFile(
        "eval/eval.bzl",
        """
        Info = provider()
        def _impl(ctx):
            return Info(result = ctx.attr.param)

        eval = rule(implementation = _impl, attrs = {"param": attr.string()})
        """);
    skyframeExecutor.invalidateFilesUnderPathForTesting(
        reporter,
        new ModifiedFileSet.Builder()
            .modify(PathFragment.create("eval/BUILD"))
            .modify(PathFragment.create("eval/eval.bzl"))
            .build(),
        Root.fromPath(rootDirectory));

    ConfiguredTarget target = getConfiguredTarget("//eval");
    return getStarlarkProvider(target, "Info").getValue("result");
  }

  /**
   * Asserts that all 5 different ways to convert an object to a string of {@code expression}
   * ({@code str}, {@code repr}, {@code '%s'}, {@code '%r'}, {@code '{}'.format} return the correct
   * {@code representation}. Not applicable for objects that have different {@code str} and {@code
   * repr} representations.
   *
   * @param expression the expression to evaluate a string representation of
   * @param representation desired string representation
   */
  private void assertStringRepresentationInBuildFile(
      String expression, String representation) throws Exception {
    assertThat(starlarkLoadingEvalInBuildFile(String.format("str(%s)", expression)))
        .isEqualTo(representation);
    assertThat(starlarkLoadingEvalInBuildFile(String.format("repr(%s)", expression)))
        .isEqualTo(representation);
    assertThat(starlarkLoadingEvalInBuildFile(String.format("'%%s' %% (%s,)", expression)))
        .isEqualTo(representation);
    assertThat(starlarkLoadingEvalInBuildFile(String.format("'%%r' %% (%s,)", expression)))
        .isEqualTo(representation);
    assertThat(starlarkLoadingEvalInBuildFile(String.format("'{}'.format(%s)", expression)))
        .isEqualTo(representation);
  }

  /**
   * Asserts that all 5 different ways to convert an object to a string of {@code expression}
   * ({@code str}, {@code repr}, {@code '%s'}, {@code '%r'}, {@code '{}'.format} return the correct
   * {@code representation}. Not applicable for objects that have different {@code str} and {@code
   * repr} representations.
   *
   * @param definition optional definition required to evaluate the {@code expression}
   * @param expression the expression to evaluate a string representation of
   * @param representation desired string representation
   */
  private void assertStringRepresentation(
      String definition, String expression, String representation) throws Exception {
    assertThat(starlarkLoadingEval(String.format("str(%s)", expression), definition))
        .isEqualTo(representation);
    assertThat(starlarkLoadingEval(String.format("repr(%s)", expression), definition))
        .isEqualTo(representation);
    assertThat(starlarkLoadingEval(String.format("'%%s' %% (%s,)", expression), definition))
        .isEqualTo(representation);
    assertThat(starlarkLoadingEval(String.format("'%%r' %% (%s,)", expression), definition))
        .isEqualTo(representation);
    assertThat(starlarkLoadingEval(String.format("'{}'.format(%s)", expression), definition))
        .isEqualTo(representation);
  }

  private void assertStringRepresentation(String expression, String representation)
      throws Exception {
    assertStringRepresentation("", expression, representation);
  }

  /**
   * Creates a set of BUILD and .bzl files that gathers objects of many different types available in
   * Starlark and creates their string representations by calling `str` and `repr` on them. The
   * strings are available in the configured target for //test/starlark:check
   */
  private void generateFilesToTestStrings() throws Exception {
    // Generate string representations of Starlark rule contexts, targets, and files.
    // Objects are gathered in the implementation of the `check` rule.
    // prepare_params(objects) converts a dict of objects to a dict of their string representations.

    scratch.file(
        "test/starlark/rules.bzl",
        """
        aspect_ctx_provider = provider()

        def prepare_params(objects):
            params = {}
            for k, v in objects.items():
                params[k + "_str"] = str(v)
                params[k + "_repr"] = repr(v)
                params[k + "_format"] = "{}".format(v)
                params[k + "_str_perc"] = "%s" % (v,)
                params[k + "_repr_perc"] = "%r" % (v,)
            return params

        def _impl_aspect(target, ctx):
            return [aspect_ctx_provider(ctx = ctx, rule = ctx.rule)]

        my_aspect = aspect(implementation = _impl_aspect)

        def _impl(ctx):
            pass

        dep = rule(implementation = _impl)

        def _genfile_impl(ctx):
            ctx.actions.write(output = ctx.outputs.my_output, content = "foo")

        genfile = rule(
            implementation = _genfile_impl,
            outputs = {"my_output": "%{name}.txt"},
        )
        CheckInfo = provider()
        def _check_impl(ctx):
            source_file = ctx.attr.srcs[0].files.to_list()[0]
            generated_file = ctx.attr.srcs[1].files.to_list()[0]
            objects = {
                "target": ctx.attr.deps[0],
                "alias_target": ctx.attr.deps[1],
                "aspect_target": ctx.attr.asp_deps[0],
                "input_target": ctx.attr.srcs[0],
                "output_target": ctx.attr.srcs[1],
                "rule_ctx": ctx,
                "aspect_ctx": ctx.attr.asp_deps[0][aspect_ctx_provider].ctx,
                "aspect_ctx.rule": ctx.attr.asp_deps[0][aspect_ctx_provider].rule,
                "source_file": source_file,
                "generated_file": generated_file,
                "source_root": source_file.root,
                "generated_root": generated_file.root,
            }
            return CheckInfo(**prepare_params(objects))

        check = rule(
            implementation = _check_impl,
            attrs = {
                "deps": attr.label_list(),
                "asp_deps": attr.label_list(aspects = [my_aspect]),
                "srcs": attr.label_list(allow_files = True),
            },
        )
        """);

    scratch.file(
        "test/starlark/BUILD",
        """
        load(":rules.bzl", "check", "dep", "genfile")

        dep(name = "foo")

        dep(name = "bar")

        alias(
            name = "foobar",
            actual = ":foo",
        )

        genfile(name = "output")

        check(
            name = "check",
            srcs = [
                "input.txt",
                "output.txt",
            ],
            asp_deps = [":bar"],
            deps = [
                ":foo",
                ":foobar",
            ],
        )
        """);
  }

  @Test
  public void testStringRepresentations_strings() throws Exception {
    assertThat(starlarkLoadingEval("str('foo')")).isEqualTo("foo");
    assertThat(starlarkLoadingEval("'%s' % 'foo'")).isEqualTo("foo");
    assertThat(starlarkLoadingEval("'{}'.format('foo')")).isEqualTo("foo");
    assertThat(starlarkLoadingEval("repr('foo')")).isEqualTo("\"foo\"");
    assertThat(starlarkLoadingEval("'%r' % 'foo'")).isEqualTo("\"foo\"");
  }

  @Test
  public void testStringRepresentations_labels() throws Exception {
    assertThat(starlarkLoadingEval("str(Label('//foo:bar'))")).isEqualTo("@@//foo:bar");
    assertThat(starlarkLoadingEval("'%s' % Label('//foo:bar')")).isEqualTo("@@//foo:bar");
    assertThat(starlarkLoadingEval("'{}'.format(Label('//foo:bar'))")).isEqualTo("@@//foo:bar");
    assertThat(starlarkLoadingEval("repr(Label('//foo:bar'))")).isEqualTo("Label(\"//foo:bar\")");
    assertThat(starlarkLoadingEval("'%r' % Label('//foo:bar')")).isEqualTo("Label(\"//foo:bar\")");

    assertThat(starlarkLoadingEval("'{}'.format([Label('//foo:bar')])"))
        .isEqualTo("[Label(\"//foo:bar\")]");
  }

  @Test
  public void testStringRepresentations_primitives() throws Exception {
    // Strings are tested in a separate test case as they have different str and repr values.
    assertStringRepresentation("1543", "1543");
    assertStringRepresentation("True", "True");
    assertStringRepresentation("False", "False");
  }

  @Test
  public void testStringRepresentations_containers() throws Exception {
    assertStringRepresentation("['a', 'b']", "[\"a\", \"b\"]");
    assertStringRepresentation("('a', 'b')", "(\"a\", \"b\")");
    assertStringRepresentation("{'a': 'b', 'c': 'd'}", "{\"a\": \"b\", \"c\": \"d\"}");
    assertStringRepresentation("struct(d = 4, c = 3)", "struct(c = 3, d = 4)");
  }

  @Test
  public void testStringRepresentations_functions() throws Exception {
    assertStringRepresentation("all", "<built-in function all>");
    assertStringRepresentation("def f(): pass", "f", "<function f from //eval:eval.bzl>");
  }

  @Test
  public void testStringRepresentations_rules() throws Exception {
    setBuildLanguageOptions("--experimental_builtins_injection_override=+xcode_config");
    assertStringRepresentation("native.xcode_config", "<rule xcode_config>");
    assertStringRepresentation("def f(): pass", "rule(implementation=f)", "<rule>");
  }

  @Test
  public void testStringRepresentations_aspects() throws Exception {
    assertStringRepresentation("def f(): pass", "aspect(implementation=f)", "<aspect>");
  }

  @Test
  public void testStringRepresentations_providers() throws Exception {
    assertStringRepresentation("provider()", "<provider>");
    assertStringRepresentation(
        "p = provider()", "p(b = 'foo', a = 1)", "struct(a = 1, b = \"foo\")");
  }

  @Test
  public void testStringRepresentations_select() throws Exception {
    assertStringRepresentation(
        "select({'//foo': ['//bar']}) + select({'//foo2': ['//bar2']})",
        "select({\"//foo\": [\"//bar\"]}) + select({\"//foo2\": [\"//bar2\"]})");
  }

  @Test
  public void testStringRepresentations_ruleContext() throws Exception {
    generateFilesToTestStrings();
    ConfiguredTarget target = getConfiguredTarget("//test/starlark:check");
    StarlarkInfo checkInfo = getStarlarkProvider(target, "CheckInfo");

    for (String suffix : SUFFIXES) {
      assertThat(checkInfo.getValue("rule_ctx" + suffix))
          .isEqualTo("<rule context for //test/starlark:check>");
      assertThat(checkInfo.getValue("aspect_ctx" + suffix))
          .isEqualTo("<aspect context for //test/starlark:bar>");
      assertThat(checkInfo.getValue("aspect_ctx.rule" + suffix))
          .isEqualTo("<rule collection for //test/starlark:bar>");
    }
  }

  @Test
  public void testStringRepresentations_files() throws Exception {
    generateFilesToTestStrings();
    ConfiguredTarget target = getConfiguredTarget("//test/starlark:check");
    StarlarkInfo checkInfo = getStarlarkProvider(target, "CheckInfo");

    for (String suffix : SUFFIXES) {
      assertThat(checkInfo.getValue("source_file" + suffix))
          .isEqualTo("<source file test/starlark/input.txt>");
      assertThat(checkInfo.getValue("generated_file" + suffix))
          .isEqualTo("<generated file test/starlark/output.txt>");
    }
  }

  @Test
  public void testStringRepresentations_root() throws Exception {
    generateFilesToTestStrings();
    ConfiguredTarget target = getConfiguredTarget("//test/starlark:check");
    StarlarkInfo checkInfo = getStarlarkProvider(target, "CheckInfo");

    for (String suffix : SUFFIXES) {
      assertThat(checkInfo.getValue("source_root" + suffix)).isEqualTo("<source root>");
      assertThat(checkInfo.getValue("generated_root" + suffix)).isEqualTo("<derived root>");
    }
  }

  @Test
  public void testStringRepresentations_glob() throws Exception {
    scratch.file("eval/one.txt");
    scratch.file("eval/two.txt");
    scratch.file("eval/three.txt");

    assertStringRepresentationInBuildFile(
        "glob(['*.txt'])",
        "[\"one.txt\", \"three.txt\", \"two.txt\"]");
  }

  @Test
  public void testStringRepresentations_attr() throws Exception {
    assertStringRepresentation("attr", "<attr>");
    assertStringRepresentation("attr.int()", "<attr.int>");
    assertStringRepresentation("attr.string()", "<attr.string>");
    assertStringRepresentation("attr.label()", "<attr.label>");
    assertStringRepresentation("attr.string_list()", "<attr.string_list>");
    assertStringRepresentation("attr.int_list()", "<attr.int_list>");
    assertStringRepresentation("attr.label_list()", "<attr.label_list>");
    assertStringRepresentation("attr.label_keyed_string_dict()", "<attr.label_keyed_string_dict>");
    assertStringRepresentation("attr.bool()", "<attr.bool>");
    assertStringRepresentation("attr.output()", "<attr.output>");
    assertStringRepresentation("attr.output_list()", "<attr.output_list>");
    assertStringRepresentation("attr.string_dict()", "<attr.string_dict>");
    assertStringRepresentation("attr.string_list_dict()", "<attr.string_list_dict>");
  }

  @Test
  public void testStringRepresentations_targets() throws Exception {
    generateFilesToTestStrings();
    ConfiguredTarget target = getConfiguredTarget("//test/starlark:check");
    StarlarkInfo checkInfo = getStarlarkProvider(target, "CheckInfo");

    for (String suffix : SUFFIXES) {
      assertThat(checkInfo.getValue("target" + suffix)).isEqualTo("<target //test/starlark:foo>");
      assertThat(checkInfo.getValue("input_target" + suffix))
          .isEqualTo("<input file target //test/starlark:input.txt>");
      assertThat(checkInfo.getValue("output_target" + suffix))
          .isEqualTo("<output file target //test/starlark:output.txt>");
      assertThat(checkInfo.getValue("alias_target" + suffix))
          .isEqualTo("<alias target //test/starlark:foobar of //test/starlark:foo>");
      assertThat(checkInfo.getValue("aspect_target" + suffix))
          .isEqualTo("<merged target //test/starlark:bar>");
    }
  }
}
