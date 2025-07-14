// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.query2.cquery;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.query2.PostAnalysisQueryEnvironment;
import com.google.devtools.build.lib.query2.common.CqueryNode;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.engine.QueryParser;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.Set;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.StarlarkSemantics;
import org.junit.Before;
import org.junit.Test;

/** Tests cquery's {@link --output=starlark} format. */
public final class StarlarkOutputFormatterCallbackTest extends ConfiguredTargetQueryTest {

  private final CqueryOptions options = new CqueryOptions();
  private final Reporter reporter = new Reporter(new EventBus());

  @Before
  public void defineSimpleRule() throws Exception {
    writeFile(
        "defs/rules.bzl",
        """
        RuleInfo = provider()
        FirstAspectInfo = provider()
        SecondAspectInfo = provider()
        def _r_impl(ctx):
            return [
                DefaultInfo(
                    files = depset([ctx.file.file]),
                ),
                OutputGroupInfo(
                    shared = depset([ctx.file.file]),
                ),
                RuleInfo(),
            ]
        r = rule(
            implementation = _r_impl,
            attrs = {
                'file': attr.label(allow_single_file = True),
            },
        )
        def _a_impl(target, ctx):
            custom_output_group = ctx.actions.declare_file(target.label.name + '_custom_a_file')
            shared_output_group = ctx.actions.declare_file(target.label.name + '_shared_a_file')
            ctx.actions.run_shell(
                outputs = [custom_output_group, shared_output_group],
                command = "touch %s && touch %s" % (custom_output_group.path, shared_output_group.path),
            )
            return [
                OutputGroupInfo(
                    custom = depset([custom_output_group]),
                    shared = depset([shared_output_group]),
                ),
                FirstAspectInfo(),
            ] + (
                [SecondAspectInfo()] if ctx.attr.collide else []
            )
        a = aspect(
            implementation = _a_impl,
            attrs = {
                'collide': attr.string(),
            },
        )
        def _b_impl(target, ctx):
            custom_output_group = ctx.actions.declare_file(target.label.name + '_custom_b_file')
            shared_output_group = ctx.actions.declare_file(target.label.name + '_shared_b_file')
            ctx.actions.run_shell(
                outputs = [custom_output_group, shared_output_group],
                command = "touch %s && touch %s" % (custom_output_group.path, shared_output_group.path),
            )
            return [
                OutputGroupInfo(
                    custom = depset([custom_output_group]),
                    shared = depset([shared_output_group]),
                ),
                SecondAspectInfo(),
            ]
        b = aspect(implementation = _b_impl)
        """);
    writeFile("defs/BUILD", "exports_files(['rules.bzl'])");
    writeFile(
        "pkg/BUILD",
        """
        load("//defs:rules.bzl", "r")

        r(
            name = "main",
            file = "BUILD",
        )
        """);
  }

  @Test
  public void basicQuery() throws Exception {
    assertThat(getOutput("[f.path for f in providers(target)['DefaultInfo'].files.to_list()]"))
        .isEqualTo(strList("pkg/BUILD"));
    assertThat(
            getOutput(
                "[f.basename for f in providers(target)['OutputGroupInfo'].shared.to_list()]"))
        .isEqualTo(strList("BUILD"));
    assertThat(getOutput("sorted(providers(target).keys())"))
        .isEqualTo(
            strList(
                "//defs:rules.bzl%RuleInfo",
                "DefaultInfo", "FileProvider", "FilesToRunProvider", "OutputGroupInfo"));
  }

  @Test
  public void basicQuery_withSingleAspect() throws Exception {
    assertThat(
            getOutput(
                "[f.path for f in providers(target)['DefaultInfo'].files.to_list()]",
                "//defs:rules.bzl%a"))
        .isEqualTo(strList("pkg/BUILD"));
    assertThat(
            getOutput(
                "[f.basename for f in providers(target)['OutputGroupInfo'].custom.to_list()]",
                "//defs:rules.bzl%a"))
        .isEqualTo(strList("main_custom_a_file"));
    assertThat(
            getOutput(
                "[f.basename for f in providers(target)['OutputGroupInfo'].shared.to_list()]",
                "//defs:rules.bzl%a"))
        .isEqualTo(strList("BUILD", "main_shared_a_file"));
    assertThat(getOutput("sorted(providers(target).keys())", "//defs:rules.bzl%a"))
        .isEqualTo(
            strList(
                "//defs:rules.bzl%FirstAspectInfo",
                "//defs:rules.bzl%RuleInfo",
                "DefaultInfo",
                "FileProvider",
                "FilesToRunProvider",
                "OutputGroupInfo"));
  }

  @Test
  public void basicQuery_withTwoAspects() throws Exception {
    assertThat(
            getOutput(
                "[f.path for f in providers(target)['DefaultInfo'].files.to_list()]",
                "//defs:rules.bzl%a",
                "//defs:rules.bzl%b"))
        .isEqualTo(strList("pkg/BUILD"));
    assertThat(
            getOutput(
                "[f.basename for f in providers(target)['OutputGroupInfo'].custom.to_list()]",
                "//defs:rules.bzl%a",
                "//defs:rules.bzl%b"))
        .isEqualTo(strList("main_custom_a_file", "main_custom_b_file"));
    assertThat(
            getOutput(
                "[f.basename for f in providers(target)['OutputGroupInfo'].shared.to_list()]",
                "//defs:rules.bzl%a",
                "//defs:rules.bzl%b"))
        .isEqualTo(strList("BUILD", "main_shared_a_file", "main_shared_b_file"));
    assertThat(
            getOutput(
                "sorted(providers(target).keys())", "//defs:rules.bzl%a", "//defs:rules.bzl%b"))
        .isEqualTo(
            strList(
                "//defs:rules.bzl%FirstAspectInfo",
                "//defs:rules.bzl%RuleInfo",
                "//defs:rules.bzl%SecondAspectInfo",
                "DefaultInfo",
                "FileProvider",
                "FilesToRunProvider",
                "OutputGroupInfo"));
  }

  private String getOutput(String starlarkExpr, String... aspects) throws Exception {
    QueryExpression expression = QueryParser.parse("//pkg:main", getDefaultFunctions());
    Set<String> targetPatternSet = new LinkedHashSet<>();
    expression.collectTargetPatterns(targetPatternSet);
    options.file = "";
    options.expr = starlarkExpr;
    PostAnalysisQueryEnvironment<CqueryNode> env =
        ((ConfiguredTargetQueryHelper) helper)
            .getPostAnalysisQueryEnvironment(targetPatternSet, Arrays.asList(aspects));

    ByteArrayOutputStream output = new ByteArrayOutputStream();
    var callback =
        new StarlarkOutputFormatterCallback(
            reporter,
            options,
            new PrintStream(output),
            getHelper().getSkyframeExecutor(),
            env.getAccessor(),
            StarlarkSemantics.DEFAULT);
    env.evaluateQuery(expression, callback);
    return output.toString().trim();
  }

  private static String strList(String... elements) {
    return Starlark.str(StarlarkList.immutableOf(elements), StarlarkSemantics.DEFAULT);
  }
}
