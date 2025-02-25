// Copyright 2020 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.packages.LabelPrinter;
import com.google.devtools.build.lib.query2.PostAnalysisQueryEnvironment;
import com.google.devtools.build.lib.query2.common.CqueryNode;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Setting;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.engine.QueryParser;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import org.junit.Before;
import org.junit.Test;

/** Tests cquery's {@link --output=graph} format. */
public class GraphOutputFormatterCallbackTest extends ConfiguredTargetQueryTest {

  private CqueryOptions options;
  private Reporter reporter;
  private final List<Event> events = new ArrayList<>();

  @Before
  public final void defineSimpleRule() throws Exception {
    writeFile(
        "defs/defs.bzl",
        """
        def _impl(ctx):
            pass

        simple_rule = rule(
            implementation = _impl,
            attrs = {
                "deps": attr.label_list(allow_files = True),
                "tool_deps": attr.label_list(cfg = "exec"),
            },
        )
        """);
    writeFile("defs/BUILD");
  }

  @Before
  public final void setUpCqueryOptions() {
    this.options = new CqueryOptions();
    options.graphNodeStringLimit = 512;
    this.reporter = new Reporter(new EventBus(), events::add);
  }

  private ImmutableList<String> getOutput(String queryExpression) throws Exception {
    QueryExpression expression = QueryParser.parse(queryExpression, getDefaultFunctions());
    Set<String> targetPatternSet = new LinkedHashSet<>();
    expression.collectTargetPatterns(targetPatternSet);
    helper.setQuerySettings(Setting.NO_IMPLICIT_DEPS);
    PostAnalysisQueryEnvironment<CqueryNode> env =
        ((ConfiguredTargetQueryHelper) helper).getPostAnalysisQueryEnvironment(targetPatternSet);

    ByteArrayOutputStream output = new ByteArrayOutputStream();
    GraphOutputFormatterCallback callback =
        new GraphOutputFormatterCallback(
            reporter,
            options,
            new PrintStream(output),
            getHelper().getSkyframeExecutor(),
            env.getAccessor(),
            ct -> env.getFwdDeps(ImmutableList.of(ct)),
            LabelPrinter.legacy());
    env.evaluateQuery(expression, callback);
    return ImmutableList.copyOf(output.toString().split("\n"));
  }

  /** Convenience method for easily injecting a config hash into an expected output sequence. */
  private static List<String> withConfigHash(String configHash, String... pattern) {
    return Arrays.stream(pattern)
        .map(entry -> entry.replace("%s", configHash))
        .collect(Collectors.toList());
  }

  @Test
  public void basicGraph() throws Exception {
    writeFile(
        "test/BUILD",
        """
        load("//defs:defs.bzl", "simple_rule")

        simple_rule(
            name = "a",
            deps = [
                ":b",
                ":c",
            ],
        )

        simple_rule(
            name = "b",
            deps = [":d"],
        )

        simple_rule(name = "c")

        simple_rule(name = "d")
        """);
    List<String> output = getOutput("deps(//test:a)");
    String firstNode = output.get(2);
    String configHash = firstNode.substring(firstNode.indexOf("(") + 1, firstNode.length() - 2);
    assertThat(getOutput("deps(//test:a)"))
        .isEqualTo(
            withConfigHash(
                configHash,
                "digraph mygraph {",
                "  node [shape=box];",
                "  \"//test:a (%s)\"",
                "  \"//test:a (%s)\" -> \"//test:b (%s)\"",
                "  \"//test:a (%s)\" -> \"//test:c (%s)\"",
                "  \"//test:c (%s)\"",
                "  \"//test:b (%s)\"",
                "  \"//test:b (%s)\" -> \"//test:d (%s)\"",
                "  \"//test:d (%s)\"",
                "}"));
  }

  @Test
  public void factorEquivalentNodes() throws Exception {
    options.graphFactored = true;
    writeFile(
        "test/BUILD",
        """
        load("//defs:defs.bzl", "simple_rule")

        simple_rule(
            name = "a",
            deps = [
                ":b",
                ":c",
            ],
        )

        simple_rule(
            name = "b",
            deps = [":d"],
        )

        simple_rule(
            name = "c",
            deps = [":d"],
        )

        simple_rule(name = "d")
        """);
    List<String> output = getOutput("deps(//test:a)");
    String firstNode = output.get(2);
    String configHash = firstNode.substring(firstNode.indexOf("(") + 1, firstNode.length() - 2);
    assertThat(getOutput("deps(//test:a)"))
        .isEqualTo(
            withConfigHash(
                configHash,
                "digraph mygraph {",
                "  node [shape=box];",
                "  \"//test:a (%s)\"",
                "  \"//test:a (%s)\" -> \"//test:b (%s)\\n//test:c (%s)\"",
                "  \"//test:b (%s)\\n//test:c (%s)\"",
                "  \"//test:b (%s)\\n//test:c (%s)\" -> \"//test:d (%s)\"",
                "  \"//test:d (%s)\"",
                "}"));
  }

  @Test
  public void nullAndToolDeps() throws Exception {
    writeFile(
        "test/BUILD",
        """
        load("//defs:defs.bzl", "simple_rule")

        simple_rule(
            name = "a",
            tool_deps = [":tool_dep"],
            deps = [
                ":b",
                ":file.src",
            ],
        )

        simple_rule(name = "b")

        simple_rule(name = "tool_dep")
        """);
    writeFile("test/file.src");
    ImmutableList<String> output = getOutput("deps(//test:a)" + getDependencyCorrection());
    String firstNode = output.get(2);
    String configHash = firstNode.substring(firstNode.indexOf("(") + 1, firstNode.length() - 2);
    String toolNode = output.get(6);
    String execConfigHash = toolNode.substring(toolNode.indexOf("(") + 1, toolNode.length() - 2);
    assertThat(output)
        .isEqualTo(
            withConfigHash(
                configHash,
                "digraph mygraph {",
                "  node [shape=box];",
                "  \"//test:a (%s)\"",
                "  \"//test:a (%s)\" -> \"//test:b (%s)\"",
                "  \"//test:a (%s)\" -> \"//test:file.src (null)\"",
                "  \"//test:a (%s)\" -> \"//test:tool_dep (" + execConfigHash + ")\"",
                "  \"//test:tool_dep (" + execConfigHash + ")\"",
                "  \"//test:file.src (null)\"",
                "  \"//test:b (%s)\"",
                "}"));
  }

  @Test
  public void selectsResolvedAndRemoved() throws Exception {
    writeFile(
        "test/BUILD",
        """
        load("//defs:defs.bzl", "simple_rule")

        config_setting(
            name = "use_a",
            define_values = {"a": "1"},
        )

        simple_rule(
            name = "a",
            deps = select({
                ":use_a": [":dep_with_a"],
                "//conditions:default": [":default_dep"],
            }),
        )

        simple_rule(name = "dep_with_a")

        simple_rule(name = "default_dep")
        """);
    getHelper().useConfiguration("--define", "a=1");
    List<String> output = getOutput("deps(//test:a)");
    String firstNode = output.get(2);
    String configHash = firstNode.substring(firstNode.indexOf("(") + 1, firstNode.length() - 2);
    assertThat(getOutput("deps(//test:a)"))
        .isEqualTo(
            withConfigHash(
                configHash,
                "digraph mygraph {",
                "  node [shape=box];",
                "  \"//test:a (%s)\"",
                "  \"//test:a (%s)\" -> \"//test:dep_with_a (%s)\"",
                "  \"//test:a (%s)\" -> \"//test:use_a (%s)\"",
                "  \"//test:use_a (%s)\"",
                "  \"//test:dep_with_a (%s)\"",
                "}"));
  }
}
