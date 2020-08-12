// Copyright 2019 The Bazel Authors. All rights reserved.
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
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static com.google.devtools.build.lib.packages.BuildType.OUTPUT;

import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.MockRule;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.query2.PostAnalysisQueryEnvironment;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Setting;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.engine.QueryParser;
import com.google.devtools.build.lib.query2.query.aspectresolvers.AspectResolver.Mode;
import com.google.devtools.build.lib.util.FileTypeSet;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import org.junit.Before;
import org.junit.Test;

/** Tests cquery's BUILD output format. */
public class BuildOutputFormatterCallbackTest extends ConfiguredTargetQueryTest {

  private CqueryOptions options;
  private Reporter reporter;
  private final List<Event> events = new ArrayList<>();

  private static MockRule.State simpleRule() {
    return MockRule.define(
        "my_rule",
        (builder, env) ->
            builder
                .add(attr("deps", LABEL_LIST).allowedFileTypes(FileTypeSet.ANY_FILE))
                .add(attr("out", OUTPUT)));
  }

  @Before
  public final void setUpCqueryOptions() throws Exception {
    this.options = new CqueryOptions();
    // TODO(bazel-team): reduce the confusion about these two seemingly similar settings.
    // options.aspectDeps impacts how proto and similar output formatters output aspect results.
    // Setting.INCLUDE_ASPECTS impacts whether or not aspect dependencies are included when
    // following target deps. See CommonQueryOptions for further flag details.
    options.aspectDeps = Mode.OFF;
    helper.setQuerySettings(Setting.INCLUDE_ASPECTS);
    this.reporter = new Reporter(new EventBus(), events::add);
    helper.useRuleClassProvider(
        setRuleClassProviders(BuildOutputFormatterCallbackTest::simpleRule).build());
  }

  private List<String> getOutput(String queryExpression) throws Exception {
    QueryExpression expression = QueryParser.parse(queryExpression, getDefaultFunctions());
    Set<String> targetPatternSet = new LinkedHashSet<>();
    expression.collectTargetPatterns(targetPatternSet);
    helper.setQuerySettings(Setting.NO_IMPLICIT_DEPS);
    PostAnalysisQueryEnvironment<ConfiguredTarget> env =
        ((ConfiguredTargetQueryHelper) helper).getPostAnalysisQueryEnvironment(targetPatternSet);

    ByteArrayOutputStream output = new ByteArrayOutputStream();
    BuildOutputFormatterCallback callback =
        new BuildOutputFormatterCallback(
            reporter,
            options,
            new PrintStream(output),
            getHelper().getSkyframeExecutor(),
            env.getAccessor());
    env.evaluateQuery(expression, callback);
    return Arrays.asList(output.toString().split(System.lineSeparator()));
  }

  @Test
  public void selectInAttribute() throws Exception {
    writeFile(
        "test/BUILD",
        "my_rule(",
        "  name = 'my_rule',",
        "  deps = select({",
        "    ':garfield': ['lasagna.java', 'naps.java'],",
        "    '//conditions:default': ['mondays.java']",
        "  })",
        ")",
        "config_setting(",
        "  name = 'garfield',",
        "  values = {'test_arg': 'cat'}",
        ")");

    getHelper().useConfiguration("--test_arg=cat");
    assertThat(getOutput("//test:my_rule"))
        .containsExactly(
            "# /workspace/test/BUILD:1:8",
            "my_rule(",
            "  name = \"my_rule\",",
            "  deps = [\"//test:lasagna.java\", \"//test:naps.java\"],",
            ")")
        .inOrder();

    getHelper().useConfiguration("--test_arg=hound");
    assertThat(getOutput("//test:my_rule"))
        .containsExactly(
            "# /workspace/test/BUILD:1:8",
            "my_rule(",
            "  name = \"my_rule\",",
            "  deps = [\"//test:mondays.java\"],",
            ")")
        .inOrder();
  }

  @Test
  public void alias() throws Exception {
    writeFile(
        "test/BUILD",
        "my_rule(",
        "  name = 'my_rule',",
        "  deps = select({",
        "    ':garfield': ['lasagna.java', 'naps.java'],",
        "    '//conditions:default': ['mondays.java']",
        "  })",
        ")",
        "config_setting(",
        "  name = 'garfield',",
        "  values = {'test_arg': 'cat'}",
        ")",
        "alias(",
        "  name = 'my_alias',",
        "  actual = ':my_rule'",
        ")");

    assertThat(getOutput("//test:my_alias"))
        .containsExactly(
            "# /workspace/test/BUILD:12:6",
            "alias(",
            "  name = \"my_alias\",",
            "  actual = \"//test:my_rule\",",
            ")")
        .inOrder();
  }

  @Test
  public void aliasWithSelect() throws Exception {
    writeFile(
        "test/BUILD",
        "my_rule(",
        "  name = 'my_first_rule',",
        "  deps = ['penne.java'],",
        ")",
        "my_rule(",
        "  name = 'my_second_rule',",
        "  deps = ['linguini.java'],",
        ")",
        "config_setting(",
        "  name = 'garfield',",
        "  values = {'test_arg': 'cat'}",
        ")",
        "alias(",
        "  name = 'my_alias',",
        "  actual = select({",
        "    ':garfield': ':my_first_rule',",
        "    '//conditions:default': ':my_second_rule'",
        "  })",
        ")");

    getHelper().useConfiguration("--test_arg=cat");
    assertThat(getOutput("//test:my_alias"))
        .containsExactly(
            "# /workspace/test/BUILD:13:6",
            "alias(",
            "  name = \"my_alias\",",
            "  actual = \"//test:my_first_rule\",",
            ")")
        .inOrder();

    getHelper().useConfiguration("--test_arg=hound");
    assertThat(getOutput("//test:my_alias"))
        .containsExactly(
            "# /workspace/test/BUILD:13:6",
            "alias(",
            "  name = \"my_alias\",",
            "  actual = \"//test:my_second_rule\",",
            ")")
        .inOrder();
  }

  @Test
  public void sourceFile() throws Exception {
    writeFile(
        "test/BUILD",
        "my_rule(",
        "  name = 'my_rule',",
        "  deps = select({",
        "    ':garfield': ['lasagna.java', 'naps.java'],",
        "    '//conditions:default': ['mondays.java']",
        "  })",
        ")",
        "config_setting(",
        "  name = 'garfield',",
        "  values = {'test_arg': 'cat'}",
        ")");

    assertThat(getOutput("//test:lasagna.java")).containsExactly("");
  }

  @Test
  public void outputFile() throws Exception {
    writeFile(
        "test/BUILD",
        "my_rule(",
        "  name = 'my_rule',",
        "  deps = select({",
        "    ':garfield': ['lasagna.java', 'naps.java'],",
        "    '//conditions:default': ['mondays.java']",
        "  }),",
        "  out = 'output.txt'",
        ")",
        "config_setting(",
        "  name = 'garfield',",
        "  values = {'test_arg': 'cat'}",
        ")");

    getHelper().useConfiguration("--test_arg=cat");
    assertThat(getOutput("//test:output.txt"))
        .containsExactly(
            "# /workspace/test/BUILD:1:8",
            "my_rule(",
            "  name = \"my_rule\",",
            "  deps = [\"//test:lasagna.java\", \"//test:naps.java\"],",
            "  out = \"//test:output.txt\",",
            ")")
        .inOrder();

    getHelper().useConfiguration("--test_arg=hound");
    assertThat(getOutput("//test:output.txt"))
        .containsExactly(
            "# /workspace/test/BUILD:1:8",
            "my_rule(",
            "  name = \"my_rule\",",
            "  deps = [\"//test:mondays.java\"],",
            "  out = \"//test:output.txt\",",
            ")")
        .inOrder();
  }
}
