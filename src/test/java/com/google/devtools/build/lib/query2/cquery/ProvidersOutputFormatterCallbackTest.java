// Copyright 2018 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.query2.PostAnalysisQueryEnvironment;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.engine.QueryParser;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import org.junit.Before;
import org.junit.Test;

public class ProvidersOutputFormatterCallbackTest extends ConfiguredTargetQueryTest {

  private CqueryOptions options;
  private Reporter reporter;
  private final List<Event> events = new ArrayList<>();

  @Before
  public final void setUpCqueryOptions() {
    this.options = new CqueryOptions();
    options.experimentalProvidersOutput = true;
    this.reporter = new Reporter(new EventBus(), events::add);
  }

  @Test
  public void testProvidersOutput() throws Exception {
    writeFile(
        "BUILD",
        "load(':my_rule.bzl', 'my_rule')",
        "my_rule(name = 'foo')",
        "my_rule(name = 'bar')");

     writeFile("my_rule.bzl",
         "MyInfo = provider()",
         "def _my_rule_impl(ctx):",
         "  return [",
         "    DefaultInfo(),",
         "    MyInfo(),",
         "    OutputGroupInfo(foo_group = depset(), bar_group = depset()),",
         "  ]",
         "my_rule = rule(implementation = _my_rule_impl, attrs = {})");

    List<String> result = getOutput("//:foo");
    assertThat(result.get(0)).isEqualTo("//:foo [MyInfo, OutputGroupInfo[bar_group, foo_group]]");

    List<String> multipleResults = getOutput("//:all");
    assertThat(
        multipleResults.get(0)).isEqualTo(
            "//:bar [MyInfo, OutputGroupInfo[bar_group, foo_group]]");
    assertThat(
        multipleResults.get(1)).isEqualTo(
            "//:foo [MyInfo, OutputGroupInfo[bar_group, foo_group]]");
  }

  private List<String> getOutput(String queryExpression)
      throws Exception {
    QueryExpression expression = QueryParser.parse(queryExpression, getDefaultFunctions());
    Set<String> targetPatternSet = new LinkedHashSet<>();
    expression.collectTargetPatterns(targetPatternSet);
    PostAnalysisQueryEnvironment<ConfiguredTarget> env =
        ((ConfiguredTargetQueryHelper) helper).getPostAnalysisQueryEnvironment(targetPatternSet);
    ProvidersOutputFormatterCallback callback =
        new ProvidersOutputFormatterCallback(
            reporter,
            options,
            /*out=*/ null,
            getHelper().getSkyframeExecutor(),
            env.getAccessor());
    env.evaluateQuery(env.transformParsedQuery(QueryParser.parse(queryExpression, env)), callback);
    return callback.getResult();
  }
}
