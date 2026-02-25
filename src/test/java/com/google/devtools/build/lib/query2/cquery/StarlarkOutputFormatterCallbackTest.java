// Copyright 2026 The Bazel Authors. All rights reserved.
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

import static org.junit.Assert.assertThrows;

import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.query2.PostAnalysisQueryEnvironment;
import com.google.devtools.build.lib.query2.common.CqueryNode;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.engine.QueryParser;
import com.google.devtools.build.lib.query2.engine.QueryUtil;
import com.google.devtools.build.lib.query2.engine.QueryUtil.AggregateAllOutputFormatterCallback;
import com.google.devtools.build.lib.server.FailureDetails.ConfigurableQuery.Code;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.util.LinkedHashSet;
import java.util.Set;
import net.starlark.java.eval.StarlarkSemantics;
import org.junit.Test;

/** Tests cquery's Starlark output formatter. */
public final class StarlarkOutputFormatterCallbackTest extends ConfiguredTargetQueryTest {

  @Test
  public void evalErrorFailsQuery() throws Exception {
    writeFile(
        "test/BUILD",
        """
        filegroup(
            name = "foo",
            srcs = [],
        )
        """);

    CqueryOptions options = new CqueryOptions();
    options.file = "";
    options.expr = "build_options()";

    QueryExpression expression = QueryParser.parse("//test:foo", getDefaultFunctions());
    Set<String> targetPatternSet = new LinkedHashSet<>();
    expression.collectTargetPatterns(targetPatternSet);
    PostAnalysisQueryEnvironment<CqueryNode> env =
        ((ConfiguredTargetQueryHelper) helper).getPostAnalysisQueryEnvironment(targetPatternSet);

    StarlarkOutputFormatterCallback callback =
        new StarlarkOutputFormatterCallback(
            new Reporter(new EventBus()),
            options,
            new PrintStream(new ByteArrayOutputStream()),
            getHelper().getSkyframeExecutor(),
            env.getAccessor(),
            StarlarkSemantics.DEFAULT);

    AggregateAllOutputFormatterCallback<CqueryNode, Set<CqueryNode>> aggregateResultsCallback =
        QueryUtil.newOrderedAggregateAllOutputFormatterCallback(env);
    env.evaluateQuery(expression, aggregateResultsCallback);

    QueryException e =
        assertThrows(
            QueryException.class, () -> callback.process(aggregateResultsCallback.getResult()));
    assertConfigurableQueryCode(e.getFailureDetail(), Code.STARLARK_EVAL_ERROR);
  }
}
