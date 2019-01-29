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
package com.google.devtools.build.lib.query2.engine;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.query2.ActionGraphQueryEnvironment;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryFunction;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetValue;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Set;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ActionGraphQueryEnvironment}. */
@RunWith(JUnit4.class)
public class ActionGraphQueryTest extends PostAnalysisQueryTest<ConfiguredTargetValue> {
  @Override
  protected HashMap<String, QueryFunction> getDefaultFunctions() {
    ImmutableList<QueryFunction> defaultFunctions =
        new ImmutableList.Builder<QueryFunction>()
            .addAll(ActionGraphQueryEnvironment.FUNCTIONS)
            .addAll(ActionGraphQueryEnvironment.AQUERY_FUNCTIONS)
            .build();
    HashMap<String, QueryFunction> functions = new HashMap<>();
    for (QueryFunction queryFunction : defaultFunctions) {
      functions.put(queryFunction.getName(), queryFunction);
    }
    return functions;
  }

  @Override
  protected BuildConfiguration getConfiguration(ConfiguredTargetValue configuredTargetValue) {
    return getHelper()
        .getSkyframeExecutor()
        .getConfiguration(
            getHelper().reporter,
            configuredTargetValue.getConfiguredTarget().getConfigurationKey());
  }

  @Override
  protected QueryHelper<ConfiguredTargetValue> createQueryHelper() {
    if (helper != null) {
      getHelper().cleanUp();
    }
    helper = new ActionGraphQueryHelper();
    return helper;
  }

  @Override
  @Test
  public void testMultipleTopLevelConfigurations_nullConfigs() throws Exception {
    writeFile("test/BUILD", "java_library(name='my_java',", "  srcs = ['foo.java'],", ")");

    Set<ConfiguredTargetValue> result = eval("//test:my_java+//test:foo.java");

    assertThat(result).hasSize(2);

    Iterator<ConfiguredTargetValue> resultIterator = result.iterator();
    ConfiguredTargetValue first = resultIterator.next();
    if (first.getConfiguredTarget().getLabel().toString().equals("//test:foo.java")) {
      assertThat(getConfiguration(first)).isNull();
      assertThat(getConfiguration(resultIterator.next())).isNotNull();
    } else {
      assertThat(getConfiguration(first)).isNotNull();
      assertThat(getConfiguration(resultIterator.next())).isNull();
    }
  }
}
