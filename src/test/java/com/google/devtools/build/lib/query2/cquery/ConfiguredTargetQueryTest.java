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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.transitions.SplitTransition;
import com.google.devtools.build.lib.analysis.test.TestConfiguration.TestOptions;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryFunction;
import com.google.devtools.build.lib.query2.testutil.PostAnalysisQueryTest;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ConfiguredTargetQueryEnvironment}. */
@RunWith(JUnit4.class)
public abstract class ConfiguredTargetQueryTest extends PostAnalysisQueryTest<ConfiguredTarget> {

  @Override
  protected QueryHelper<ConfiguredTarget> createQueryHelper() {
    if (helper != null) {
      getHelper().cleanUp();
    }
    helper = new ConfiguredTargetQueryHelper();
    return helper;
  }

  @Override
  public HashMap<String, QueryFunction> getDefaultFunctions() {
    ImmutableList<QueryFunction> defaultFunctions =
        ImmutableList.copyOf(ConfiguredTargetQueryEnvironment.FUNCTIONS);
    HashMap<String, QueryFunction> functions = new HashMap<>();
    for (QueryFunction queryFunction : defaultFunctions) {
      functions.put(queryFunction.getName(), queryFunction);
    }
    return functions;
  }

  @Override
  protected final BuildConfiguration getConfiguration(ConfiguredTarget ct) {
    return getHelper()
        .getSkyframeExecutor()
        .getConfiguration(getHelper().getReporter(), ct.getConfigurationKey());
  }

  /** SplitTransition on --test_arg */
  protected static class TestArgSplitTransition implements SplitTransition {
    String toOption1;
    String toOption2;

    public TestArgSplitTransition(String toOption1, String toOptions2) {
      this.toOption1 = toOption1;
      this.toOption2 = toOptions2;
    }

    @Override
    public Map<String, BuildOptions> split(BuildOptions options, EventHandler eventHandler) {
      BuildOptions result1 = options.clone();
      BuildOptions result2 = options.clone();
      result1.get(TestOptions.class).testArguments = Collections.singletonList(toOption1);
      result2.get(TestOptions.class).testArguments = Collections.singletonList(toOption2);
      return ImmutableMap.of("result1", result1, "result2", result2);
    }
  }

  @Override
  @Test
  public void testMultipleTopLevelConfigurations_nullConfigs() throws Exception {
    writeFile("test/BUILD", "java_library(name='my_java',", "  srcs = ['foo.java'],", ")");

    Set<ConfiguredTarget> result = eval("//test:my_java+//test:foo.java");

    assertThat(result).hasSize(2);

    Iterator<ConfiguredTarget> resultIterator = result.iterator();
    ConfiguredTarget first = resultIterator.next();
    if (first.getLabel().toString().equals("//test:foo.java")) {
      assertThat(getConfiguration(first)).isNull();
      assertThat(getConfiguration(resultIterator.next())).isNotNull();
    } else {
      assertThat(getConfiguration(first)).isNotNull();
      assertThat(getConfiguration(resultIterator.next())).isNull();
    }
  }

  @Override
  public void testMultipleTopLevelConfigurations_multipleConfigsPrefersTopLevel() {
    // When the same target exists in multiple configurations, cquery doesn't guarantee which
    // instance is evaluated first. So disable this test.
  }
}
