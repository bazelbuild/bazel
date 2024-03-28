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
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptionsView;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.transitions.SplitTransition;
import com.google.devtools.build.lib.analysis.util.DummyTestFragment.DummyTestOptions;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.query2.common.CqueryNode;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryFunction;
import com.google.devtools.build.lib.query2.testutil.PostAnalysisQueryTest;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ConfiguredTargetQueryEnvironment}. */
@RunWith(JUnit4.class)
public abstract class ConfiguredTargetQueryTest extends PostAnalysisQueryTest<CqueryNode> {

  @Override
  protected QueryHelper<CqueryNode> createQueryHelper() {
    return new ConfiguredTargetQueryHelper();
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
  protected final BuildConfigurationValue getConfiguration(CqueryNode kct) {
    return getHelper()
        .getSkyframeExecutor()
        .getConfiguration(getHelper().getReporter(), kct.getConfigurationKey());
  }

  /** SplitTransition on --foo */
  protected static class FooSplitTransition implements SplitTransition {
    String toOption1;
    String toOption2;

    public FooSplitTransition(String toOption1, String toOptions2) {
      this.toOption1 = toOption1;
      this.toOption2 = toOptions2;
    }

    @Override
    public ImmutableSet<Class<? extends FragmentOptions>> requiresOptionFragments() {
      return ImmutableSet.of(DummyTestOptions.class);
    }

    @Override
    public Map<String, BuildOptions> split(BuildOptionsView options, EventHandler eventHandler) {
      BuildOptionsView result1 = options.clone();
      BuildOptionsView result2 = options.clone();
      result1.get(DummyTestOptions.class).foo = toOption1;
      result2.get(DummyTestOptions.class).foo = toOption2;
      return ImmutableMap.of("result1", result1.underlying(), "result2", result2.underlying());
    }
  }

  @Override
  @Test
  public void testMultipleTopLevelConfigurations_nullConfigs() throws Exception {
    writeFile("test/BUILD", "java_library(name='my_java',", "  srcs = ['foo.java'],", ")");

    Set<CqueryNode> result = eval("//test:my_java+//test:foo.java");

    assertThat(result).hasSize(2);

    Iterator<CqueryNode> resultIterator = result.iterator();
    CqueryNode first = resultIterator.next();
    if (first.getLabel().toString().equals("//test:foo.java")) {
      assertThat(getConfiguration(first)).isNull();
      assertThat(getConfiguration(resultIterator.next())).isNotNull();
    } else {
      assertThat(getConfiguration(first)).isNotNull();
      assertThat(getConfiguration(resultIterator.next())).isNull();
    }
  }
}
