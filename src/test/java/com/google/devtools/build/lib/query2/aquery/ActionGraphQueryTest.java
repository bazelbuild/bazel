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
package com.google.devtools.build.lib.query2.aquery;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryFunction;
import com.google.devtools.build.lib.query2.testutil.PostAnalysisQueryTest;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Set;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ActionGraphQueryEnvironment}. */
@RunWith(JUnit4.class)
public class ActionGraphQueryTest extends PostAnalysisQueryTest<KeyedConfiguredTargetValue> {
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
  protected BuildConfigurationValue getConfiguration(
      KeyedConfiguredTargetValue configuredTargetValue) {
    return getHelper()
        .getSkyframeExecutor()
        .getConfiguration(
            getHelper().getReporter(),
            configuredTargetValue.getConfiguredTarget().getConfigurationKey());
  }

  @Override
  protected QueryHelper<KeyedConfiguredTargetValue> createQueryHelper() {
    return new ActionGraphQueryHelper();
  }

  @Override
  @Test
  public void testMultipleTopLevelConfigurations_nullConfigs() throws Exception {
    writeFile("test/BUILD", "java_library(name='my_java',", "  srcs = ['foo.java'],", ")");

    Set<KeyedConfiguredTargetValue> result = eval("//test:my_java+//test:foo.java");

    assertThat(result).hasSize(2);

    Iterator<KeyedConfiguredTargetValue> resultIterator = result.iterator();
    KeyedConfiguredTargetValue first = resultIterator.next();
    if (first.getConfiguredTarget().getLabel().toString().equals("//test:foo.java")) {
      assertThat(getConfiguration(first)).isNull();
      assertThat(getConfiguration(resultIterator.next())).isNotNull();
    } else {
      assertThat(getConfiguration(first)).isNotNull();
      assertThat(getConfiguration(resultIterator.next())).isNull();
    }
  }

  // Regression test for b/235526333.
  @Test
  public void testImplicitToolchainBinding_containsToolchainTarget() throws Exception {
    writeFile(
        "q/BUILD",
        "load(':q.bzl', 'r', 'tc')",
        "genrule(",
        "    name = 'gr',",
        "    srcs = [],",
        "    outs = ['gro'],",
        "    cmd = 'echo GRO > $@',",
        ")",
        "tc(",
        "    name = 'tc',",
        "    dep = ':gr',",
        ")",
        "toolchain_type(name = 'type')",
        "toolchain(",
        "    name = 'tc.toolchain',",
        "    toolchain = ':tc',",
        "    toolchain_type = ':type',",
        ")",
        "r(name = 'r')");
    writeFile(
        "q/q.bzl",
        "def _r_impl(ctx):",
        "    gro = ctx.toolchains['//q:type'].gro",
        "    o = ctx.actions.declare_file(ctx.label.name + '.output')",
        "    ctx.actions.run_shell(",
        "        inputs = depset([gro]),",
        "        outputs = [o],",
        "        command = 'cp ' + gro.path + ' ' + o.path,",
        "    )",
        "    return DefaultInfo(files = depset([o]))",
        "def _tc_impl(ctx):",
        "    gro = ctx.files.dep[0]",
        "    return [platform_common.ToolchainInfo(gro = gro)]",
        "tc = rule(",
        "    implementation = _tc_impl,",
        "    attrs = {'dep': attr.label()},",
        ")",
        "r = rule(",
        "    implementation = _r_impl,",
        "    toolchains = ['//q:type'],",
        ")");
    appendToWorkspace("register_toolchains('//q:tc.toolchain')");

    Set<KeyedConfiguredTargetValue> result = eval("deps('//q:r')");

    assertDoesNotContainEvent("Targets were missing from graph");
    assertThat(
            result.stream()
                .map(x -> x.getConfiguredTarget().getOriginalLabel().getCanonicalForm())
                .collect(toImmutableList()))
        .contains("//q:tc");
  }
}
