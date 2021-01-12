// Copyright 2020 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.lib.dynamic;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.common.options.Converters;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link com.google.devtools.build.lib.dynamic.DynamicExecutionModule}. */
@RunWith(JUnit4.class)
public class DynamicExecutionModuleTest {
  private DynamicExecutionModule module;
  private DynamicExecutionOptions options;

  @Before
  public void setUp() {
    module = new DynamicExecutionModule();
    options = new DynamicExecutionOptions();
    options.dynamicWorkerStrategy = ""; // default
    options.dynamicLocalStrategy = Collections.emptyList(); // default
    options.dynamicRemoteStrategy = Collections.emptyList(); // default
  }

  @Test
  public void testGetLocalStrategies_getsDefaultWithNoOptions()
      throws AbruptExitException, OptionsParsingException {
    assertThat(module.getLocalStrategies(options)).isEqualTo(parseStrategies("worker,sandboxed"));
  }

  @Test
  public void testGetLocalStrategies_dynamicWorkerStrategyTakesSingleValue()
      throws AbruptExitException, OptionsParsingException {
    options.dynamicWorkerStrategy = "local,worker";
    // This looks weird, but it's expected behaviour that dynamic_worker_strategy
    // doesn't get parsed.
    Map<String, List<String>> expected = parseStrategies("sandboxed");
    expected.get("").add(0, "local,worker");
    assertThat(module.getLocalStrategies(options))
        .isEqualTo(ImmutableMap.copyOf(expected.entrySet()));
  }

  @Test
  public void testGetLocalStrategies_genericOptionOverridesFallbacks()
      throws AbruptExitException, OptionsParsingException {
    options.dynamicLocalStrategy = parseStrategiesToOptions("local,worker");
    assertThat(module.getLocalStrategies(options)).isEqualTo(parseStrategies("local,worker"));
  }

  @Test
  public void testGetLocalStrategies_specificOptionKeepsFallbacks()
      throws AbruptExitException, OptionsParsingException {
    options.dynamicLocalStrategy = parseStrategiesToOptions("Foo=local,worker");
    assertThat(module.getLocalStrategies(options))
        .isEqualTo(parseStrategies("Foo=local,worker", "worker,sandboxed"));
  }

  @Test
  public void testGetLocalStrategies_canMixSpecificsAndGenericOptions()
      throws AbruptExitException, OptionsParsingException {
    options.dynamicLocalStrategy = parseStrategiesToOptions("Foo=local,worker", "worker");
    assertThat(module.getLocalStrategies(options))
        .isEqualTo(parseStrategies("Foo=local,worker", "worker"));
  }

  private static List<Map.Entry<String, List<String>>> parseStrategiesToOptions(
      String... strategies) throws OptionsParsingException {
    Map<String, List<String>> result = parseStrategies(strategies);
    return Lists.newArrayList(result.entrySet());
  }

  private static Map<String, List<String>> parseStrategies(String... strategies)
      throws OptionsParsingException {
    Map<String, List<String>> result = new LinkedHashMap<>();
    Converters.StringToStringListConverter converter = new Converters.StringToStringListConverter();
    for (String s : strategies) {
      Map.Entry<String, List<String>> converted = converter.convert(s);
      // Have to avoid using Immutable* to allow overwriting elements.
      result.put(converted.getKey(), Lists.newArrayList(converted.getValue()));
    }
    return result;
  }
}
