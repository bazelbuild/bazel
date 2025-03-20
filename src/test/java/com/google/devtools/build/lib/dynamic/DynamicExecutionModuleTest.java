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
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Lists;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.exec.BinTools;
import com.google.devtools.build.lib.exec.util.SpawnBuilder;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.common.options.Converters;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.devtools.common.options.OptionsParsingResult;
import java.io.IOException;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link com.google.devtools.build.lib.dynamic.DynamicExecutionModule}. */
@RunWith(JUnit4.class)
public class DynamicExecutionModuleTest {
  private static final PathFragment OUTPUT_BASE = PathFragment.create("blaze-out");

  private DynamicExecutionModule module;
  private DynamicExecutionOptions options;
  private BlazeRuntime blazeRuntime;

  @Before
  public void setUp() throws IOException, AbruptExitException {
    module = new DynamicExecutionModule(Executors.newCachedThreadPool());
    options = new DynamicExecutionOptions();
    options.dynamicLocalStrategy = Collections.emptyList(); // default
    options.dynamicRemoteStrategy = Collections.emptyList(); // default
  }

  @Test
  public void testGetLocalStrategies_getsDefaultWithNoOptions()
      throws AbruptExitException, OptionsParsingException {
    assertThat(module.getLocalStrategies(options, /* sandboxingSupported= */ true))
        .isEqualTo(parseStrategies("worker,sandboxed"));
    assertThat(module.getLocalStrategies(options, /* sandboxingSupported= */ false))
        .isEqualTo(parseStrategies("worker"));
  }

  @Test
  public void testGetLocalStrategies_genericOptionOverridesFallbacks()
      throws AbruptExitException, OptionsParsingException {
    options.dynamicLocalStrategy = parseStrategiesToOptions("local,worker");
    assertThat(module.getLocalStrategies(options, /* sandboxingSupported= */ true))
        .isEqualTo(parseStrategies("local,worker"));
  }

  @Test
  public void testGetLocalStrategies_specificOptionKeepsFallbacks()
      throws AbruptExitException, OptionsParsingException {
    options.dynamicLocalStrategy = parseStrategiesToOptions("Foo=local,worker");
    assertThat(module.getLocalStrategies(options, /* sandboxingSupported= */ true))
        .isEqualTo(parseStrategies("Foo=local,worker", "worker,sandboxed"));
  }

  @Test
  public void testGetLocalStrategies_canMixSpecificsAndGenericOptions()
      throws AbruptExitException, OptionsParsingException {
    options.dynamicLocalStrategy = parseStrategiesToOptions("Foo=local,worker", "worker");
    assertThat(module.getLocalStrategies(options, /* sandboxingSupported= */ true))
        .isEqualTo(parseStrategies("Foo=local,worker", "worker"));
  }

  @Test
  public void canIgnoreFailure_simpleCases() throws IOException, AbruptExitException {
    setupRuntime();
    Spawn spawn = new SpawnBuilder().withOutput("output").build();
    CommandEnvironment mockCommandEnvironment = mock(CommandEnvironment.class);
    OptionsParsingResult mockOptions = mock(OptionsParsingResult.class);
    when(mockCommandEnvironment.getOptions()).thenReturn(mockOptions);
    EventBus mockEventBus = mock(EventBus.class);
    when(mockCommandEnvironment.getEventBus()).thenReturn(mockEventBus);
    when(mockCommandEnvironment.getBlazeWorkspace()).thenReturn(blazeRuntime.getWorkspace());
    DynamicExecutionOptions options = new DynamicExecutionOptions();
    when(mockOptions.getOptions(DynamicExecutionOptions.class)).thenReturn(options);
    ActionExecutionContext context = mock(ActionExecutionContext.class);

    options.ignoreLocalSignals = ImmutableSet.of();
    module.beforeCommand(mockCommandEnvironment);
    assertThat(module.canIgnoreFailure(spawn, context, 130, "Failed", null, true)).isFalse();

    options.ignoreLocalSignals = ImmutableSet.of(9);
    module.beforeCommand(mockCommandEnvironment);
    assertThat(module.canIgnoreFailure(spawn, context, 130, "Failed", null, true)).isFalse();

    options.ignoreLocalSignals = ImmutableSet.of(2, 9);
    module.beforeCommand(mockCommandEnvironment);
    assertThat(module.canIgnoreFailure(spawn, context, 130, "Failed", null, false)).isFalse();
    assertThat(module.canIgnoreFailure(spawn, context, 0, "Failed", null, true)).isFalse();
    assertThat(module.canIgnoreFailure(spawn, context, 130, "Failed", null, true)).isTrue();
    assertThat(module.canIgnoreFailure(spawn, context, 137, "Failed", null, true)).isTrue();
  }

  private void setupRuntime() throws IOException, AbruptExitException {
    Scratch scratch = new Scratch();
    Path execDir = scratch.dir("/foo");
    Root root = Root.fromPath(execDir);
    ServerDirectories serverDirectories =
        new ServerDirectories(
            scratch.dir("/installBase"),
            root.getRelative(OUTPUT_BASE),
            scratch.dir("/output-user"));
    blazeRuntime =
        new BlazeRuntime.Builder()
            .setFileSystem(scratch.getFileSystem())
            .setProductName(TestConstants.PRODUCT_NAME)
            .setServerDirectories(serverDirectories)
            .setStartupOptionsProvider(mock(OptionsParsingResult.class))
            .build();
    BinTools binTools = BinTools.forUnitTesting(execDir, ImmutableList.of());
    blazeRuntime.initWorkspace(
        new BlazeDirectories(
            serverDirectories,
            scratch.dir(TestConstants.WORKSPACE_NAME),
            null,
            TestConstants.PRODUCT_NAME),
        binTools);
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
