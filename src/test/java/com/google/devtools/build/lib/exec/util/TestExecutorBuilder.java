// Copyright 2009 The Bazel Authors. All Rights Reserved.
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
package com.google.devtools.build.lib.exec.util;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.ActionContext;
import com.google.devtools.build.lib.actions.ExecutorInitException;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.actions.LocalTemplateExpansionStrategy;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.exec.ActionContextProvider;
import com.google.devtools.build.lib.exec.BinTools;
import com.google.devtools.build.lib.exec.BlazeExecutor;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.exec.FileWriteStrategy;
import com.google.devtools.build.lib.exec.SpawnActionContextMaps;
import com.google.devtools.build.lib.exec.SymlinkTreeStrategy;
import com.google.devtools.build.lib.runtime.CommonCommandOptions;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * Builder for the test instance of the {@link BlazeExecutor} class.
 */
public class TestExecutorBuilder {
  public static final ImmutableList<Class<? extends OptionsBase>> DEFAULT_OPTIONS =
      ImmutableList.of(ExecutionOptions.class, CommonCommandOptions.class);
  private final FileSystem fileSystem;
  private final BlazeDirectories directories;
  private EventBus bus = new EventBus();
  private Reporter reporter = new Reporter(bus);
  private OptionsParser optionsParser = OptionsParser.newOptionsParser(DEFAULT_OPTIONS);
  private List<ActionContext> strategies = new ArrayList<>();
  private final Map<String, List<SpawnActionContext>> spawnStrategyMap =
      new TreeMap<>(String.CASE_INSENSITIVE_ORDER);

  public TestExecutorBuilder(
      FileSystem fileSystem, BlazeDirectories directories, BinTools binTools) {
    this.fileSystem = fileSystem;
    this.directories = directories;
    strategies.add(new FileWriteStrategy());
    strategies.add(new LocalTemplateExpansionStrategy());
    strategies.add(new SymlinkTreeStrategy(null, binTools));
  }

  public TestExecutorBuilder setReporter(Reporter reporter) {
    this.reporter = reporter;
    return this;
  }

  public TestExecutorBuilder setBus(EventBus bus) {
    this.bus = bus;
    return this;
  }

  public TestExecutorBuilder setOptionsParser(OptionsParser optionsParser) {
    this.optionsParser = optionsParser;
    return this;
  }

  public TestExecutorBuilder parseOptions(String... options) throws OptionsParsingException {
    this.optionsParser.parse(options);
    return this;
  }

  public TestExecutorBuilder addStrategy(ActionContext strategy) {
    strategies.add(strategy);
    return this;
  }

  public TestExecutorBuilder addStrategyFactory(ActionContextProvider factory) {
    Iterables.addAll(strategies, factory.getActionContexts());
    return this;
  }

  public TestExecutorBuilder setExecution(String mnemonic, SpawnActionContext strategy) {
    spawnStrategyMap.put(mnemonic, ImmutableList.of(strategy));
    strategies.add(strategy);
    return this;
  }

  public BlazeExecutor build() throws ExecutorInitException {
    return new BlazeExecutor(
        fileSystem,
        directories.getExecRoot(TestConstants.WORKSPACE_NAME),
        reporter,
        bus,
        BlazeClock.instance(),
        optionsParser,
        SpawnActionContextMaps.createStub(strategies, spawnStrategyMap),
        ImmutableList.of());
  }
}
