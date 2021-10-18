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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Lists;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.ActionContext.ActionContextRegistry;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.BaseSpawn;
import com.google.devtools.build.lib.actions.DynamicStrategyRegistry.DynamicMode;
import com.google.devtools.build.lib.actions.EmptyRunfilesSupplier;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.SandboxedSpawnStrategy;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnContinuation;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil.NullAction;
import com.google.devtools.build.lib.buildtool.BuildResult;
import com.google.devtools.build.lib.buildtool.buildevent.BuildCompleteEvent;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.exec.SpawnStrategyRegistry;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.util.FileSystems;
import com.google.devtools.common.options.Converters;
import com.google.devtools.common.options.OptionsParsingException;
import java.io.IOException;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;
import javax.annotation.Nullable;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link com.google.devtools.build.lib.dynamic.DynamicExecutionModule}. */
@RunWith(JUnit4.class)
public class DynamicExecutionModuleTest {
  private DynamicExecutionModule module;
  private DynamicExecutionOptions options;
  private Path testRoot;

  @Before
  public void setUp() throws IOException {
    testRoot = TestUtils.createUniqueTmpDir(FileSystems.getNativeFileSystem());
    module = new DynamicExecutionModule(Executors.newCachedThreadPool());
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
  public void testRegisterSpawnStrategies_unsetsLocalStrategiesOnFirstBuild()
      throws AbruptExitException, OptionsParsingException, UserExecException {
    options.skipFirstBuild = true;
    options.internalSpawnScheduler = true;
    options.dynamicLocalStrategy = parseStrategiesToOptions("Foo=local", "local");
    options.dynamicRemoteStrategy = parseStrategiesToOptions("Foo=remote", "remote");
    SpawnStrategyRegistry.Builder registryBuilder = new SpawnStrategyRegistry.Builder();
    registerStrategy(registryBuilder, "local");
    registerStrategy(registryBuilder, "remote");
    // "dynamic" strategy will  be added by registerSpawnStrategies().

    Reporter reporter = new Reporter(new EventBus());
    Spawn spawn = newCustomSpawn("Foo", ImmutableMap.of());

    // First build skips dynamic execution
    module.registerSpawnStrategies(registryBuilder, options, reporter);
    assertThat(registryBuilder.build().getDynamicSpawnActionContexts(spawn, DynamicMode.LOCAL))
        .isEmpty();

    // Pretend that the build failed
    BuildResult buildResult = new BuildResult(1L);
    buildResult.setDetailedExitCode(
        DetailedExitCode.of(ExitCode.ANALYSIS_FAILURE, FailureDetail.newBuilder().build()));
    module.buildCompleteEvent(new BuildCompleteEvent(buildResult));

    // Second build, after no successful builds, skips dynamic execution.
    module.registerSpawnStrategies(registryBuilder, options, reporter);
    assertThat(registryBuilder.build().getDynamicSpawnActionContexts(spawn, DynamicMode.LOCAL))
        .isEmpty();

    // Pretend that the build succeeded
    buildResult.setDetailedExitCode(DetailedExitCode.success());
    module.buildCompleteEvent(new BuildCompleteEvent(buildResult));

    // Third build - since the second succeeded, we shouldn't skip any more.
    module.registerSpawnStrategies(registryBuilder, options, reporter);
    assertThat(registryBuilder.build().getDynamicSpawnActionContexts(spawn, DynamicMode.LOCAL))
        .isNotEmpty();
  }

  private void registerStrategy(SpawnStrategyRegistry.Builder registryBuilder, String name) {
    registryBuilder.registerStrategy(
        new SandboxedSpawnStrategy() {
          @Override
          public ImmutableList<SpawnResult> exec(
              Spawn spawn,
              ActionExecutionContext actionExecutionContext,
              @Nullable StopConcurrentSpawns stopConcurrentSpawns) {
            return ImmutableList.of();
          }

          @Override
          public ImmutableList<SpawnResult> exec(
              Spawn spawn, ActionExecutionContext actionExecutionContext) {
            return ImmutableList.of();
          }

          @Override
          public SpawnContinuation beginExecution(
              Spawn spawn, ActionExecutionContext actionExecutionContext)
              throws InterruptedException {
            return SandboxedSpawnStrategy.super.beginExecution(spawn, actionExecutionContext);
          }

          @Override
          public boolean canExec(Spawn spawn, ActionContextRegistry actionContextRegistry) {
            return false;
          }

          @Override
          public boolean canExecWithLegacyFallback(
              Spawn spawn, ActionContextRegistry actionContextRegistry) {
            return SandboxedSpawnStrategy.super.canExecWithLegacyFallback(
                spawn, actionContextRegistry);
          }

          @Override
          public void usedContext(ActionContextRegistry actionContextRegistry) {
            SandboxedSpawnStrategy.super.usedContext(actionContextRegistry);
          }

          @Override
          public String toString() {
            return "Test strategy \"" + name + "\"";
          }
        },
        name);
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

  /** Constructs a new spawn with a custom mnemonic and execution info. */
  Spawn newCustomSpawn(String mnemonic, ImmutableMap<String, String> executionInfo) {
    Artifact inputArtifact =
        ActionsTestUtil.createArtifact(
            ArtifactRoot.asSourceRoot(Root.fromPath(testRoot)), "input.txt");
    Artifact outputArtifact =
        ActionsTestUtil.createArtifact(
            ArtifactRoot.asSourceRoot(Root.fromPath(testRoot)), "output.txt");

    ActionExecutionMetadata action =
        new NullActionWithMnemonic(mnemonic, ImmutableList.of(inputArtifact), outputArtifact);
    return new BaseSpawn(
        ImmutableList.of(),
        ImmutableMap.of(),
        executionInfo,
        EmptyRunfilesSupplier.INSTANCE,
        action,
        ResourceSet.create(1, 0, 0));
  }

  private static class NullActionWithMnemonic extends NullAction {
    private final String mnemonic;

    private NullActionWithMnemonic(String mnemonic, List<Artifact> inputs, Artifact... outputs) {
      super(inputs, outputs);
      this.mnemonic = mnemonic;
    }

    @Override
    public String getMnemonic() {
      return mnemonic;
    }
  }
}
