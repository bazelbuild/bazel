// Copyright 2018 The Bazel Authors. All Rights Reserved.
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
package com.google.devtools.build.lib.exec;

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.testutil.Suite;
import com.google.devtools.build.lib.testutil.TestSpec;
import com.google.devtools.build.lib.util.RegexFilter.RegexFilterConverter;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mockito;

/** Tests of {@link SpawnActionContextMaps}. */
@RunWith(JUnit4.class)
@TestSpec(size = Suite.SMALL_TESTS)
public class SpawnActionContextMapsTest {

  private SpawnActionContextMaps.Builder builder;
  private final RegexFilterConverter converter = new RegexFilterConverter();
  private final EventBus bus = new EventBus();
  private final Reporter reporter = new Reporter(bus);

  private static final AC1 ac1 = new AC1();
  private static final AC2 ac2 = new AC2();

  private static final ImmutableList<ActionContextProvider> PROVIDERS =
      ImmutableList.of(
          new ActionContextProvider() {
            @Override
            public void registerActionContexts(ActionContextCollector collector) {
              collector
                  .forType(SpawnActionContext.class)
                  .registerContext(ac1, "ac1")
                  .registerContext(ac2, "ac2");
            }
          });

  @Before
  public void setUp() {
    builder = new SpawnActionContextMaps.Builder();
  }

  @Test
  public void duplicateMnemonics_bothGetStored() throws Exception {
    builder.strategyByMnemonicMap().put("Spawn1", "ac1");
    builder.strategyByMnemonicMap().put("Spawn1", "ac2");
    SpawnActionContextMaps maps = builder.build(PROVIDERS);
    List<SpawnActionContext> result =
        maps.getSpawnActionContexts(mockSpawn("Spawn1", null), reporter);
    assertThat(result).containsExactly(ac1, ac2);
  }

  @Test
  public void emptyStrategyFallsBackToEmptyMnemonicNotToDefault() throws Exception {
    builder.strategyByMnemonicMap().put("Spawn1", "");
    builder.strategyByMnemonicMap().put("", "ac2");
    SpawnActionContextMaps maps = builder.build(PROVIDERS);
    List<SpawnActionContext> result =
        maps.getSpawnActionContexts(mockSpawn("Spawn1", null), reporter);
    assertThat(result).containsExactly(ac2);
  }

  @Test
  public void multipleRegexps_firstMatchWins() throws Exception {
    builder.addStrategyByRegexp(converter.convert("foo"), ImmutableList.of("ac1"));
    builder.addStrategyByRegexp(converter.convert("foo/bar"), ImmutableList.of("ac2"));
    SpawnActionContextMaps maps = builder.build(PROVIDERS);

    List<SpawnActionContext> result =
        maps.getSpawnActionContexts(mockSpawn(null, "Doing something with foo/bar/baz"), reporter);

    assertThat(result).containsExactly(ac1);
  }

  @Test
  public void regexpAndMnemonic_regexpWins() throws Exception {
    builder.strategyByMnemonicMap().put("Spawn1", "ac1");
    builder.addStrategyByRegexp(converter.convert("foo/bar"), ImmutableList.of("ac2"));
    SpawnActionContextMaps maps = builder.build(PROVIDERS);

    List<SpawnActionContext> result =
        maps.getSpawnActionContexts(
            mockSpawn("Spawn1", "Doing something with foo/bar/baz"), reporter);

    assertThat(result).containsExactly(ac2);
  }

  @Test
  public void duplicateContext_noException() throws Exception {
    builder.strategyByContextMap().put(AC1.class, "one");
    builder.strategyByContextMap().put(AC1.class, "two");
    builder.strategyByContextMap().put(AC1.class, "");
  }

  private Spawn mockSpawn(String mnemonic, String message) {
    Spawn mockSpawn = Mockito.mock(Spawn.class);
    ActionExecutionMetadata mockOwner = Mockito.mock(ActionExecutionMetadata.class);
    when(mockOwner.getProgressMessage()).thenReturn(message);
    when(mockSpawn.getResourceOwner()).thenReturn(mockOwner);
    when(mockSpawn.getMnemonic()).thenReturn(mnemonic);
    return mockSpawn;
  }

  private static class AC1 implements SpawnActionContext {
    @Override
    public ImmutableList<SpawnResult> exec(
        Spawn spawn, ActionExecutionContext actionExecutionContext)
        throws ExecException, InterruptedException {
      throw new UnsupportedOperationException();
    }

    @Override
    public boolean canExec(Spawn spawn, ActionContextRegistry actionContextRegistry) {
      return true;
    }
  }

  private static class AC2 implements SpawnActionContext {
    @Override
    public ImmutableList<SpawnResult> exec(
        Spawn spawn, ActionExecutionContext actionExecutionContext)
        throws ExecException, InterruptedException {
      throw new UnsupportedOperationException();
    }

    @Override
    public boolean canExec(Spawn spawn, ActionContextRegistry actionContextRegistry) {
      return true;
    }
  }
}
