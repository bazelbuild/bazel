// Copyright 2016 The Bazel Authors. All Rights Reserved.
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

import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.ActionContext;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.SpawnStrategy;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.exec.util.TestExecutorBuilder;
import com.google.devtools.build.lib.testutil.Suite;
import com.google.devtools.build.lib.testutil.TestSpec;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.common.options.OptionsParser;
import javax.annotation.Nullable;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link BlazeExecutor}. */
@RunWith(JUnit4.class)
@TestSpec(size = Suite.SMALL_TESTS)
public class BlazeExecutorTest {
  private FileSystem fileSystem;
  private BlazeDirectories directories;
  private BinTools binTools;

  @Before
  public final void setUpDirectoriesAndTools() throws Exception {
    fileSystem = new InMemoryFileSystem();
    directories =
        new BlazeDirectories(
            new ServerDirectories(
                fileSystem.getPath("/install"),
                fileSystem.getPath("/base"),
                fileSystem.getPath("/root")),
            fileSystem.getPath("/workspace"),
            /* defaultSystemJavabase= */ null,
            "mock-product-name");
    binTools = BinTools.empty(directories);
  }

  @Test
  public void testDebugPrintActionContexts() throws Exception {
    OptionsParser parser =
        OptionsParser.builder().optionsClasses(TestExecutorBuilder.DEFAULT_OPTIONS).build();
    parser.parse("--debug_print_action_contexts");

    Reporter reporter = new Reporter(new EventBus());
    StoredEventHandler storedEventHandler = new StoredEventHandler();
    reporter.addHandler(storedEventHandler);

    FakeSpawnStrategy strategy = new FakeSpawnStrategy();

    new TestExecutorBuilder(fileSystem, directories, binTools)
        .setReporter(reporter)
        .setOptionsParser(parser)
        .setExecution("fake", "fake")
        .addStrategy(SpawnStrategy.class, new FakeSpawnStrategy(), "fake")
        .build();

    Event event =
        Iterables.find(
            storedEventHandler.getEvents(),
            new Predicate<Event>() {
              @Override
              public boolean apply(@Nullable Event event) {
                return event.getMessage().contains("SpawnActionContextMap: \"fake\" = ");
              }
            });
    assertThat(event).isNotNull();
    assertThat(event.getMessage())
        .contains("\"fake\" = [" + strategy.getClass().getSimpleName() + "]");
  }

  private static class FakeSpawnStrategy implements SpawnStrategy {

    @Override
    public ImmutableList<SpawnResult> exec(
        Spawn spawn, ActionExecutionContext actionExecutionContext) {
      throw new UnsupportedOperationException();
    }

    @Override
    public boolean canExec(Spawn spawn, ActionContext.ActionContextRegistry actionContextRegistry) {
      return false;
    }
  }
}
