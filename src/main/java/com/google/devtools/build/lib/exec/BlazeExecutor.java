// Copyright 2014 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.actions.ActionContext;
import com.google.devtools.build.lib.actions.ActionExecutionContext.ShowSubcommands;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.ExecutorInitException;
import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.common.options.OptionsProvider;
import javax.annotation.Nullable;

/**
 * The Executor class provides a dynamic abstraction of the various actual primitive system
 * operations that might be performed during a build step.
 *
 * <p>Constructions of this class might perform distributed execution, "virtual" execution for
 * testing purposes, or just print out the sequence of commands that would be executed, like Make's
 * "-n" option.
 */
@ThreadSafe
public final class BlazeExecutor implements Executor {

  private final boolean verboseFailures;
  private final ShowSubcommands showSubcommands;
  private final FileSystem fileSystem;
  private final Path execRoot;
  private final Clock clock;
  private final OptionsProvider options;
  private final ActionContext.ActionContextRegistry actionContextRegistry;

  /**
   * Constructs an Executor, bound to a specified output base path, and which will use the specified
   * reporter to announce SUBCOMMAND events, the given event bus to delegate events and the given
   * output streams for streaming output. The list of strategy implementation classes is used to
   * construct instances of the strategies mapped by their declared abstract type. This list is
   * uniquified before using. Each strategy instance is created with a reference to this Executor as
   * well as the given options object.
   *
   * <p>Don't forget to call startBuildRequest() and stopBuildRequest() for each request, and
   * shutdown() when you're done with this executor.
   */
  public BlazeExecutor(
      FileSystem fileSystem,
      Path execRoot,
      Reporter reporter,
      Clock clock,
      OptionsProvider options,
      SpawnActionContextMaps spawnActionContextMaps,
      Iterable<ActionContextProvider> contextProviders)
      throws ExecutorInitException {
    ExecutionOptions executionOptions = options.getOptions(ExecutionOptions.class);
    this.verboseFailures = executionOptions.verboseFailures;
    this.showSubcommands = executionOptions.showSubcommands;
    this.fileSystem = fileSystem;
    this.execRoot = execRoot;
    this.clock = clock;
    this.options = options;
    this.actionContextRegistry = spawnActionContextMaps;

    if (executionOptions.debugPrintActionContexts) {
      spawnActionContextMaps.debugPrintSpawnActionContextMaps(reporter);
    }

    spawnActionContextMaps.notifyUsed();

    for (ActionContextProvider factory : contextProviders) {
      factory.executorCreated();
    }
  }

  @Override
  public FileSystem getFileSystem() {
    return fileSystem;
  }

  @Override
  public Path getExecRoot() {
    return execRoot;
  }

  @Override
  public Clock getClock() {
    return clock;
  }

  @Override
  public ShowSubcommands reportsSubcommands() {
    return showSubcommands;
  }

  @Override
  @Nullable
  public <T extends ActionContext> T getContext(Class<T> type) {
    return actionContextRegistry.getContext(type);
  }

  /** Returns true iff the --verbose_failures option was enabled. */
  @Override
  public boolean getVerboseFailures() {
    return verboseFailures;
  }

  /** Returns the options associated with the execution. */
  @Override
  public OptionsProvider getOptions() {
    return options;
  }
}
