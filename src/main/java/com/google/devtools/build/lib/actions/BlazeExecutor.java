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
package com.google.devtools.build.lib.actions;

import com.google.common.collect.ImmutableMap;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.util.Clock;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.common.options.OptionsClassProvider;

import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

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

  private final Path outputPath;
  private final boolean verboseFailures;
  private final boolean showSubcommands;
  private final Path execRoot;
  private final Reporter reporter;
  private final EventBus eventBus;
  private final Clock clock;
  private final OptionsClassProvider options;
  private AtomicBoolean inExecutionPhase;

  private final Map<String, SpawnActionContext> spawnActionContextMap;
  private final Map<Class<? extends ActionContext>, ActionContext> contextMap =
      new HashMap<>();

  /**
   * Constructs an Executor, bound to a specified output base path, and which
   * will use the specified reporter to announce SUBCOMMAND events,
   * the given event bus to delegate events and the given output streams
   * for streaming output. The list of
   * strategy implementation classes is used to construct instances of the
   * strategies mapped by their declared abstract type. This list is uniquified
   * before using. Each strategy instance is created with a reference to this
   * Executor as well as the given options object.
   * <p>
   * Don't forget to call startBuildRequest() and stopBuildRequest() for each
   * request, and shutdown() when you're done with this executor.
   */
  public BlazeExecutor(Path execRoot,
      Path outputPath,
      Reporter reporter,
      EventBus eventBus,
      Clock clock,
      OptionsClassProvider options,
      boolean verboseFailures,
      boolean showSubcommands,
      List<ActionContext> contextImplementations,
      Map<String, SpawnActionContext> spawnActionContextMap,
      Iterable<ActionContextProvider> contextProviders)
      throws ExecutorInitException {
    this.outputPath = outputPath;
    this.verboseFailures = verboseFailures;
    this.showSubcommands = showSubcommands;
    this.execRoot = execRoot;
    this.reporter = reporter;
    this.eventBus = eventBus;
    this.clock = clock;
    this.options = options;
    this.inExecutionPhase = new AtomicBoolean(false);

    // We need to keep only the last occurrences of the entries in contextImplementations
    // (so we respect insertion order but also instantiate them only once).
    LinkedHashSet<ActionContext> allContexts = new LinkedHashSet<>();
    allContexts.addAll(contextImplementations);
    allContexts.addAll(spawnActionContextMap.values());
    this.spawnActionContextMap = ImmutableMap.copyOf(spawnActionContextMap);

    for (ActionContext context : contextImplementations) {
      ExecutionStrategy annotation = context.getClass().getAnnotation(ExecutionStrategy.class);
      if (annotation != null) {
        contextMap.put(annotation.contextType(), context);
      }
      contextMap.put(context.getClass(), context);
    }

    for (ActionContextProvider factory : contextProviders) {
      factory.executorCreated(allContexts);
    }
  }

  @Override
  public Path getExecRoot() {
    return execRoot;
  }

  @Override
  public EventHandler getEventHandler() {
    return reporter;
  }

  @Override
  public EventBus getEventBus() {
    return eventBus;
  }

  @Override
  public Clock getClock() {
    return clock;
  }

  @Override
  public boolean reportsSubcommands() {
    return showSubcommands;
  }

  /**
   * Report a subcommand event to this Executor's Reporter and, if action
   * logging is enabled, post it on its EventBus.
   */
  @Override
  public void reportSubcommand(String reason, String message) {
    reporter.handle(new Event(EventKind.SUBCOMMAND, null, "# " + reason + "\n" + message));
  }

  /**
   * This method is called before the start of the execution phase of each
   * build request.
   */
  public void executionPhaseStarting() {
    Preconditions.checkState(!inExecutionPhase.getAndSet(true));
    Profiler.instance().startTask(ProfilerTask.INFO, "Initializing executors");
    Profiler.instance().completeTask(ProfilerTask.INFO);
  }

  /**
   * This method is called after the end of the execution phase of each build
   * request (even if there was an interrupt).
   */
  public void executionPhaseEnding() {
    if (!inExecutionPhase.get()) {
      return;
    }

    Profiler.instance().startTask(ProfilerTask.INFO, "Shutting down executors");
    Profiler.instance().completeTask(ProfilerTask.INFO);
    inExecutionPhase.set(false);
  }

  public static void shutdownHelperPool(EventHandler reporter, ExecutorService pool,
      String name) {
    pool.shutdownNow();

    boolean interrupted = false;
    while (true) {
      try {
        if (!pool.awaitTermination(10, TimeUnit.SECONDS)) {
          reporter.handle(Event.warn(name + " threadpool shutdown took greater than ten seconds"));
        }
        break;
      } catch (InterruptedException e) {
        interrupted = true;
      }
    }

    if (interrupted) {
      Thread.currentThread().interrupt();
    }
  }

  @Override
  public <T extends ActionContext> T getContext(Class<? extends T> type) {
    Preconditions.checkArgument(type != SpawnActionContext.class, 
        "should use getSpawnActionContext instead");
    return type.cast(contextMap.get(type));
  }

  /**
   * Returns the {@link SpawnActionContext} to use for the given mnemonic. If no execution mode is
   * set, then it returns the default strategy for spawn actions.
   */
  @Override
  public SpawnActionContext getSpawnActionContext(String mnemonic) {
     SpawnActionContext context = spawnActionContextMap.get(mnemonic);
     return context == null ? spawnActionContextMap.get("") : context;
   }

  /** Returns true iff the --verbose_failures option was enabled. */
  @Override
  public boolean getVerboseFailures() {
    return verboseFailures;
  }

  /** Returns the options associated with the execution. */
  @Override
  public OptionsClassProvider getOptions() {
    return options;
  }

  public Path getOutputPath() {
    return outputPath;
  }
}
