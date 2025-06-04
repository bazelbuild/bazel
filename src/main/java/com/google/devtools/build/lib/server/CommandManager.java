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

package com.google.devtools.build.lib.server;

import static com.google.common.base.Preconditions.checkState;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.server.CommandProtos.CancelRequest;
import com.google.devtools.build.lib.util.ThreadUtils;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicLong;
import javax.annotation.Nullable;
import javax.annotation.concurrent.GuardedBy;

/** Helper class for commands that are currently running on the server. */
class CommandManager {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  /**
   * The list of currently running commands. Note that, even though most commands run serially
   * because of the output base lock, they're registered here before blocking for the lock, so the
   * map is effectively unbounded.
   */
  @GuardedBy("runningCommandsMap")
  private final Map<String, RunningCommand> runningCommandsMap = new HashMap<>();

  /** Whether idle tasks are enabled. */
  private final boolean doIdleServerTasks;

  /** The current IdleTaskManager. Null when a command is running or if idle tasks are disabled. */
  @GuardedBy("this")
  @Nullable
  private IdleTaskManager idleTaskManager;

  /**
   * Idle task results from the most recent idle period following a command that registered idle
   * tasks. Null after a subsequent command retrieves them or if idle tasks are disabled.
   */
  @GuardedBy("this")
  @Nullable
  private ImmutableList<IdleTask.Result> idleTaskResults;

  private final AtomicLong interruptCounter = new AtomicLong(0);
  @Nullable private final String slowInterruptMessageSuffix;

  CommandManager(boolean doIdleServerTasks, @Nullable String slowInterruptMessageSuffix) {
    this.doIdleServerTasks = doIdleServerTasks;
    this.slowInterruptMessageSuffix = slowInterruptMessageSuffix;
    idle(Optional.empty());
  }

  void preemptEligibleCommands() {
    synchronized (runningCommandsMap) {
      ImmutableSet.Builder<String> commandsToInterruptBuilder = new ImmutableSet.Builder<>();

      for (RunningCommand command : runningCommandsMap.values()) {
        if (command.isPreemptible()) {
          command.thread.interrupt();
          commandsToInterruptBuilder.add(command.id);
        }
      }

      ImmutableSet<String> commandsToInterrupt = commandsToInterruptBuilder.build();
      if (!commandsToInterrupt.isEmpty()) {
        startSlowInterruptWatcher(commandsToInterrupt);
      }
    }
  }

  void interruptInflightCommands() {
    synchronized (runningCommandsMap) {
      for (RunningCommand command : runningCommandsMap.values()) {
        command.thread.interrupt();
      }

      startSlowInterruptWatcher(ImmutableSet.copyOf(runningCommandsMap.keySet()));
    }
  }

  void doCancel(CancelRequest request) {
    try (RunningCommand cancelCommand = createCommand()) {
      synchronized (runningCommandsMap) {
        RunningCommand pendingCommand = runningCommandsMap.get(request.getCommandId());
        if (pendingCommand != null) {
          logger.atInfo().log(
              "Interrupting command %s on thread %s",
              request.getCommandId(), pendingCommand.thread.getName());
          pendingCommand.thread.interrupt();
          startSlowInterruptWatcher(ImmutableSet.of(request.getCommandId()));
        } else {
          logger.atInfo().log("Cannot find command %s to interrupt", request.getCommandId());
        }
      }
    }
  }

  boolean isEmpty() {
    synchronized (runningCommandsMap) {
      return runningCommandsMap.isEmpty();
    }
  }

  void waitForChange() throws InterruptedException {
    synchronized (runningCommandsMap) {
      runningCommandsMap.wait();
    }
  }

  void waitForChange(long timeout) throws InterruptedException {
    synchronized (runningCommandsMap) {
      runningCommandsMap.wait(timeout);
    }
  }

  RunningCommand createPreemptibleCommand() {
    RunningCommand command = new RunningCommand(true);
    registerCommand(command);
    return command;
  }

  RunningCommand createCommand() {
    RunningCommand command = new RunningCommand(false);
    registerCommand(command);
    return command;
  }

  private void registerCommand(RunningCommand command) {
    synchronized (runningCommandsMap) {
      if (runningCommandsMap.isEmpty()) {
        busy();
      }
      runningCommandsMap.put(command.id, command);
      runningCommandsMap.notify();
    }
    logger.atInfo().log("Starting command %s on thread %s", command.id, command.thread.getName());
  }

  /**
   * Enters an idle period.
   *
   * <p>Called when the set of running commands becomes empty.
   *
   * @param idleTasks idle tasks to run during the idle period, if any.
   */
  private void idle(Optional<ImmutableList<IdleTask>> idleTasks) {
    if (doIdleServerTasks && idleTasks.isPresent()) {
      synchronized (this) {
        checkState(idleTaskManager == null);
        idleTaskManager = new IdleTaskManager(idleTasks.get());
        idleTaskManager.idle();
      }
    }
  }

  /**
   * Leaves an idle period.
   *
   * <p>Called when the set of running commands becomes non-empty.
   */
  private void busy() {
    synchronized (this) {
      if (idleTaskManager != null) {
        idleTaskResults = idleTaskManager.busy();
        idleTaskManager = null;
      }
    }
  }

  private void startSlowInterruptWatcher(final ImmutableSet<String> commandIds) {
    if (commandIds.isEmpty()) {
      return;
    }

    Runnable interruptWatcher =
        () -> {
          try {
            Thread.sleep(10 * 1000);
            boolean ok;
            synchronized (runningCommandsMap) {
              ok = Collections.disjoint(commandIds, runningCommandsMap.keySet());
            }
            if (!ok) {
              // At least one command was not interrupted. Interrupt took too long.
              ThreadUtils.warnAboutSlowInterrupt(slowInterruptMessageSuffix);
            }
          } catch (InterruptedException e) {
            // Ignore.
          }
        };

    Thread interruptWatcherThread =
        new Thread(interruptWatcher, "interrupt-watcher-" + interruptCounter.incrementAndGet());
    interruptWatcherThread.setDaemon(true);
    interruptWatcherThread.start();
  }

  /**
   * Returns idle task results returned by {@link IdleTaskManager} during a previous idle period, if
   * available and not yet retrieved.
   *
   * <p>Clears the stored idle task results as a side effect.
   */
  @Nullable
  public synchronized ImmutableList<IdleTask.Result> getIdleTaskResults() {
    var result = idleTaskResults;
    idleTaskResults = null;
    return result;
  }

  final class RunningCommand implements AutoCloseable {
    private final Thread thread;
    private final String id;
    private final boolean preemptible;
    private Optional<ImmutableList<IdleTask>> idleTasks = Optional.empty();

    private RunningCommand(boolean preemptible) {
      thread = Thread.currentThread();
      id = UUID.randomUUID().toString();
      this.preemptible = preemptible;
    }

    @Override
    public void close() {
      synchronized (runningCommandsMap) {
        runningCommandsMap.remove(id);
        if (runningCommandsMap.isEmpty()) {
          idle(idleTasks);
        }
        runningCommandsMap.notify();
      }

      logger.atInfo().log("Finished command %s on thread %s", id, thread.getName());
    }

    String getId() {
      return id;
    }

    boolean isPreemptible() {
      return preemptible;
    }

    /**
     * Set idle tasks to be run by {@link IdleTaskManager} during an idle period immediately
     * following this command, if one occurs and idle tasks are enabled.
     */
    void setIdleTasks(ImmutableList<IdleTask> idleTasks) {
      this.idleTasks = Optional.of(idleTasks);
    }
  }
}
