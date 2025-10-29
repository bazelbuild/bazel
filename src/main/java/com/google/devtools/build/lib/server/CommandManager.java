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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.server.CommandProtos.CancelRequest;
import com.google.devtools.build.lib.server.IdleServerTasks.IdleServerCleanupStrategy;
import com.google.devtools.build.lib.util.ThreadUtils;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicLong;
import javax.annotation.Nullable;
import javax.annotation.concurrent.GuardedBy;

/** Helper class for commands that are currently running on the server. */
class CommandManager {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  @GuardedBy("runningCommandsMap")
  private final Map<String, RunningCommand> runningCommandsMap = new HashMap<>();

  private final AtomicLong interruptCounter = new AtomicLong(0);
  private final boolean doIdleServerTasks;

  private IdleServerTasks idleServerTasks;
  @Nullable private final String slowInterruptMessageSuffix;

  CommandManager(boolean doIdleServerTasks, @Nullable String slowInterruptMessageSuffix) {
    this.doIdleServerTasks = doIdleServerTasks;
    this.slowInterruptMessageSuffix = slowInterruptMessageSuffix;
    idle(IdleServerCleanupStrategy.DELAYED);
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

  private void idle(IdleServerCleanupStrategy cleanupStrategy) {
    Preconditions.checkState(idleServerTasks == null);
    if (doIdleServerTasks) {
      idleServerTasks = new IdleServerTasks();
      idleServerTasks.idle(cleanupStrategy);
    }
  }

  private void busy() {
    if (doIdleServerTasks) {
      Preconditions.checkState(idleServerTasks != null);
      idleServerTasks.busy();
      idleServerTasks = null;
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

  final class RunningCommand implements AutoCloseable {
    private final Thread thread;
    private final String id;
    private final boolean preemptible;
    private IdleServerCleanupStrategy cleanupStrategy = IdleServerCleanupStrategy.DELAYED;

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
          idle(cleanupStrategy);
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

    /** Requests a manual GC as soon as the server becomes idle. */
    void requestEagerIdleServerCleanup() {
      cleanupStrategy = IdleServerCleanupStrategy.EAGER;
    }
  }
}
