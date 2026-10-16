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

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.server.CommandManager.RunningCommand;
import com.google.devtools.build.lib.server.ServerWatcherRunnable.ProcMeminfoLowMemoryChecker;
import com.google.devtools.build.lib.testutil.ManualClock;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.unix.ProcMeminfoParser;
import com.google.devtools.build.lib.util.OS;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ServerWatcherRunnable}. */
@RunWith(JUnit4.class)
public class ServerWatcherRunnableTest {
  private ManualClock clock;
  private GrpcCommandServer mockGrpcCommandServer;

  @Before
  public final void setManualClock() {
    clock = new ManualClock();
    mockGrpcCommandServer = mock(GrpcCommandServer.class);
    BlazeClock.setClock(clock);
  }

  @Test
  public void testBasicIdleCheck() throws Exception {
    CommandManager mockCommands = mock(CommandManager.class);
    ServerWatcherRunnable underTest =
        new ServerWatcherRunnable(
            mockGrpcCommandServer,
            /* maxIdleSeconds= */ 10,
            /* shutdownOnLowSysMem= */ false,
            mockCommands);
    Thread thread = new Thread(underTest);
    when(mockCommands.isEmpty()).thenReturn(true);
    AtomicInteger checkIdleCounter = new AtomicInteger();
    doAnswer(
            invocation -> {
              checkIdleCounter.incrementAndGet();
              verify(mockGrpcCommandServer, never()).shutdown();
              clock.advanceMillis(Duration.ofSeconds(5).toMillis());
              return null;
            })
        .when(mockCommands)
        .waitForChange(anyLong());

    thread.start();
    thread.join(TestUtils.WAIT_TIMEOUT_MILLISECONDS);

    verify(mockGrpcCommandServer).shutdown();
    assertThat(checkIdleCounter.get()).isEqualTo(2);
  }

  @Test
  public void testCompletedCommandDuringWaitResetsIdleTimeout() throws Exception {
    CommandManager commandManager =
        spy(new CommandManager(/* doIdleServerTasks= */ false, "slow interrupt message suffix"));
    ServerWatcherRunnable underTest =
        new ServerWatcherRunnable(
            mockGrpcCommandServer,
            /* maxIdleSeconds= */ 15,
            /* shutdownOnLowSysMem= */ false,
            commandManager);
    Thread thread = new Thread(underTest);
    List<Long> waitTimeouts = new ArrayList<>();
    AtomicInteger checkIdleCounter = new AtomicInteger();
    doAnswer(
            invocation -> {
              long timeoutMillis = invocation.getArgument(0);
              waitTimeouts.add(timeoutMillis);
              int step = checkIdleCounter.incrementAndGet();
              verify(mockGrpcCommandServer, never()).shutdown();
              if (step == 1 || step == 2) {
                clock.advanceMillis(Duration.ofSeconds(5).toMillis());
              } else if (step == 3) {
                // Simulate wait(5000) returning 10ms early (e.g. Windows timer quantization)
                // at t = 14,990ms.
                clock.advanceMillis(4990);
              } else if (step == 4) {
                // While waiting for the remaining 10ms, a client sends Ping() which starts and
                // finishes a RunningCommand before ServerWatcherRunnable wakes up at t = 15,005ms.
                try (RunningCommand unused = commandManager.createCommand()) {}
                clock.advanceMillis(15);
              } else {
                clock.advanceMillis(Duration.ofSeconds(5).toMillis());
              }
              return null;
            })
        .when(commandManager)
        .waitForChange(anyLong());

    thread.start();
    thread.join(TestUtils.WAIT_TIMEOUT_MILLISECONDS);

    verify(mockGrpcCommandServer).shutdown();
    // Steps 1-3 wait up to 14,990ms; step 4 waits for the remaining 10ms and receives a Ping() at
    // 15,005ms, which resets the 15s deadline to 30,005ms (requiring steps 5, 6, 7 of 5000ms each).
    assertThat(waitTimeouts)
        .containsExactly(5000L, 5000L, 5000L, 10L, 5000L, 5000L, 5000L)
        .inOrder();
    assertThat(clock.currentTimeMillis()).isEqualTo(30005L);
  }

  @Test
  public void testCommandClosingBetweenIsEmptyAndWaitForChangeDoesNotHang() throws Exception {
    CommandManager commandManager =
        spy(new CommandManager(/* doIdleServerTasks= */ false, "slow interrupt message suffix"));
    ServerWatcherRunnable underTest =
        new ServerWatcherRunnable(
            mockGrpcCommandServer,
            /* maxIdleSeconds= */ 10,
            /* shutdownOnLowSysMem= */ false,
            commandManager);
    Thread thread = new Thread(underTest);
    AtomicInteger checkIdleCounter = new AtomicInteger();
    final RunningCommand[] inflightCommand = new RunningCommand[1];
    doAnswer(
            invocation -> {
              int step = checkIdleCounter.incrementAndGet();
              verify(mockGrpcCommandServer, never()).shutdown();
              if (step == 1) {
                clock.advanceMillis(Duration.ofSeconds(5).toMillis());
                // A short command starts right as waitForChange(5000) returns so isEmpty() will
                // observe false (busy).
                inflightCommand[0] = commandManager.createCommand();
              } else {
                clock.advanceMillis(Duration.ofSeconds(5).toMillis());
              }
              return null;
            })
        .when(commandManager)
        .waitForChange(anyLong());
    doAnswer(
            invocation -> {
              // The command finishes right before ServerWatcherRunnable enters the unbounded
              // waitForChange() call on the !idle branch.
              inflightCommand[0].close();
              clock.advanceMillis(100);
              return invocation.callRealMethod();
            })
        .when(commandManager)
        .waitForChange();

    thread.start();
    thread.join(TestUtils.WAIT_TIMEOUT_MILLISECONDS);

    verify(mockGrpcCommandServer).shutdown();
    // 5000ms before command + 100ms command duration + 10,000ms new idle timeout = 15,100ms.
    assertThat(clock.currentTimeMillis()).isEqualTo(15100L);
  }

  @Test
  public void runLowAbsoluteHighPercentageMemoryCheck() throws Exception {
    if (!usingLinux()) {
      return;
    }
    assertThat(doesIdleLowMemoryCheckShutdown(/*freeRamKb=*/ 5000, /*totalRamKb=*/ 10000))
        .isFalse();
  }

  @Test
  public void runHighAbsoluteLowPercentageMemoryCheck() throws Exception {
    if (!usingLinux()) {
      return;
    }
    assertThat(doesIdleLowMemoryCheckShutdown(/*freeRamKb=*/ 1L << 21, /*totalRamKb=*/ 1L << 30))
        .isFalse();
  }

  @Test
  public void runLowAbsoluteLowPercentageMemoryCheck() throws Exception {
    if (!usingLinux()) {
      return;
    }
    assertThat(doesIdleLowMemoryCheckShutdown(/*freeRamKb=*/ 5000, /*totalRamKb=*/ 1000000))
        .isTrue();
  }

  @Test
  public void testshutdownOnLowSysMemDisabled() throws Exception {
    if (!usingLinux()) {
      return;
    }
    assertThat(
            doesIdleLowMemoryCheckShutdown(
                /*freeRamKb=*/ 5000, /*totalRamKb=*/ 1000000, /*shutdownOnLowSysMem=*/ false))
        .isFalse();
  }

  @Test
  public void testCommandDuringLowMemoryCheckSkipsShutdownAndResetsIdleTimeout() throws Exception {
    CommandManager commandManager =
        spy(new CommandManager(/* doIdleServerTasks= */ false, "slow interrupt message suffix"));
    AtomicInteger checkCounter = new AtomicInteger();
    ServerWatcherRunnable.LowMemoryChecker lowMemoryChecker =
        new ServerWatcherRunnable.LowMemoryChecker() {
          @Override
          boolean check() {
            if (checkCounter.incrementAndGet() == 1) {
              // Simulate a fast command starting and finishing while shouldShutdown() is checking
              // memory pressure, advancing the clock past the original 310s maxIdleSeconds
              // deadline.
              try (RunningCommand unused = commandManager.createCommand()) {}
              clock.advanceMillis(Duration.ofSeconds(20).toMillis());
            }
            return true;
          }
        };
    ServerWatcherRunnable underTest =
        new ServerWatcherRunnable(
            mockGrpcCommandServer,
            /* maxIdleSeconds= */ Duration.ofMinutes(5).plusSeconds(10).toSeconds(),
            /* shutdownOnLowSysMem= */ true,
            commandManager,
            lowMemoryChecker);
    Thread thread = new Thread(underTest);
    AtomicInteger serverWatcherLoopCounter = new AtomicInteger();
    doAnswer(
            invocation -> {
              int step = serverWatcherLoopCounter.incrementAndGet();
              if (step == 6) {
                // Delegate to the real waitForChange(timeout) right after the skipped low-memory
                // check to verify that lastObservedChangeCounter was not clobbered by isEmpty()
                // and that waitForChange returns immediately.
                return invocation.callRealMethod();
              }
              clock.advanceMillis(Duration.ofMinutes(1).toMillis());
              return null;
            })
        .when(commandManager)
        .waitForChange(anyLong());

    thread.start();
    thread.join(TestUtils.WAIT_TIMEOUT_MILLISECONDS);

    verify(mockGrpcCommandServer).shutdown();
    assertThat(checkCounter.get()).isEqualTo(2);
    // 5 minutes until first memory check (t=300s, skipped due to concurrent command that advances
    // clock to t=320s > initial 310s deadline), immediate return from real waitForChange(timeout)
    // on step 6, reset of both shutdownTimeMillis (to 630s) and lowMemoryChecker (to 620s) at
    // t=320s, plus 5 more minutes of idleness until second memory check (t=620s) = 11 wait steps.
    assertThat(serverWatcherLoopCounter.get()).isEqualTo(11);
    assertThat(clock.currentTimeMillis())
        .isEqualTo(Duration.ofMinutes(10).plusSeconds(20).toMillis());
  }

  private boolean doesIdleLowMemoryCheckShutdown(long freeRamKb, long totalRamKb) throws Exception {
    return doesIdleLowMemoryCheckShutdown(freeRamKb, totalRamKb, /*shutdownOnLowSysMem=*/ true);
  }

  private boolean doesIdleLowMemoryCheckShutdown(
      long freeRamKb, long totalRamKb, boolean shutdownOnLowSysMem) throws Exception {
    CommandManager mockCommandManager = mock(CommandManager.class);
    ProcMeminfoParser mockParser = mock(ProcMeminfoParser.class);
    ServerWatcherRunnable underTest =
        new ServerWatcherRunnable(
            mockGrpcCommandServer,
            // Shut down after an hour if we see no memory issues.
            /* maxIdleSeconds= */ Duration.ofHours(1).toSeconds(),
            shutdownOnLowSysMem,
            mockCommandManager,
            new ProcMeminfoLowMemoryChecker(() -> mockParser));
    Thread thread = new Thread(underTest);
    when(mockCommandManager.isEmpty()).thenReturn(true);
    AtomicInteger serverWatcherLoopCounter = new AtomicInteger();

    when(mockParser.getFreeRamKb()).thenReturn(freeRamKb);
    when(mockParser.getTotalKb()).thenReturn(totalRamKb);
    doAnswer(
            invocation -> {
              serverWatcherLoopCounter.incrementAndGet();
              clock.advanceMillis(Duration.ofMinutes(1).toMillis());
              return null;
            })
        .when(mockCommandManager)
        .waitForChange(Duration.ofSeconds(5).toMillis());

    thread.start();
    thread.join(TestUtils.WAIT_TIMEOUT_MILLISECONDS);
    verify(mockGrpcCommandServer).shutdown();

    // If we shut down due to memory pressure, it will only be after 5 minutes of being idle.
    return serverWatcherLoopCounter.get() == 5;
  }

  private boolean usingLinux() {
    return OS.getCurrent() == OS.LINUX;
  }
}
