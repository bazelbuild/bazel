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
import static org.mockito.Matchers.anyLong;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.testutil.ManualClock;
import com.google.devtools.build.lib.testutil.TestUtils;
import io.grpc.Server;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ServerWatcherRunnable}. */
@RunWith(JUnit4.class)
public class ServerWatcherRunnableTest {
  private ManualClock clock;
  private Server mockServer;

  @Before
  public final void setManualClock() {
    clock = new ManualClock();
    mockServer = mock(Server.class);
    BlazeClock.setClock(clock);
  }

  @Test
  public void testBasicIdleCheck() throws Exception {
    CommandManager mockCommands = mock(CommandManager.class);
    ServerWatcherRunnable underTest =
        new ServerWatcherRunnable(mockServer, /*maxIdleSeconds=*/ 10, mockCommands);
    Thread thread = new Thread(underTest);
    when(mockCommands.isEmpty()).thenReturn(true);
    AtomicInteger checkIdleCounter = new AtomicInteger();
    doAnswer(
            invocation -> {
              checkIdleCounter.incrementAndGet();
              verify(mockServer, never()).shutdown();
              clock.advanceMillis(5 * 1000L);
              return null;
            })
        .when(mockCommands)
        .waitForChange(anyLong());

    thread.start();
    thread.join(TestUtils.WAIT_TIMEOUT_MILLISECONDS);

    verify(mockServer).shutdown();
    assertThat(checkIdleCounter.get()).isEqualTo(2);
  }
}
