// Copyright 2026 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.shell;

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import java.util.concurrent.Callable;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class JavaSubprocessFactoryTest {

  @Test
  public void createFromVirtualThreadForksOnPlatformThread() throws Exception {
    ExecutorService mockExecutorService = mock(ExecutorService.class);
    AtomicReference<Thread> forkThread = new AtomicReference<>();

    // Observe the thread executing the submit function, so we can tell if the fork happens from
    // the correct thread. We still call the real submit function to actually run the callable.
    doAnswer(
            invocation -> {
              Callable<?> original = invocation.getArgument(0);
              return JavaSubprocessFactory.DEFAULT_FORK_EXECUTOR.submit(
                  () -> {
                    forkThread.set(Thread.currentThread());
                    return original.call();
                  });
            })
        .when(mockExecutorService)
        .submit(any(Callable.class));

    JavaSubprocessFactory subprocessFactory =
        JavaSubprocessFactory.createForTesting(mockExecutorService);

    AtomicReference<Boolean> callerWasVirtual = new AtomicReference<>();
    AtomicReference<Integer> exitCode = new AtomicReference<>();
    AtomicReference<Throwable> error = new AtomicReference<>();
    CountDownLatch done = new CountDownLatch(1);

    // Create and execute a subprocess on a virtual thread
    Thread.ofVirtual()
        .name("test-virtual")
        .start(
            () -> {
              try {
                SubprocessBuilder subprocessBuilder =
                    new SubprocessBuilder(ImmutableMap.of(), subprocessFactory);
                subprocessBuilder.setArgv(ImmutableList.of("/bin/true"));

                Subprocess subprocess = subprocessBuilder.start();
                callerWasVirtual.set(Thread.currentThread().isVirtual());
                subprocess.waitFor();
                exitCode.set(subprocess.exitValue());
                subprocess.close();
              } catch (Exception e) {
                error.set(e);
              } finally {
                done.countDown();
              }
            });

    done.await();
    if (error.get() != null) {
      throw new AssertionError("Error running subprocess from virtual thread", error.get());
    }

    // The subprocess creation happened successfully on a virtual thread
    assertThat(callerWasVirtual.get()).isTrue();
    assertThat(exitCode.get()).isEqualTo(0);

    // The fork was routed through the correct executor and ran on the right thread
    verify(mockExecutorService).submit(any(Callable.class));
    assertThat(forkThread.get().getName()).isEqualTo("subprocess-fork");
    assertThat(forkThread.get().isVirtual()).isFalse();
    assertThat(forkThread.get().isDaemon()).isTrue();
  }
}
