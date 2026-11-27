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

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.collect.MoreCollectors.onlyElement;
import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.util.OS;
import java.nio.file.Path;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import jdk.jfr.Recording;
import jdk.jfr.consumer.RecordedEvent;
import jdk.jfr.consumer.RecordedThread;
import jdk.jfr.consumer.RecordingFile;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class JavaSubprocessFactoryTest {

  @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();

  @Test
  public void createFromVirtualThreadForksOnPlatformThread() throws Exception {
    Path recordingPath = tempFolder.newFile("process-start.jfr").toPath();
    long pid;
    try (Recording recording = new Recording();
        ExecutorService virtualExecutor = Executors.newVirtualThreadPerTaskExecutor()) {
      recording.enable("jdk.ProcessStart");
      recording.start();

      pid =
          virtualExecutor
              .submit(() -> {
                assertThat(Thread.currentThread().isVirtual()).isTrue();
                SubprocessBuilder subprocessBuilder =
                    new SubprocessBuilder(ImmutableMap.of(), JavaSubprocessFactory.INSTANCE);
                subprocessBuilder.setArgv(
                    ImmutableList.of(OS.getCurrent() == OS.DARWIN ? "/usr/bin/true" : "/bin/true"));
                try (Subprocess subprocess = subprocessBuilder.start()) {
                  subprocess.waitFor();
                  assertThat(subprocess.exitValue()).isEqualTo(0);
                  return subprocess.getProcessId();
                }
              })
              .get();

      recording.stop();
      recording.dump(recordingPath);
    }

    ImmutableList<RecordedEvent> processStartEvents =
        RecordingFile.readAllEvents(recordingPath).stream()
            .filter(e -> e.getEventType().getName().equals("jdk.ProcessStart"))
            .filter(e -> e.getLong("pid") == pid)
            .collect(toImmutableList());
    assertThat(processStartEvents).hasSize(1);

    RecordedThread recordedForkThread = processStartEvents.get(0).getThread();
    assertThat(recordedForkThread).isNotNull();
    assertThat(recordedForkThread.getJavaName()).isEqualTo("subprocess-fork");
    assertThat(recordedForkThread.isVirtual()).isFalse();

    Thread forkThread =
        Thread.getAllStackTraces().keySet().stream()
            .filter(t -> t.threadId() == recordedForkThread.getJavaThreadId())
            .collect(onlyElement());
    assertThat(forkThread.isDaemon()).isTrue();
  }
}
