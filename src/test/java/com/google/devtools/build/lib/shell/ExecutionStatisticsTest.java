// Copyright 2017 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.BufferedOutputStream;
import java.time.Duration;
import java.util.Optional;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ExecutionStatistics}. */
@RunWith(JUnit4.class)
public final class ExecutionStatisticsTest {
  private Path workingDir;

  @Before
  public final void createFileSystem() throws Exception {
    FileSystem testFS = new InMemoryFileSystem(DigestHashFunction.SHA256);
    workingDir = TestUtils.createUniqueTmpDir(testFS);
  }

  private Path createExecutionStatisticsProtoFile(
      com.google.devtools.build.lib.shell.Protos.ExecutionStatistics executionStatisticsProto)
      throws Exception {
    Path encodedProtoFile = workingDir.getRelative("encoded_action_execution_proto");
    try (BufferedOutputStream bufferedOutputStream =
        new BufferedOutputStream(encodedProtoFile.getOutputStream())) {
      executionStatisticsProto.writeTo(bufferedOutputStream);
    }
    return encodedProtoFile;
  }

  @Test
  public void testNoResourceUsage_whenNoResourceUsageProto() throws Exception {
    com.google.devtools.build.lib.shell.Protos.ExecutionStatistics executionStatisticsProto =
        com.google.devtools.build.lib.shell.Protos.ExecutionStatistics.getDefaultInstance();
    Path protoFilename = createExecutionStatisticsProtoFile(executionStatisticsProto);

    Optional<ExecutionStatistics.ResourceUsage> resourceUsage =
        ExecutionStatistics.getResourceUsage(protoFilename);
    assertThat(resourceUsage).isEmpty();
  }

  @Test
  public void testStatiticsProvided_fromProtoFilename() throws Exception {
    Duration riggedUserExecutionTime = Duration.ofSeconds(42).plusNanos(19790000);
    Duration riggedSystemExecutionTime = Duration.ofSeconds(33).plusNanos(290000);
    long riggedMaximumResidentSetSize = 1;
    long riggedIntegralSharedMemorySize = 2;
    long riggedIntegralUnsharedDataSize = 3;
    long riggedIntegralUnsharedStackSize = 4;
    long riggedPageReclaims = 5;
    long riggedPageFaults = 6;
    long riggedSwaps = 7;
    long riggedBlockInputOperations = 8;
    long riggedBlockOutputOperations = 9;
    long riggedIpcMessagesSent = 10;
    long riggedIpcMessagesReceived = 11;
    long riggedSignalsReceived = 12;
    long riggedVoluntaryContextSwitches = 13;
    long riggedInvoluntaryContextSwitches = 14;

    com.google.devtools.build.lib.shell.Protos.ResourceUsage resourceUsageProto =
        com.google.devtools.build.lib.shell.Protos.ResourceUsage.newBuilder()
            .setUtimeSec(riggedUserExecutionTime.getSeconds())
            .setUtimeUsec((long) (riggedUserExecutionTime.getNano() / 1000))
            .setStimeSec(riggedSystemExecutionTime.getSeconds())
            .setStimeUsec((long) (riggedSystemExecutionTime.getNano() / 1000))
            .setMaxrss(riggedMaximumResidentSetSize)
            .setIxrss(riggedIntegralSharedMemorySize)
            .setIdrss(riggedIntegralUnsharedDataSize)
            .setIsrss(riggedIntegralUnsharedStackSize)
            .setMinflt(riggedPageReclaims)
            .setMajflt(riggedPageFaults)
            .setNswap(riggedSwaps)
            .setInblock(riggedBlockInputOperations)
            .setOublock(riggedBlockOutputOperations)
            .setMsgsnd(riggedIpcMessagesSent)
            .setMsgrcv(riggedIpcMessagesReceived)
            .setNsignals(riggedSignalsReceived)
            .setNvcsw(riggedVoluntaryContextSwitches)
            .setNivcsw(riggedInvoluntaryContextSwitches)
            .build();

    com.google.devtools.build.lib.shell.Protos.ExecutionStatistics executionStatisticsProto =
        com.google.devtools.build.lib.shell.Protos.ExecutionStatistics.newBuilder()
            .setResourceUsage(resourceUsageProto)
            .build();
    Path protoFilename = createExecutionStatisticsProtoFile(executionStatisticsProto);

    Optional<ExecutionStatistics.ResourceUsage> maybeResourceUsage =
        ExecutionStatistics.getResourceUsage(protoFilename);
    assertThat(maybeResourceUsage).isPresent();
    ExecutionStatistics.ResourceUsage resourceUsage = maybeResourceUsage.get();

    assertThat(resourceUsage.getUserExecutionTime()).isEqualTo(riggedUserExecutionTime);
    assertThat(resourceUsage.getSystemExecutionTime()).isEqualTo(riggedSystemExecutionTime);
    assertThat(resourceUsage.getMaximumResidentSetSize()).isEqualTo(riggedMaximumResidentSetSize);
    assertThat(resourceUsage.getIntegralSharedMemorySize())
        .isEqualTo(riggedIntegralSharedMemorySize);
    assertThat(resourceUsage.getIntegralUnsharedDataSize())
        .isEqualTo(riggedIntegralUnsharedDataSize);
    assertThat(resourceUsage.getIntegralUnsharedStackSize())
        .isEqualTo(riggedIntegralUnsharedStackSize);
    assertThat(resourceUsage.getPageReclaims()).isEqualTo(riggedPageReclaims);
    assertThat(resourceUsage.getPageFaults()).isEqualTo(riggedPageFaults);
    assertThat(resourceUsage.getSwaps()).isEqualTo(riggedSwaps);
    assertThat(resourceUsage.getBlockInputOperations()).isEqualTo(riggedBlockInputOperations);
    assertThat(resourceUsage.getBlockOutputOperations()).isEqualTo(riggedBlockOutputOperations);
    assertThat(resourceUsage.getIpcMessagesSent()).isEqualTo(riggedIpcMessagesSent);
    assertThat(resourceUsage.getIpcMessagesReceived()).isEqualTo(riggedIpcMessagesReceived);
    assertThat(resourceUsage.getSignalsReceived()).isEqualTo(riggedSignalsReceived);
    assertThat(resourceUsage.getVoluntaryContextSwitches())
        .isEqualTo(riggedVoluntaryContextSwitches);
    assertThat(resourceUsage.getInvoluntaryContextSwitches())
        .isEqualTo(riggedInvoluntaryContextSwitches);
  }
}
