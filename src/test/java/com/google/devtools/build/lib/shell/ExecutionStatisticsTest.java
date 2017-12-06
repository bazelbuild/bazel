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

import static com.google.common.truth.Truth8.assertThat;

import com.google.devtools.build.lib.shell.Protos.ResourceUsage;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.time.Duration;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ExecutionStatistics}. */
@RunWith(JUnit4.class)
public final class ExecutionStatisticsTest {

  private com.google.devtools.build.lib.shell.Protos.ExecutionStatistics
      makeExecutionStatisticsProto(Duration userExecutionTime, Duration systemExecutionTime) {
    ResourceUsage resourceUsage =
        ResourceUsage.newBuilder()
            .setUtimeSec(userExecutionTime.getSeconds())
            .setUtimeUsec((long) (userExecutionTime.getNano() / 1000))
            .setStimeSec(systemExecutionTime.getSeconds())
            .setStimeUsec((long) (systemExecutionTime.getNano() / 1000))
            .build();

    return com.google.devtools.build.lib.shell.Protos.ExecutionStatistics.newBuilder()
        .setResourceUsage(resourceUsage)
        .build();
  }

  @Test
  public void testStatiticsProvided_fromProtoFilename() throws Exception {
    Duration riggedUserExecutionTime = Duration.ofSeconds(42).plusNanos(19790000);
    Duration riggedSystemExecutionTime = Duration.ofSeconds(33).plusNanos(290000);

    com.google.devtools.build.lib.shell.Protos.ExecutionStatistics executionStatisticsProto =
        makeExecutionStatisticsProto(riggedUserExecutionTime, riggedSystemExecutionTime);

    byte[] protoBytes = executionStatisticsProto.toByteArray();
    File encodedProtoFile = File.createTempFile("encoded_action_execution_proto", "");
    String protoFilename = encodedProtoFile.getPath();
    try (BufferedOutputStream bufferedOutputStream =
        new BufferedOutputStream(new FileOutputStream(encodedProtoFile))) {
      bufferedOutputStream.write(protoBytes);
    }

    ExecutionStatistics executionStatistics = new ExecutionStatistics(protoFilename);

    assertThat(executionStatistics.getUserExecutionTime()).hasValue(riggedUserExecutionTime);
    assertThat(executionStatistics.getSystemExecutionTime()).hasValue(riggedSystemExecutionTime);
  }
}
