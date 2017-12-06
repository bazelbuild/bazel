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

import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.time.Duration;
import java.util.Optional;

/** Provides execution statistics (e.g. resource usage) for external commands. */
public final class ExecutionStatistics {
  private final com.google.devtools.build.lib.shell.Protos.ExecutionStatistics
      executionStatisticsProto;

  /**
   * Provides execution statistics based on a {@code execution_statistics.proto} file.
   *
   * @param executionStatisticsProtoPath path to a materialized ExecutionStatistics proto
   */
  public ExecutionStatistics(String executionStatisticsProtoPath) throws IOException {
    try (InputStream protoInputStream =
        new BufferedInputStream(new FileInputStream(executionStatisticsProtoPath))) {
      executionStatisticsProto =
          com.google.devtools.build.lib.shell.Protos.ExecutionStatistics.parseFrom(
              protoInputStream);
    }
  }

  /** Returns the user time for command execution, if available. */
  public Optional<Duration> getUserExecutionTime() {
    if (executionStatisticsProto.hasResourceUsage()) {
      return Optional.of(
          Duration.ofSeconds(
              executionStatisticsProto.getResourceUsage().getUtimeSec(),
              executionStatisticsProto.getResourceUsage().getUtimeUsec() * 1000));
    } else {
      return Optional.empty();
    }
  }

  /** Returns the system time for command execution, if available. */
  public Optional<Duration> getSystemExecutionTime() {
    if (executionStatisticsProto.hasResourceUsage()) {
      return Optional.of(
          Duration.ofSeconds(
              executionStatisticsProto.getResourceUsage().getStimeSec(),
              executionStatisticsProto.getResourceUsage().getStimeUsec() * 1000));
    } else {
      return Optional.empty();
    }
  }
}
