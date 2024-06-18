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

import com.google.devtools.build.lib.vfs.Path;
import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.time.Duration;
import java.util.Optional;

/** Provides execution statistics (e.g. resource usage) for external commands. */
public final class ExecutionStatistics {
  /**
   * Provides execution statistics based on a {@code execution_statistics.proto} file.
   *
   * @param executionStatisticsProtoPath path to a materialized ExecutionStatistics proto
   * @return a {@link ResourceUsage} object containing execution statistics, if available
   */
  public static Optional<ResourceUsage> getResourceUsage(Path executionStatisticsProtoPath)
      throws IOException {
    if (!executionStatisticsProtoPath.exists()) {
      // Collecting resource usage is best-effort and the file may be missing if the wrapper around
      // the command terminated abnormally.
      return Optional.empty();
    }
    try (InputStream protoInputStream =
        new BufferedInputStream(executionStatisticsProtoPath.getInputStream())) {
      Protos.ExecutionStatistics executionStatisticsProto =
          Protos.ExecutionStatistics.parseFrom(protoInputStream);
      if (executionStatisticsProto.hasResourceUsage()) {
        return Optional.of(new ResourceUsage(executionStatisticsProto.getResourceUsage()));
      } else {
        return Optional.empty();
      }
    }
  }

  /**
   * Provides resource usage statistics for command execution, derived from the getrusage() system
   * call.
   */
  public static class ResourceUsage {
    private final Protos.ResourceUsage resourceUsageProto;

    /** Provides resource usage statistics via a ResourceUsage proto object. */
    public ResourceUsage(Protos.ResourceUsage resourceUsageProto) {
      this.resourceUsageProto = resourceUsageProto;
    }

    /** Returns the user time for command execution, if available. */
    public Duration getUserExecutionTime() {
      return Duration.ofSeconds(
          resourceUsageProto.getUtimeSec(), resourceUsageProto.getUtimeUsec() * 1000);
    }

    /** Returns the system time for command execution, if available. */
    public Duration getSystemExecutionTime() {
      return Duration.ofSeconds(
          resourceUsageProto.getStimeSec(), resourceUsageProto.getStimeUsec() * 1000);
    }

    /** Returns the maximum resident set size (in bytes) during command execution, if available. */
    public long getMaximumResidentSetSize() {
      return resourceUsageProto.getMaxrss();
    }

    /**
     * Returns the integral shared memory size (in bytes) during command execution, if available.
     */
    public long getIntegralSharedMemorySize() {
      return resourceUsageProto.getIxrss();
    }

    /**
     * Returns the integral unshared data size (in bytes) during command execution, if available.
     */
    public long getIntegralUnsharedDataSize() {
      return resourceUsageProto.getIdrss();
    }

    /**
     * Returns the integral unshared stack size (in bytes) during command execution, if available.
     */
    public long getIntegralUnsharedStackSize() {
      return resourceUsageProto.getIsrss();
    }

    /**
     * Returns the number of page reclaims (soft page faults) during command execution, if
     * available.
     */
    public long getPageReclaims() {
      return resourceUsageProto.getMinflt();
    }

    /** Returns the number of (hard) page faults during command execution, if available. */
    public long getPageFaults() {
      return resourceUsageProto.getMajflt();
    }

    /** Returns the number of swaps during command execution, if available. */
    public long getSwaps() {
      return resourceUsageProto.getNswap();
    }

    /** Returns the number of block input operations during command execution, if available. */
    public long getBlockInputOperations() {
      return resourceUsageProto.getInblock();
    }

    /** Returns the number of block output operations during command execution, if available. */
    public long getBlockOutputOperations() {
      return resourceUsageProto.getOublock();
    }

    /** Returns the number of IPC messages sent during command execution, if available. */
    public long getIpcMessagesSent() {
      return resourceUsageProto.getMsgsnd();
    }

    /** Returns the number of IPC messages received during command execution, if available. */
    public long getIpcMessagesReceived() {
      return resourceUsageProto.getMsgrcv();
    }

    /** Returns the number of signals received during command execution, if available. */
    public long getSignalsReceived() {
      return resourceUsageProto.getNsignals();
    }

    /** Returns the number of voluntary context switches during command execution, if available. */
    public long getVoluntaryContextSwitches() {
      return resourceUsageProto.getNvcsw();
    }

    /**
     * Returns the number of involuntary context switches during command execution, if available.
     */
    public long getInvoluntaryContextSwitches() {
      return resourceUsageProto.getNivcsw();
    }
  }
}
