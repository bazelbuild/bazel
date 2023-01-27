// Copyright 2023 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.util;

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.auto.value.AutoValue;
import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableMap;
import com.google.common.flogger.GoogleLogger;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.List;

/**
 * Helps to collect infomation about all process using ps command. Works for Linux and MacOS systems
 */
public final class PsInfoCollector {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private PsInfoCollector() {
    // prevent construction.
  }

  /** Collects memory usage for every process. */
  public static ImmutableMap<Long, PsInfo> collectDataFromPs() {
    try {
      Process psProcess = buildPsProcess();
      return collectDataFromPsProcess(psProcess);
    } catch (IOException e) {
      logger.atWarning().withCause(e).log("Error while executing command ps");
      return ImmutableMap.of();
    }
  }

  static ImmutableMap<Long, PsInfo> collectDataFromPsProcess(Process psProcess) {
    BufferedReader psOutput =
        new BufferedReader(new InputStreamReader(psProcess.getInputStream(), UTF_8));

    ImmutableMap.Builder<Long, PsInfo> psInfos = ImmutableMap.builder();

    try {
      // The output of the above ps command looks similar to this:
      // PID     PPID   RSS
      // 211706  1      222972
      // 2612333 211706 6180
      // We skip over the first line (the header) and then parse the PID and the resident memory
      // size in kilobytes.
      String output = null;
      boolean isFirst = true;
      while ((output = psOutput.readLine()) != null) {
        if (isFirst) {
          isFirst = false;
          continue;
        }
        List<String> line = Splitter.on(" ").trimResults().omitEmptyStrings().splitToList(output);
        if (line.size() != 3) {
          logger.atWarning().log("Unexpected length of split line %s %d", output, line.size());
          continue;
        }

        long pid = Long.parseLong(line.get(0));
        long parentPid = Long.parseLong(line.get(1));
        int memoryInKb = Integer.parseInt(line.get(2));

        psInfos.put(pid, PsInfo.create(pid, parentPid, memoryInKb));
      }
    } catch (IllegalArgumentException | IOException e) {
      logger.atWarning().withCause(e).log("Error while parsing psOutput: %s", psOutput);
    }

    return psInfos.buildOrThrow();
  }

  private static Process buildPsProcess() throws IOException {
    return new ProcessBuilder("ps", "-e", "-o", "pid,ppid,rss").start();
  }

  /** Parsed information about process collected after ps command call. */
  @AutoValue
  public abstract static class PsInfo {
    public abstract long getPid();

    public abstract long getParentPid();

    public abstract int getMemoryInKb();

    public static PsInfo create(long pid, long parentPid, int memoryinKb) {
      return new AutoValue_PsInfoCollector_PsInfo(pid, parentPid, memoryinKb);
    }
  }
}
