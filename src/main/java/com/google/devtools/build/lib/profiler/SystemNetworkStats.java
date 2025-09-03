// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.profiler;

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.base.Splitter;
import com.google.devtools.build.lib.jni.JniLoader;
import com.google.devtools.build.lib.util.OS;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** Utility class for query system network stats. */
public class SystemNetworkStats {
  private static final Splitter SPLITTER = Splitter.on(" ").omitEmptyStrings().trimResults();

  static {
    JniLoader.loadJni();
  }

  private SystemNetworkStats() {}

  /**
   * Value class for network IO counters.
   *
   * @param bytesSent Number of bytes sent.
   * @param bytesRecv Number of bytes received.
   * @param packetsSent Number of packets sent.
   * @param packetsRecv Number of packets received.
   */
  public record NetIoCounter(long bytesSent, long bytesRecv, long packetsSent, long packetsRecv) {
    public static NetIoCounter create(
        long bytesSent, long bytesRecv, long packetsSent, long packetsRecv) {
      return new NetIoCounter(bytesSent, bytesRecv, packetsSent, packetsRecv);
    }
  }

  /** Returns a map from network interface name to the respective I/O counters. */
  public static Map<String, NetIoCounter> getNetIoCounters() throws IOException {
    HashMap<String, NetIoCounter> countersMap = new HashMap<>();
    switch (OS.getCurrent()) {
      case OS.LINUX -> getNetIoCountersLinux(countersMap);
      default -> {
        if (JniLoader.isJniAvailable()) {
          getNetIoCountersNative(countersMap);
        }
      }
    }
    return countersMap;
  }

  private static void getNetIoCountersLinux(Map<String, NetIoCounter> countersMap)
      throws IOException {
    List<String> lines = Files.readAllLines(Paths.get("/proc/net/dev"), UTF_8);

    // skip table header (first 2 lines)
    for (String line : lines.subList(2, lines.size())) {
      int colonAt = line.indexOf(':');
      if (colonAt < 0) {
        continue;
      }
      String name = line.substring(0, colonAt).strip();
      long[] fields =
          SPLITTER.splitToStream(line.substring(colonAt + 1)).mapToLong(Long::parseLong).toArray();
      if (fields.length > 9) {
        long bytesRecv = fields[0];
        long packetsRecv = fields[1];
        long bytesSent = fields[8];
        long packetsSent = fields[9];
        countersMap.put(name, NetIoCounter.create(bytesSent, bytesRecv, packetsSent, packetsRecv));
      }
    }
  }

  private static native void getNetIoCountersNative(Map<String, NetIoCounter> countersMap)
      throws IOException;
}
