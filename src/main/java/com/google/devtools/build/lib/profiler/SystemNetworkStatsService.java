// Copyright 2025 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.runtime.BlazeService;
import java.io.IOException;
import java.util.Map;

/** Service for querying system network stats. */
public interface SystemNetworkStatsService extends BlazeService {

  /** Returns a map from network interface name to the respective I/O counters. */
  Map<String, NetIoCounter> getNetIoCounters() throws IOException;

  /**
   * Value class for network IO counters.
   *
   * @param bytesSent Number of bytes sent.
   * @param bytesRecv Number of bytes received.
   * @param packetsSent Number of packets sent.
   * @param packetsRecv Number of packets received.
   */
  record NetIoCounter(long bytesSent, long bytesRecv, long packetsSent, long packetsRecv) {
    public static NetIoCounter create(
        long bytesSent, long bytesRecv, long packetsSent, long packetsRecv) {
      return new NetIoCounter(bytesSent, bytesRecv, packetsSent, packetsRecv);
    }
  }
}
