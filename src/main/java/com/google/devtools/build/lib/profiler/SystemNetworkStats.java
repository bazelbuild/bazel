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

import com.google.auto.value.AutoValue;
import com.google.devtools.build.lib.jni.JniLoader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** Utility class for query system network stats. */
public class SystemNetworkStats {
  static {
    JniLoader.loadJni();
  }

  private SystemNetworkStats() {}

  /** Value class for network IO counters. */
  @AutoValue
  public abstract static class NetIoCounter {
    public static NetIoCounter create(
        long bytesSent, long bytesRecv, long packetsSent, long packetsRecv) {
      return new AutoValue_SystemNetworkStats_NetIoCounter(
          bytesSent, bytesRecv, packetsSent, packetsRecv);
    }

    // Number of bytes sent.
    public abstract long bytesSent();
    // Number of bytes received.
    public abstract long bytesRecv();
    // Number of packets sent.
    public abstract long packetsSent();
    // Number of packets received.
    public abstract long packetsRecv();
  }

  public static Map<String, NetIoCounter> getNetIoCounters() throws IOException {
    Map<String, NetIoCounter> counters = new HashMap<>();
    getNetIoCountersNative(counters);
    return counters;
  }

  /** Value class for network interface address. */
  @AutoValue
  public abstract static class NetIfAddr {
    /** Address family for the address. */
    public enum Family {
      AF_INET,
      AF_INET6,
      UNKNOWN,
    }

    public static NetIfAddr create(String name, Family family, String ipAddr) {
      return new AutoValue_SystemNetworkStats_NetIfAddr(name, family, ipAddr);
    }

    // Name of the interface.
    public abstract String name();

    public abstract Family family();

    // IP address for this interface if the family is AF_INET or AF_INET6.
    public abstract String ipAddr();
  }

  public static Map<String, List<NetIfAddr>> getNetIfAddrs() throws IOException {
    List<NetIfAddr> addrs = new ArrayList<>();
    getNetIfAddrsNative(addrs);

    Map<String, List<NetIfAddr>> result = new HashMap<>();
    for (NetIfAddr addr : addrs) {
      result.computeIfAbsent(addr.name(), ignored -> new ArrayList<>()).add(addr);
    }
    return result;
  }

  private static native void getNetIoCountersNative(Map<String, NetIoCounter> countersOut)
      throws IOException;

  private static native void getNetIfAddrsNative(List<NetIfAddr> addrsOut) throws IOException;
}
