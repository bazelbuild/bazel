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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableSet;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildMetrics.NetworkMetrics;
import com.google.devtools.build.lib.jni.JniLoader;
import com.google.devtools.build.lib.util.OS;
import java.io.IOException;
import java.net.NetworkInterface;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/** Collects and populates network metrics during an invocation. */
public final class NetworkMetricsCollector {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();
  private static final Splitter SPLITTER = Splitter.on(' ').omitEmptyStrings().trimResults();

  static {
    JniLoader.loadJni();
  }

  /** The metrics collector (a static singleton instance). Inactive by default. */
  private static final NetworkMetricsCollector instance = new NetworkMetricsCollector();

  @Nullable private ImmutableSet<String> loopbackInterfaceNames = null;
  @Nullable private Map<String, NetIoCounter> previousNetworkIoCounters = null;
  private final NetworkMetrics.SystemNetworkStats.Builder systemNetworkStats =
      NetworkMetrics.SystemNetworkStats.newBuilder();

  private NetworkMetricsCollector() {}

  public static NetworkMetricsCollector instance() {
    return instance;
  }

  @Nullable
  public NetworkMetrics collectMetrics() {
    if (systemNetworkStats.getBytesRecv() == 0 || systemNetworkStats.getBytesSent() == 0) {
      return null;
    }

    return NetworkMetrics.newBuilder().setSystemNetworkStats(systemNetworkStats.build()).build();
  }

  @SuppressWarnings("GoodTime") // performance-sensitive
  @Nullable
  public SystemNetworkUsages collectSystemNetworkUsages(double deltaNanos) {
    if (loopbackInterfaceNames == null) {
      try {
        loopbackInterfaceNames = getLoopbackInterfaceNames();
      } catch (IOException e) {
        logger.atWarning().withCause(e).log("Failed to get loopback interface names");
      }
    }

    Map<String, NetIoCounter> nextNetworkIoCounters = null;
    try {
      nextNetworkIoCounters = getNetIoCounters();
    } catch (IOException e) {
      logger.atWarning().withCause(e).log("Failed to get Net IO counters");
    }

    if (previousNetworkIoCounters == null) {
      previousNetworkIoCounters = nextNetworkIoCounters;
    }

    SystemNetworkUsages usages = null;
    if (previousNetworkIoCounters != null && nextNetworkIoCounters != null) {
      long deltaBytesSent = 0;
      long deltaBytesRecv = 0;
      long deltaPacketsSent = 0;
      long deltaPacketsRecv = 0;
      for (Map.Entry<String, NetIoCounter> entry : previousNetworkIoCounters.entrySet()) {
        String name = entry.getKey();
        if (loopbackInterfaceNames.contains(name)) {
          continue;
        }
        NetIoCounter previous = entry.getValue();
        NetIoCounter next = nextNetworkIoCounters.get(name);
        if (next != null) {
          deltaBytesSent += calcDelta(previous.bytesSent(), next.bytesSent());
          deltaBytesRecv += calcDelta(previous.bytesRecv(), next.bytesRecv());
          deltaPacketsSent += calcDelta(previous.packetsSent(), next.packetsSent());
          deltaPacketsRecv += calcDelta(previous.packetsRecv(), next.packetsRecv());
        }
      }

      systemNetworkStats.setBytesRecv(systemNetworkStats.getBytesRecv() + deltaBytesRecv);
      systemNetworkStats.setBytesSent(systemNetworkStats.getBytesSent() + deltaBytesSent);
      systemNetworkStats.setPacketsRecv(systemNetworkStats.getPacketsRecv() + deltaPacketsRecv);
      systemNetworkStats.setPacketsSent(systemNetworkStats.getPacketsSent() + deltaPacketsSent);

      usages =
          SystemNetworkUsages.create(
              calcValuePerSec(deltaBytesSent, deltaNanos),
              calcValuePerSec(deltaBytesRecv, deltaNanos),
              calcValuePerSec(deltaPacketsSent, deltaNanos),
              calcValuePerSec(deltaPacketsRecv, deltaNanos));

      if (usages.bytesSentPerSec() > systemNetworkStats.getPeakBytesSentPerSec()) {
        systemNetworkStats.setPeakBytesSentPerSec((long) usages.bytesSentPerSec());
      }
      if (usages.bytesRecvPerSec() > systemNetworkStats.getPeakBytesRecvPerSec()) {
        systemNetworkStats.setPeakBytesRecvPerSec((long) usages.bytesRecvPerSec());
      }
      if (usages.packetsSentPerSec() > systemNetworkStats.getPeakPacketsSentPerSec()) {
        systemNetworkStats.setPeakPacketsSentPerSec((long) usages.packetsSentPerSec());
      }
      if (usages.packetsRecvPerSec() > systemNetworkStats.getPeakPacketsRecvPerSec()) {
        systemNetworkStats.setPeakPacketsRecvPerSec((long) usages.packetsRecvPerSec());
      }
    }

    previousNetworkIoCounters = nextNetworkIoCounters;

    return usages;
  }

  /** Aggregated system network usages over all interfaces except local loopback. */
  public record SystemNetworkUsages(
      double bytesSentPerSec,
      double bytesRecvPerSec,
      double packetsSentPerSec,
      double packetsRecvPerSec) {
    public static SystemNetworkUsages create(
        double bytesSentPerSec,
        double bytesRecvPerSec,
        double packetsSentPerSec,
        double packetsRecvPerSec) {
      return new SystemNetworkUsages(
          bytesSentPerSec, bytesRecvPerSec, packetsSentPerSec, packetsRecvPerSec);
    }

    public double megabitsSentPerSec() {
      return bytesPerSecToMegabitsPerSec(bytesSentPerSec());
    }

    public double megabitsRecvPerSec() {
      return bytesPerSecToMegabitsPerSec(bytesRecvPerSec());
    }
  }

  /** Value class for network IO counters. */
  @VisibleForTesting
  record NetIoCounter(long bytesSent, long bytesRecv, long packetsSent, long packetsRecv) {}

  /** Returns a map from network interface name to the respective I/O counters. */
  @VisibleForTesting
  static Map<String, NetIoCounter> getNetIoCounters() throws IOException {
    HashMap<String, NetIoCounter> countersMap = new HashMap<>();
    switch (OS.getCurrent()) {
      case LINUX -> getNetIoCountersLinux(countersMap);
      case DARWIN -> getNetIoCountersNative(countersMap);
      default -> {}
    }
    return countersMap;
  }

  private static void getNetIoCountersLinux(Map<String, NetIoCounter> countersMap)
      throws IOException {
    List<String> lines = Files.readAllLines(Path.of("/proc/net/dev"), UTF_8);

    // skip table header (first 2 lines)
    for (String line : lines.subList(2, lines.size())) {
      int colonAt = line.indexOf(':');
      if (colonAt < 0) {
        continue;
      }
      String name = line.substring(0, colonAt).strip();
      long[] fields =
          SPLITTER
              .splitToStream(line.substring(colonAt + 1))
              .mapToLong(Long::parseUnsignedLong)
              .toArray();
      if (fields.length > 9) {
        long bytesRecv = fields[0];
        long packetsRecv = fields[1];
        long bytesSent = fields[8];
        long packetsSent = fields[9];
        countersMap.put(name, new NetIoCounter(bytesSent, bytesRecv, packetsSent, packetsRecv));
      }
    }
  }

  private static native void getNetIoCountersNative(Map<String, NetIoCounter> countersMap)
      throws IOException;

  private static ImmutableSet<String> getLoopbackInterfaceNames() throws IOException {
    ImmutableSet.Builder<String> result = ImmutableSet.builder();
    Enumeration<NetworkInterface> ifaces = NetworkInterface.getNetworkInterfaces();
    if (ifaces != null) {
      for (NetworkInterface iface : Collections.list(ifaces)) {
        if (iface.isLoopback()) {
          result.add(iface.getName());
        }
      }
    }
    return result.build();
  }

  private static long calcDelta(long prev, long next) {
    if (Long.compareUnsigned(next, prev) < 0) {
      return next;
    }
    return next - prev;
  }

  private static double calcValuePerSec(long deltaValue, double deltaNanos) {
    return deltaValue / deltaNanos * 1e9;
  }

  private static double bytesPerSecToMegabitsPerSec(double bytesPerSec) {
    return bytesPerSec / 1e6 * 8;
  }
}
