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

import com.google.common.collect.ImmutableSet;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildMetrics.NetworkMetrics;
import com.google.devtools.build.lib.profiler.SystemNetworkStats.NetIoCounter;
import java.io.IOException;
import java.net.NetworkInterface;
import java.util.Enumeration;
import java.util.Map;
import javax.annotation.Nullable;

/** Collects and populates network metrics during an invocation. */
public final class NetworkMetricsCollector {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

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
      nextNetworkIoCounters = SystemNetworkStats.getNetIoCounters();
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

  private static ImmutableSet<String> getLoopbackInterfaceNames() throws IOException {
    ImmutableSet.Builder<String> result = ImmutableSet.builder();
    Enumeration<NetworkInterface> ifaces = NetworkInterface.getNetworkInterfaces();
    while (ifaces.hasMoreElements()) {
      NetworkInterface iface = ifaces.nextElement();
      if (iface.isLoopback()) {
        result.add(iface.getName());
      }
    }
    return result.build();
  }

  private static long calcDelta(long prev, long next) {
    // The next could wrap, and if that happens, assume prev is 0 (best effort).
    if (next < prev) {
      return next;
    } else {
      return next - prev;
    }
  }

  private static double calcValuePerSec(long deltaValue, double deltaNanos) {
    return deltaValue / deltaNanos * 1e9;
  }

  private static double bytesPerSecToMegabitsPerSec(double bytesPerSec) {
    return bytesPerSec / 1e6 * 8;
  }
}
