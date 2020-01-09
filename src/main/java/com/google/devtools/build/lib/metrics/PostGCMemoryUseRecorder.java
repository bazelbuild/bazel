// Copyright 2020 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.metrics;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Supplier;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.ServerBuilder;
import com.google.devtools.build.lib.runtime.commands.InfoItem;
import com.google.devtools.build.lib.util.StringUtilities;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParsingResult;
import com.sun.management.GarbageCollectionNotificationInfo;
import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryUsage;
import java.util.Map;
import java.util.logging.Logger;
import javax.management.Notification;
import javax.management.NotificationEmitter;
import javax.management.NotificationListener;
import javax.management.openmbean.CompositeData;

/**
 * Keeps track of the peak heap usage directly after a full GC by listening for GC notifications.
 *
 * <p>The idea behind this class is as follows. We assume that:
 *
 * <pre>
 * sizeof(heap used) = sizeof(data) + sizeof(garbage)
 * </pre>
 *
 * and that after a full GC sizeof(garbage) is close to 0.
 *
 * <p>This allows us to measure sizeof(data) by measuring sizeof(heap used) immediately after a full
 * GC.
 */
public final class PostGCMemoryUseRecorder implements NotificationListener {
  private static PostGCMemoryUseRecorder instance = null;

  public static synchronized PostGCMemoryUseRecorder get() {
    if (instance == null) {
      instance = new PostGCMemoryUseRecorder(ManagementFactory.getGarbageCollectorMXBeans());
    }
    return instance;
  }

  private static final Logger logger = Logger.getLogger(PostGCMemoryUseRecorder.class.getName());

  // Protected by PostGCMemoryUseRecorder's lock.
  private long peakPostGCHeapMemoryUsed = 0;

  // Protected by PostGcMemoryUseRecorder's lock. Set to true iff a GarbageCollectionNotification
  // reported that we were using no memory.
  private boolean memoryUsageReportedZero = false;

  @VisibleForTesting
  PostGCMemoryUseRecorder(Iterable<GarbageCollectorMXBean> mxBeans) {
    for (GarbageCollectorMXBean mxBean : mxBeans) {
      // The "Copy" collector only does minor collections.
      if ("Copy".equals(mxBean.getName())) {
        continue;
      }
      logger.info("Listening for notifications from GC: " + mxBean.getName());
      ((NotificationEmitter) mxBean).addNotificationListener(this, null, null);
    }
  }

  public synchronized long getPeakPostGCHeapMemoryUsed() {
    return peakPostGCHeapMemoryUsed;
  }

  public synchronized boolean wasMemoryUsageReportedZero() {
    return memoryUsageReportedZero;
  }

  public synchronized void reset() {
    peakPostGCHeapMemoryUsed = 0;
    memoryUsageReportedZero = false;
  }

  private synchronized void updatePostGCHeapMemoryUsed(long used) {
    peakPostGCHeapMemoryUsed = Math.max(used, peakPostGCHeapMemoryUsed);
  }

  private synchronized void updateMemoryUsageReportedZero(boolean value) {
    memoryUsageReportedZero = value;
  }

  @Override
  public void handleNotification(Notification notification, Object handback) {
    doHandleNotification(notification, handback);
  }

  public void doHandleNotification(Notification notification, Object handback) {
    if (!notification
        .getType()
        .equals(GarbageCollectionNotificationInfo.GARBAGE_COLLECTION_NOTIFICATION)) {
      return;
    }

    GarbageCollectionNotificationInfo info =
        GarbageCollectionNotificationInfo.from((CompositeData) notification.getUserData());
    long durationNs = info.getGcInfo().getDuration() * 1_000_000;
    long end = Profiler.nanoTimeMaybe();
    if (wasStopTheWorldGc(info)) {
      Profiler.instance()
          .logSimpleTask(
              end - durationNs,
              end,
              ProfilerTask.HANDLE_GC_NOTIFICATION,
              info.getGcAction().replaceFirst("^end of ", ""));
    }
    if (!info.getGcAction().equals("end of major GC")) {
      return;
    }

    long used = 0;
    Map<String, MemoryUsage> mem = info.getGcInfo().getMemoryUsageAfterGc();
    for (MemoryUsage mu : mem.values()) {
      used += mu.getUsed();
    }
    updatePostGCHeapMemoryUsed(used);
    if (used > 0) {
      logger.info("Memory use after full GC: " + used);
    } else {
      logger.info(
          "Amount of memory used after GC incorrectly reported as "
              + used
              + " by JVM with values "
              + mem);
      updateMemoryUsageReportedZero(true);
    }
  }

  /** Module to support "blaze info peak-heap-size". */
  public static class PostGCMemoryUseRecorderModule extends BlazeModule {
    @Override
    public void serverInit(OptionsParsingResult startupOptions, ServerBuilder builder) {
      builder.addInfoItems(new PeakMemInfoItem());
    }

    @Override
    public void beforeCommand(CommandEnvironment env) {
      if (!env.getCommandName().equals("info")) {
        PostGCMemoryUseRecorder.get().reset();
      }
    }
  }

  static class PeakMemInfoItem extends InfoItem {
    PeakMemInfoItem() {
      super(
          "peak-heap-size",
          "The peak amount of used memory in bytes after any call to System.gc().",
          true);
    }

    @Override
    public byte[] get(Supplier<BuildConfiguration> configurationSupplier, CommandEnvironment env) {
      return print(
          StringUtilities.prettyPrintBytes(
              PostGCMemoryUseRecorder.get().getPeakPostGCHeapMemoryUsed()));
    }
  }

  /** Module to run a full GC after a build is complete on a Blaze server. * */
  public static class GcAfterBuildModule extends BlazeModule {

    private boolean forceGc = false;

    /** Command options for forcing a GC after a build. * */
    public static class Options extends OptionsBase {
      @Option(
          name = "experimental_force_gc_after_build",
          defaultValue = "false",
          documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
          effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
          help =
              "If true calls System.gc() after a build to try and get a post-gc peak heap"
                  + " measurement.")
      public boolean experimentalForceGcAfterBuild;
    }

    @Override
    public Iterable<Class<? extends OptionsBase>> getCommandOptions(Command command) {
      return ImmutableList.of(Options.class);
    }

    @Override
    public void beforeCommand(CommandEnvironment env) {
      Options options = env.getOptions().getOptions(Options.class);
      if (options != null
          && ("test".equals(env.getCommand().name()) || "build".equals(env.getCommand().name()))) {
        forceGc = options.experimentalForceGcAfterBuild;
      } else {
        forceGc = false;
      }
    }

    @Override
    public void afterCommand() {
      if (forceGc && PostGCMemoryUseRecorder.get().getPeakPostGCHeapMemoryUsed() == 0L) {
        System.gc();
      }
    }
  }

  private static boolean wasStopTheWorldGc(GarbageCollectionNotificationInfo info) {
    // Weak heuristic to determine if this was a STW gc.
    return !"No GC".equals(info.getGcCause());
  }
}
