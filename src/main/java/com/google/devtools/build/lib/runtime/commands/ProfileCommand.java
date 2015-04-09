// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.runtime.commands;

import com.google.common.base.Function;
import com.google.common.base.Functions;
import com.google.common.base.Preconditions;
import com.google.common.collect.Maps;
import com.google.common.collect.Multimaps;
import com.google.common.collect.Ordering;
import com.google.common.collect.TreeMultimap;
import com.google.devtools.build.lib.actions.MiddlemanAction;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.profiler.ProfileInfo;
import com.google.devtools.build.lib.profiler.ProfileInfo.CriticalPathEntry;
import com.google.devtools.build.lib.profiler.ProfileInfo.InfoListener;
import com.google.devtools.build.lib.profiler.ProfilePhase;
import com.google.devtools.build.lib.profiler.ProfilePhaseStatistics;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.profiler.chart.AggregatingChartCreator;
import com.google.devtools.build.lib.profiler.chart.Chart;
import com.google.devtools.build.lib.profiler.chart.ChartCreator;
import com.google.devtools.build.lib.profiler.chart.DetailedChartCreator;
import com.google.devtools.build.lib.profiler.chart.HtmlChartVisitor;
import com.google.devtools.build.lib.runtime.BlazeCommand;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.StringUtil;
import com.google.devtools.build.lib.util.TimeUtilities;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.common.options.Converters;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsProvider;

import java.io.BufferedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.EnumMap;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;

/**
 * Command line wrapper for analyzing Blaze build profiles.
 */
@Command(name = "analyze-profile",
         options = { ProfileCommand.ProfileOptions.class },
         shortDescription = "Analyzes build profile data.",
         help = "resource:analyze-profile.txt",
         allowResidue = true,
         completion = "path",
         mustRunInWorkspace = false)
public final class ProfileCommand implements BlazeCommand {

  private final String TWO_COLUMN_FORMAT = "%-37s %10s\n";
  private final String THREE_COLUMN_FORMAT = "%-28s %10s %8s\n";

  public static class DumpConverter extends Converters.StringSetConverter {
    public DumpConverter() {
      super("text", "raw", "text-unsorted", "raw-unsorted");
    }
  }

  public static class ProfileOptions extends OptionsBase {
    @Option(name = "dump",
        abbrev='d',
        converter = DumpConverter.class,
        defaultValue = "null",
        help = "output full profile data dump either in human-readable 'text' format or"
            + " script-friendly 'raw' format, either sorted or unsorted.")
    public String dumpMode;

    @Option(name = "html",
        defaultValue = "false",
        help = "If present, an HTML file visualizing the tasks of the profiled build is created. "
            + "The name of the html file is the name of the profile file plus '.html'.")
    public boolean html;

    @Option(name = "html_pixels_per_second",
        defaultValue = "50",
        help = "Defines the scale of the time axis of the task diagram. The unit is "
            + "pixels per second. Default is 50 pixels per second. ")
    public int htmlPixelsPerSecond;

    @Option(name = "html_details",
        defaultValue = "false",
        help = "If --html_details is present, the task diagram contains all tasks of the profile. "
            + "If --nohtml_details is present, an aggregated diagram is generated. The default is "
            + "to generate an aggregated diagram.")
    public boolean htmlDetails;

    @Option(name = "vfs_stats",
        defaultValue = "false",
        help = "If present, include VFS path statistics.")
    public boolean vfsStats;

    @Option(name = "vfs_stats_limit",
        defaultValue = "-1",
        help = "Maximum number of VFS path statistics to print.")
    public int vfsStatsLimit;
  }

  private Function<String, String> currentPathMapping = Functions.<String>identity();

  private InfoListener getInfoListener(final BlazeRuntime runtime) {
    return new InfoListener() {
      private final EventHandler reporter = runtime.getReporter();

      @Override
      public void info(String text) {
        reporter.handle(Event.info(text));
      }

      @Override
      public void warn(String text) {
        reporter.handle(Event.warn(text));
      }
    };
  }

  @Override
  public void editOptions(BlazeRuntime runtime, OptionsParser optionsParser) {}

  @Override
  public ExitCode exec(final BlazeRuntime runtime, OptionsProvider options) {
    ProfileOptions opts =
        options.getOptions(ProfileOptions.class);

    if (!opts.vfsStats) {
      opts.vfsStatsLimit = 0;
    }

    currentPathMapping = new Function<String, String>() {
      @Override
      public String apply(String input) {
        if (runtime.getWorkspaceName().isEmpty()) {
          return input;
        } else {
          return input.substring(input.lastIndexOf("/" + runtime.getWorkspaceName()) + 1);
        }
      }
    };

    PrintStream out = new PrintStream(runtime.getReporter().getOutErr().getOutputStream());
    try {
      runtime.getReporter().handle(Event.warn(
          null, "This information is intended for consumption by Blaze developers"
              + " only, and may change at any time.  Script against it at your own risk"));

      for (String name : options.getResidue()) {
        Path profileFile = runtime.getWorkingDirectory().getRelative(name);
        try {
          ProfileInfo info = ProfileInfo.loadProfileVerbosely(
              profileFile, getInfoListener(runtime));
          if (opts.dumpMode != null) {
            dumpProfile(runtime, info, out, opts.dumpMode);
          } else if (opts.html) {
            createHtml(runtime, info, profileFile, opts);
          } else {
            createText(runtime, info, out, opts);
          }
        } catch (IOException e) {
          runtime.getReporter().handle(Event.error(
              null, "Failed to process file " + name + ": " + e.getMessage()));
        }
      }
    } finally {
      out.flush();
    }
    return ExitCode.SUCCESS;
  }

  private void createText(BlazeRuntime runtime, ProfileInfo info, PrintStream out,
      ProfileOptions opts) {
    List<ProfilePhaseStatistics> statistics = getStatistics(runtime, info, opts);

    for (ProfilePhaseStatistics stat : statistics) {
      String title = stat.getTitle();

      if (!title.isEmpty()) {
        out.println("\n=== " + title.toUpperCase() + " ===\n");
      }
      out.print(stat.getStatistics());
    }
  }

  private void createHtml(BlazeRuntime runtime, ProfileInfo info, Path profileFile,
      ProfileOptions opts)
      throws IOException {
    Path htmlFile =
        profileFile.getParentDirectory().getChild(profileFile.getBaseName() + ".html");
    List<ProfilePhaseStatistics> statistics = getStatistics(runtime, info, opts);

    runtime.getReporter().handle(Event.info("Creating HTML output in " + htmlFile));

    ChartCreator chartCreator =
        opts.htmlDetails ? new DetailedChartCreator(info, statistics)
                         : new AggregatingChartCreator(info, statistics);
    Chart chart = chartCreator.create();
    OutputStream out = new BufferedOutputStream(htmlFile.getOutputStream());
    try {
      chart.accept(new HtmlChartVisitor(new PrintStream(out), opts.htmlPixelsPerSecond));
    } finally {
      try {
        out.close();
      } catch (IOException e) {
        // Ignore
      }
    }
  }

  private List<ProfilePhaseStatistics> getStatistics(
      BlazeRuntime runtime, ProfileInfo info, ProfileOptions opts) {
    try {
      ProfileInfo.aggregateProfile(info, getInfoListener(runtime));
      runtime.getReporter().handle(Event.info("Analyzing relationships"));

      info.analyzeRelationships();

      List<ProfilePhaseStatistics> statistics = new ArrayList<>();

      // Print phase durations and total execution time
      ByteArrayOutputStream byteOutput = new ByteArrayOutputStream();
      PrintStream out = new PrintStream(byteOutput, false, "UTF-8");
      long duration = 0;
      for (ProfilePhase phase : ProfilePhase.values()) {
        ProfileInfo.Task phaseTask = info.getPhaseTask(phase);
        if (phaseTask != null) {
          duration += info.getPhaseDuration(phaseTask);
        }
      }
      for (ProfilePhase phase : ProfilePhase.values()) {
        ProfileInfo.Task phaseTask = info.getPhaseTask(phase);
        if (phaseTask != null) {
          long phaseDuration = info.getPhaseDuration(phaseTask);
          out.printf(THREE_COLUMN_FORMAT, "Total " + phase.nick + " phase time",
              TimeUtilities.prettyTime(phaseDuration), prettyPercentage(phaseDuration, duration));
        }
      }
      out.printf(THREE_COLUMN_FORMAT, "Total run time", TimeUtilities.prettyTime(duration),
          "100.00%");
      statistics.add(new ProfilePhaseStatistics("Phase Summary Information",
          new String(byteOutput.toByteArray(), "UTF-8")));

      // Print details of major phases
      if (duration > 0) {
        statistics.add(formatInitPhaseStatistics(info, opts));
        statistics.add(formatLoadingPhaseStatistics(info, opts));
        statistics.add(formatAnalysisPhaseStatistics(info, opts));
        ProfilePhaseStatistics stat = formatExecutionPhaseStatistics(info, opts);
        if (stat != null) {
          statistics.add(stat);
        }
      }

      return statistics;
    } catch (UnsupportedEncodingException e) {
      throw new AssertionError("Should not happen since, UTF8 is available on all JVMs");
    }
  }

  private void dumpProfile(
      BlazeRuntime runtime, ProfileInfo info, PrintStream out, String dumpMode) {
    if (!dumpMode.contains("unsorted")) {
      ProfileInfo.aggregateProfile(info, getInfoListener(runtime));
    }
    if (dumpMode.contains("raw")) {
      for (ProfileInfo.Task task : info.allTasksById) {
        dumpRaw(task, out);
      }
    } else if (dumpMode.contains("unsorted")) {
      for (ProfileInfo.Task task : info.allTasksById) {
        dumpTask(task, out, 0);
      }
    } else {
      for (ProfileInfo.Task task : info.rootTasksById) {
        dumpTask(task, out, 0);
      }
    }
  }

  private void dumpTask(ProfileInfo.Task task, PrintStream out, int indent) {
    StringBuilder builder = new StringBuilder(String.format(
        "\n%s %s\nThread: %-6d  Id: %-6d  Parent: %d\nStart time: %-12s   Duration: %s",
        task.type, task.getDescription(), task.threadId, task.id, task.parentId,
        TimeUtilities.prettyTime(task.startTime), TimeUtilities.prettyTime(task.duration)));
    if (task.hasStats()) {
      builder.append("\n");
      ProfileInfo.AggregateAttr[] stats = task.getStatAttrArray();
      for (ProfilerTask type : ProfilerTask.values()) {
        ProfileInfo.AggregateAttr attr = stats[type.ordinal()];
        if (attr != null) {
          builder.append(type.toString().toLowerCase()).append("=(").
              append(attr.count).append(", ").
              append(TimeUtilities.prettyTime(attr.totalTime)).append(") ");
        }
      }
    }
    out.println(StringUtil.indent(builder.toString(), indent));
    for (ProfileInfo.Task subtask : task.subtasks) {
      dumpTask(subtask, out, indent + 1);
    }
  }

  private void dumpRaw(ProfileInfo.Task task, PrintStream out) {
    StringBuilder aggregateString = new StringBuilder();
    ProfileInfo.AggregateAttr[] stats = task.getStatAttrArray();
    for (ProfilerTask type : ProfilerTask.values()) {
      ProfileInfo.AggregateAttr attr = stats[type.ordinal()];
      if (attr != null) {
        aggregateString.append(type.toString().toLowerCase()).append(",").
            append(attr.count).append(",").append(attr.totalTime).append(" ");
      }
    }
    out.println(
        task.threadId + "|" + task.id + "|" + task.parentId + "|"
        + task.startTime + "|" + task.duration + "|"
        + aggregateString.toString().trim() + "|"
        + task.type + "|" + task.getDescription());
  }

  /**
   * Converts relative duration to the percentage string
   * @return formatted percentage string or "N/A" if result is undefined.
   */
  private static String prettyPercentage(long duration, long total) {
    if (total == 0) {
      // Return "not available" string if total is 0 and result is undefined.
      return "N/A";
    }
    return String.format("%5.2f%%", duration*100.0/total);
  }

  private void printCriticalPath(String title, PrintStream out, CriticalPathEntry path) {
    out.printf("\n%s (%s):%n", title, TimeUtilities.prettyTime(path.cumulativeDuration));

    boolean lightCriticalPath = isLightCriticalPath(path);
    out.println(lightCriticalPath ?
        String.format("%6s %11s %8s   %s", "Id", "Time", "Percentage", "Description")
        : String.format("%6s %11s %8s %8s   %s", "Id", "Time", "Share", "Critical", "Description"));

    long totalPathTime = path.cumulativeDuration;
    int middlemanCount = 0;
    long middlemanDuration = 0L;
    long middlemanCritTime = 0L;

    for (; path != null ; path = path.next) {
      if (path.task.id < 0) {
        // Ignore fake actions.
        continue;
      } else if (path.task.getDescription().startsWith(MiddlemanAction.MIDDLEMAN_MNEMONIC + " ")
          || path.task.getDescription().startsWith("TargetCompletionMiddleman")) {
        // Aggregate middleman actions.
        middlemanCount++;
        middlemanDuration += path.duration;
        middlemanCritTime += path.getCriticalTime();
      } else {
        String desc = path.task.getDescription().replace(':', ' ');
        if (lightCriticalPath) {
          out.printf("%6d %11s %8s   %s%n", path.task.id, TimeUtilities.prettyTime(path.duration),
              prettyPercentage(path.duration, totalPathTime), desc);
        } else {
          out.printf("%6d %11s %8s %8s   %s%n", path.task.id,
              TimeUtilities.prettyTime(path.duration),
              prettyPercentage(path.duration, totalPathTime),
              prettyPercentage(path.getCriticalTime(), totalPathTime), desc);
        }
      }
    }
    if (middlemanCount > 0) {
      if (lightCriticalPath) {
        out.printf("       %11s %8s   [%d middleman actions]%n",
            TimeUtilities.prettyTime(middlemanDuration),
            prettyPercentage(middlemanDuration, totalPathTime), middlemanCount);
      } else {
        out.printf("       %11s %8s %8s   [%d middleman actions]%n",
            TimeUtilities.prettyTime(middlemanDuration),
            prettyPercentage(middlemanDuration, totalPathTime),
            prettyPercentage(middlemanCritTime, totalPathTime), middlemanCount);
      }
    }
  }

  private boolean isLightCriticalPath(CriticalPathEntry path) {
    return path.task.type == ProfilerTask.CRITICAL_PATH_COMPONENT;
  }

  private void printShortPhaseAnalysis(ProfileInfo info, PrintStream out, ProfilePhase phase) {
    ProfileInfo.Task phaseTask = info.getPhaseTask(phase);
    if (phaseTask != null) {
      long phaseDuration = info.getPhaseDuration(phaseTask);
      out.printf(TWO_COLUMN_FORMAT, "Total " + phase.nick + " phase time",
          TimeUtilities.prettyTime(phaseDuration));
      printTimeDistributionByType(info, out, phaseTask);
    }
  }

  private void printTimeDistributionByType(ProfileInfo info, PrintStream out,
      ProfileInfo.Task phaseTask) {
    List<ProfileInfo.Task> taskList = info.getTasksForPhase(phaseTask);
    long phaseDuration = info.getPhaseDuration(phaseTask);
    long totalDuration = phaseDuration;
    for (ProfileInfo.Task task : taskList) {
      // Tasks on the phaseTask thread already accounted for in the phaseDuration.
      if (task.threadId != phaseTask.threadId) {
        totalDuration += task.duration;
      }
    }
    boolean headerNeeded = true;
    for (ProfilerTask type : ProfilerTask.values()) {
      ProfileInfo.AggregateAttr stats = info.getStatsForType(type, taskList);
      if (stats.count > 0 && stats.totalTime > 0) {
        if (headerNeeded) {
          out.println("\nTotal time (across all threads) spent on:");
          out.printf("%18s %8s %8s %11s%n", "Type", "Total", "Count", "Average");
          headerNeeded = false;
        }
        out.printf("%18s %8s %8d %11s%n", type.toString(),
            prettyPercentage(stats.totalTime, totalDuration), stats.count,
            TimeUtilities.prettyTime(stats.totalTime / stats.count));
      }
    }
  }

  static class Stat implements Comparable<Stat> {
    public long duration;
    public long frequency;

    @Override
    public int compareTo(Stat o) {
      return this.duration == o.duration ? Long.compare(this.frequency, o.frequency)
          : Long.compare(this.duration, o.duration);
    }
  }

  /**
   * Print the time spent on VFS operations on each path. Output is grouped by operation and sorted
   * by descending duration. If multiple of the same VFS operation were logged for the same path,
   * print the total duration.
   *
   * @param info profiling data.
   * @param out output stream.
   * @param phase build phase.
   * @param limit maximum number of statistics to print, or -1 for no limit.
   */
  private void printVfsStatistics(ProfileInfo info, PrintStream out,
                                  ProfilePhase phase, int limit) {
    ProfileInfo.Task phaseTask = info.getPhaseTask(phase);
    if (phaseTask == null) {
      return;
    }

    if (limit == 0) {
      return;
    }

    // Group into VFS operations and build maps from path to duration.

    List<ProfileInfo.Task> taskList = info.getTasksForPhase(phaseTask);
    EnumMap<ProfilerTask, Map<String, Stat>> stats = Maps.newEnumMap(ProfilerTask.class);

    collectVfsEntries(stats, taskList);

    if (!stats.isEmpty()) {
      out.printf("\nVFS path statistics:\n");
      out.printf("%15s %10s %10s %s\n", "Type", "Frequency", "Duration", "Path");
    }

    // Reverse the maps to get maps from duration to path. We use a TreeMultimap to sort by duration
    // and because durations are not unique.

    for (ProfilerTask type : stats.keySet()) {
      Map<String, Stat> statsForType = stats.get(type);
      TreeMultimap<Stat, String> sortedStats =
          TreeMultimap.create(Ordering.natural().reverse(), Ordering.natural());

      Multimaps.invertFrom(Multimaps.forMap(statsForType), sortedStats);

      int numPrinted = 0;
      for (Map.Entry<Stat, String> stat : sortedStats.entries()) {
        if (limit != -1 && numPrinted++ == limit) {
          out.printf("... %d more ...\n", sortedStats.size() - limit);
          break;
        }
        out.printf("%15s %10d %10s %s\n",
            type.name(), stat.getKey().frequency, TimeUtilities.prettyTime(stat.getKey().duration),
            stat.getValue());
      }
    }
  }

  private void collectVfsEntries(EnumMap<ProfilerTask, Map<String, Stat>> stats,
      List<ProfileInfo.Task> taskList) {
    for (ProfileInfo.Task task : taskList) {
      collectVfsEntries(stats, Arrays.asList(task.subtasks));
      if (!task.type.name().startsWith("VFS_")) {
        continue;
      }

      Map<String, Stat> statsForType = stats.get(task.type);
      if (statsForType == null) {
        statsForType = Maps.newHashMap();
        stats.put(task.type, statsForType);
      }

      String path = currentPathMapping.apply(task.getDescription());

      Stat stat = statsForType.get(path);
      if (stat == null) {
        stat = new Stat();
      }

      stat.duration += task.duration;
      stat.frequency++;
      statsForType.put(path, stat);
    }
  }

  /**
   * Returns set of profiler tasks to be filtered from critical path.
   * Also always filters out ACTION_LOCK and WAIT tasks to simulate
   * unlimited resource critical path (see comments inside formatExecutionPhaseStatistics()
   * method).
   */
  private EnumSet<ProfilerTask> getTypeFilter(ProfilerTask... tasks) {
    EnumSet<ProfilerTask> filter = EnumSet.of(ProfilerTask.ACTION_LOCK, ProfilerTask.WAIT);
    Collections.addAll(filter, tasks);
    return filter;
  }

  private ProfilePhaseStatistics formatInitPhaseStatistics(ProfileInfo info, ProfileOptions opts)
      throws UnsupportedEncodingException {
    return formatSimplePhaseStatistics(info, opts, "Init", ProfilePhase.INIT);
  }

  private ProfilePhaseStatistics formatLoadingPhaseStatistics(ProfileInfo info, ProfileOptions opts)
      throws UnsupportedEncodingException {
    return formatSimplePhaseStatistics(info, opts, "Loading", ProfilePhase.LOAD);
  }

  private ProfilePhaseStatistics formatAnalysisPhaseStatistics(ProfileInfo info,
                                                               ProfileOptions opts)
      throws UnsupportedEncodingException {
    return formatSimplePhaseStatistics(info, opts, "Analysis", ProfilePhase.ANALYZE);
  }

  private ProfilePhaseStatistics formatSimplePhaseStatistics(ProfileInfo info,
                                                             ProfileOptions opts,
                                                             String name,
                                                             ProfilePhase phase)
      throws UnsupportedEncodingException {
    ByteArrayOutputStream byteOutput = new ByteArrayOutputStream();
    PrintStream out = new PrintStream(byteOutput, false, "UTF-8");

    printShortPhaseAnalysis(info, out, phase);
    printVfsStatistics(info, out, phase, opts.vfsStatsLimit);
    return new ProfilePhaseStatistics(name + " Phase Information",
        new String(byteOutput.toByteArray(), "UTF-8"));
  }

  private ProfilePhaseStatistics formatExecutionPhaseStatistics(ProfileInfo info,
                                                                ProfileOptions opts)
      throws UnsupportedEncodingException {
    ByteArrayOutputStream byteOutput = new ByteArrayOutputStream();
    PrintStream out = new PrintStream(byteOutput, false, "UTF-8");

    ProfileInfo.Task prepPhase = info.getPhaseTask(ProfilePhase.PREPARE);
    ProfileInfo.Task execPhase = info.getPhaseTask(ProfilePhase.EXECUTE);
    ProfileInfo.Task finishPhase = info.getPhaseTask(ProfilePhase.FINISH);
    if (execPhase == null) {
      return null;
    }

    List<ProfileInfo.Task> execTasks = info.getTasksForPhase(execPhase);
    long graphTime = info.getStatsForType(ProfilerTask.ACTION_GRAPH, execTasks).totalTime;
    long execTime = info.getPhaseDuration(execPhase) - graphTime;

    if (prepPhase != null) {
      out.printf(TWO_COLUMN_FORMAT, "Total preparation time",
          TimeUtilities.prettyTime(info.getPhaseDuration(prepPhase)));
    }
    out.printf(TWO_COLUMN_FORMAT, "Total execution phase time",
        TimeUtilities.prettyTime(info.getPhaseDuration(execPhase)));
    if (finishPhase != null) {
      out.printf(TWO_COLUMN_FORMAT, "Total time finalizing build",
          TimeUtilities.prettyTime(info.getPhaseDuration(finishPhase)));
    }
    out.println();
    out.printf(TWO_COLUMN_FORMAT, "Action dependency map creation",
        TimeUtilities.prettyTime(graphTime));
    out.printf(TWO_COLUMN_FORMAT, "Actual execution time",
        TimeUtilities.prettyTime(execTime));

    EnumSet<ProfilerTask> typeFilter = EnumSet.noneOf(ProfilerTask.class);
    CriticalPathEntry totalPath = info.getCriticalPath(typeFilter);
    info.analyzeCriticalPath(typeFilter, totalPath);

    typeFilter = getTypeFilter();
    CriticalPathEntry optimalPath = info.getCriticalPath(typeFilter);
    info.analyzeCriticalPath(typeFilter, optimalPath);

    if (totalPath != null) {
      printCriticalPathTimingBreakdown(info, totalPath, optimalPath, execTime, out);
    } else {
      out.println("\nCritical path not available because no action graph was generated.");
    }

    printTimeDistributionByType(info, out, execPhase);

    if (totalPath != null) {
      printCriticalPath("Critical path", out, totalPath);
      // In light critical path we do not record scheduling delay data so it does not make sense
      // to differentiate it.
      if (!isLightCriticalPath(totalPath)) {
        printCriticalPath("Critical path excluding scheduling delays", out, optimalPath);
      }
    }

    if (info.getMissingActionsCount() > 0) {
      out.println("\n" + info.getMissingActionsCount() + " action(s) are present in the"
          + " action graph but missing instrumentation data. Most likely profile file"
          + " has been created for the failed or aborted build.");
    }

    printVfsStatistics(info, out, ProfilePhase.EXECUTE, opts.vfsStatsLimit);

    return new ProfilePhaseStatistics("Execution Phase Information",
        new String(byteOutput.toByteArray(), "UTF-8"));
  }

  void printCriticalPathTimingBreakdown(ProfileInfo info, CriticalPathEntry totalPath,
      CriticalPathEntry optimalPath, long execTime, PrintStream out) {
    Preconditions.checkNotNull(totalPath);
    Preconditions.checkNotNull(optimalPath);
    // TODO(bazel-team): Print remote vs build stats recorded by CriticalPathStats
    if (isLightCriticalPath(totalPath)) {
      return;
    }
    out.println(totalPath.task.type);
    // Worker thread pool scheduling delays for the actual critical path.
    long workerWaitTime = 0;
    long mainThreadWaitTime = 0;
    for (ProfileInfo.CriticalPathEntry entry = totalPath; entry != null; entry = entry.next) {
      workerWaitTime += info.getActionWaitTime(entry.task);
      mainThreadWaitTime += info.getActionQueueTime(entry.task);
    }
    out.printf(TWO_COLUMN_FORMAT, "Worker thread scheduling delays",
        TimeUtilities.prettyTime(workerWaitTime));
    out.printf(TWO_COLUMN_FORMAT, "Main thread scheduling delays",
        TimeUtilities.prettyTime(mainThreadWaitTime));

    out.println("\nCritical path time:");
    // Actual critical path.
    long totalTime = totalPath.cumulativeDuration;
    out.printf("%-37s %10s (%s of execution time)\n", "Actual time",
        TimeUtilities.prettyTime(totalTime),
        prettyPercentage(totalTime, execTime));
    // Unlimited resource critical path. Essentially, we assume that if we
    // remove all scheduling delays caused by resource semaphore contention,
    // each action execution time would not change (even though load now would
    // be substantially higher - so this assumption might be incorrect but it is
    // still useful for modeling). Given those assumptions we calculate critical
    // path excluding scheduling delays.
    long optimalTime = optimalPath.cumulativeDuration;
    out.printf("%-37s %10s (%s of execution time)\n", "Time excluding scheduling delays",
        TimeUtilities.prettyTime(optimalTime),
        prettyPercentage(optimalTime, execTime));

    // Artificial critical path if we ignore all the time spent in all tasks,
    // except time directly attributed to the ACTION tasks.
    out.println("\nTime related to:");

    EnumSet<ProfilerTask> typeFilter = EnumSet.allOf(ProfilerTask.class);
    ProfileInfo.CriticalPathEntry path = info.getCriticalPath(typeFilter);
    out.printf(TWO_COLUMN_FORMAT, "the builder overhead",
        prettyPercentage(path.cumulativeDuration, totalTime));

    typeFilter = getTypeFilter();
    for (ProfilerTask task : ProfilerTask.values()) {
      if (task.name().startsWith("VFS_")) {
        typeFilter.add(task);
      }
    }
    path = info.getCriticalPath(typeFilter);
    out.printf(TWO_COLUMN_FORMAT, "the VFS calls",
        prettyPercentage(optimalTime - path.cumulativeDuration, optimalTime));

    typeFilter = getTypeFilter(ProfilerTask.ACTION_CHECK);
    path = info.getCriticalPath(typeFilter);
    out.printf(TWO_COLUMN_FORMAT, "the dependency checking",
        prettyPercentage(optimalTime - path.cumulativeDuration, optimalTime));

    typeFilter = getTypeFilter(ProfilerTask.ACTION_EXECUTE);
    path = info.getCriticalPath(typeFilter);
    out.printf(TWO_COLUMN_FORMAT, "the execution setup",
        prettyPercentage(optimalTime - path.cumulativeDuration, optimalTime));

    typeFilter = getTypeFilter(ProfilerTask.SPAWN, ProfilerTask.LOCAL_EXECUTION);
    path = info.getCriticalPath(typeFilter);
    out.printf(TWO_COLUMN_FORMAT, "local execution",
        prettyPercentage(optimalTime - path.cumulativeDuration, optimalTime));

    typeFilter = getTypeFilter(ProfilerTask.SCANNER);
    path = info.getCriticalPath(typeFilter);
    out.printf(TWO_COLUMN_FORMAT, "the include scanner",
        prettyPercentage(optimalTime - path.cumulativeDuration, optimalTime));

    typeFilter = getTypeFilter(ProfilerTask.REMOTE_EXECUTION, ProfilerTask.PROCESS_TIME,
        ProfilerTask.LOCAL_PARSE,  ProfilerTask.UPLOAD_TIME,
        ProfilerTask.REMOTE_QUEUE,  ProfilerTask.REMOTE_SETUP, ProfilerTask.FETCH);
    path = info.getCriticalPath(typeFilter);
    out.printf(TWO_COLUMN_FORMAT, "Remote execution (cumulative)",
        prettyPercentage(optimalTime - path.cumulativeDuration, optimalTime));

    typeFilter = getTypeFilter( ProfilerTask.UPLOAD_TIME, ProfilerTask.REMOTE_SETUP);
    path = info.getCriticalPath(typeFilter);
    out.printf(TWO_COLUMN_FORMAT, "  file uploads",
        prettyPercentage(optimalTime - path.cumulativeDuration, optimalTime));

    typeFilter = getTypeFilter(ProfilerTask.FETCH);
    path = info.getCriticalPath(typeFilter);
    out.printf(TWO_COLUMN_FORMAT, "  file fetching",
        prettyPercentage(optimalTime - path.cumulativeDuration, optimalTime));

    typeFilter = getTypeFilter(ProfilerTask.PROCESS_TIME);
    path = info.getCriticalPath(typeFilter);
    out.printf(TWO_COLUMN_FORMAT, "  process time",
        prettyPercentage(optimalTime - path.cumulativeDuration, optimalTime));

    typeFilter = getTypeFilter(ProfilerTask.REMOTE_QUEUE);
    path = info.getCriticalPath(typeFilter);
    out.printf(TWO_COLUMN_FORMAT, "  remote queueing",
        prettyPercentage(optimalTime - path.cumulativeDuration, optimalTime));

    typeFilter = getTypeFilter(ProfilerTask.LOCAL_PARSE);
    path = info.getCriticalPath(typeFilter);
    out.printf(TWO_COLUMN_FORMAT, "  remote execution parse",
        prettyPercentage(optimalTime - path.cumulativeDuration, optimalTime));

    typeFilter = getTypeFilter(ProfilerTask.REMOTE_EXECUTION);
    path = info.getCriticalPath(typeFilter);
    out.printf(TWO_COLUMN_FORMAT, "  other remote activities",
        prettyPercentage(optimalTime - path.cumulativeDuration, optimalTime));
  }
}
