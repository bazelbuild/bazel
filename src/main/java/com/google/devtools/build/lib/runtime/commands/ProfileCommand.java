// Copyright 2014 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Joiner;
import com.google.common.base.Optional;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.profiler.ProfilePhase;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.profiler.analysis.ProfileInfo;
import com.google.devtools.build.lib.profiler.analysis.ProfileInfo.InfoListener;
import com.google.devtools.build.lib.profiler.output.PhaseText;
import com.google.devtools.build.lib.profiler.statistics.CriticalPathStatistics;
import com.google.devtools.build.lib.profiler.statistics.PhaseStatistics;
import com.google.devtools.build.lib.profiler.statistics.PhaseSummaryStatistics;
import com.google.devtools.build.lib.runtime.BlazeCommand;
import com.google.devtools.build.lib.runtime.BlazeCommandResult;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.StringUtil;
import com.google.devtools.build.lib.util.TimeUtilities;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.common.options.Converters;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParsingResult;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.EnumMap;

/** Command line wrapper for analyzing Blaze build profiles. */
@Command(
  name = "analyze-profile",
  options = {ProfileCommand.ProfileOptions.class},
  shortDescription = "Analyzes build profile data.",
  help = "resource:analyze-profile.txt",
  allowResidue = true,
  completion = "path",
  mustRunInWorkspace = false
)
public final class ProfileCommand implements BlazeCommand {

  public static class DumpConverter extends Converters.StringSetConverter {
    public DumpConverter() {
      super("text", "raw");
    }
  }

  public static class ProfileOptions extends OptionsBase {
    @Option(
        name = "dump",
        abbrev = 'd',
        converter = DumpConverter.class,
        defaultValue = "null",
        documentationCategory = OptionDocumentationCategory.LOGGING,
        effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
        help =
            "output full profile data dump either in human-readable 'text' format or"
                + " script-friendly 'raw' format.")
    public String dumpMode;
  }

  private InfoListener getInfoListener(final CommandEnvironment env) {
    return new InfoListener() {
      private final EventHandler reporter = env.getReporter();

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
  public BlazeCommandResult exec(final CommandEnvironment env, OptionsParsingResult options) {
    ProfileOptions opts =
        options.getOptions(ProfileOptions.class);

    try (PrintStream out = getOutputStream(env)) {
      env.getReporter()
          .handle(
              Event.warn(
                  null,
                  "This information is intended for consumption by Bazel developers"
                      + " only, and may change at any time. Script against it at your own risk"));
        for (String name : options.getResidue()) {
          Path profileFile = env.getWorkingDirectory().getRelative(name);
          try {
            ProfileInfo info = ProfileInfo.loadProfileVerbosely(profileFile, getInfoListener(env));

          if (opts.dumpMode == null) {
              ProfileInfo.aggregateProfile(info, getInfoListener(env));
          } else {
              dumpProfile(info, out, opts.dumpMode);
              continue;
            }

            PhaseSummaryStatistics phaseSummaryStatistics = new PhaseSummaryStatistics(info);
            EnumMap<ProfilePhase, PhaseStatistics> phaseStatistics =
                new EnumMap<>(ProfilePhase.class);

            Path workspace = env.getWorkspace();
            for (ProfilePhase phase : ProfilePhase.values()) {
            phaseStatistics.put(
                phase,
                new PhaseStatistics(
                    phase, info, (workspace == null ? "<workspace>" : workspace.getBaseName())));
            }

            CriticalPathStatistics critPathStats = new CriticalPathStatistics(info);
          new PhaseText(out, phaseSummaryStatistics, phaseStatistics, Optional.of(critPathStats))
              .print();
          } catch (IOException e) {
            System.out.println(e);
            env
                .getReporter()
                .handle(Event.error("Failed to analyze profile file(s): " + e.getMessage()));
            return BlazeCommandResult.exitCode(ExitCode.PARSING_FAILURE);
          }
        }
      }
    return BlazeCommandResult.exitCode(ExitCode.SUCCESS);
  }

  private static PrintStream getOutputStream(CommandEnvironment env) {
    return new PrintStream(
        new BufferedOutputStream(env.getReporter().getOutErr().getOutputStream()), false);
  }

  /**
   * Dumps all tasks in the requested format.
   */
  private void dumpProfile(ProfileInfo info, PrintStream out, String dumpMode) {
    if (dumpMode.contains("raw")) {
      for (ProfileInfo.Task task : info.allTasksById) {
        dumpRaw(task, out);
      }
    } else {
      for (ProfileInfo.Task task : info.rootTasksById) {
        dumpTask(task, out, 0);
      }
    }
  }

  /**
   * Dumps the task information and all subtasks.
   */
  private void dumpTask(ProfileInfo.Task task, PrintStream out, int indent) {
    StringBuilder builder =
        new StringBuilder(
            String.format(
                Joiner.on('\n')
                    .join(
                        "",
                        "%s %s",
                        "Thread: %-6d  Id: %-6d  Parent: %d",
                        "Start time: %-12s   Duration: %s"),
                task.type,
                task.getDescription(),
                task.threadId,
                task.id,
                task.parentId,
                TimeUtilities.prettyTime(task.startTime),
                TimeUtilities.prettyTime(task.durationNanos)));
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
        Joiner.on('|')
            .join(
                task.threadId,
                task.id,
                task.parentId,
                task.startTime,
                task.durationNanos,
                aggregateString.toString().trim(),
                task.type,
                task.getDescription()));
  }
}
