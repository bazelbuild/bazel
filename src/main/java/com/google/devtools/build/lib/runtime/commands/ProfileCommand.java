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
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.profiler.JsonProfile;
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
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
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

  /**
   * Note that this is just a basic check whether the file is zlib compressed.
   *
   * <p>Other checks (e.g. the magic bytes of the binary profile file) are done later.
   */
  private static boolean isOldBinaryProfile(File profile) {
    try (InputStream inputStream = new FileInputStream(profile)) {
      byte[] magicBytes = new byte[2];
      int numBytesRead = inputStream.read(magicBytes);
      if (numBytesRead == 2
          && magicBytes[0] == (byte) 0x78
          && (magicBytes[1] == (byte) 0x01
              || magicBytes[1] == (byte) 0x9C
              || magicBytes[1] == (byte) 0xDA)) {
        return true;
      }
    } catch (Exception e) {
      // silently ignore any exception
    }
    return false;
  }

  @Override
  public BlazeCommandResult exec(final CommandEnvironment env, OptionsParsingResult options) {
    ProfileOptions profileOptions = options.getOptions(ProfileOptions.class);
    String dumpMode = profileOptions.dumpMode;

    try (PrintStream out = getOutputStream(env)) {
      Reporter reporter = env.getReporter();
      InfoListener infoListener = getInfoListener(env);

      reporter.handle(
          Event.warn(
              "This information is intended for consumption by Bazel developers"
                  + " only, and may change at any time. Script against it at your own risk"));
      for (String name : options.getResidue()) {
        Path profileFile = env.getWorkingDirectory().getRelative(name);
        if (isOldBinaryProfile(profileFile.getPathFile())) {
          BlazeCommandResult commandResult =
              handleOldBinaryProfile(env, dumpMode, out, profileFile, infoListener);
          if (commandResult != null) {
            return commandResult;
          }
          reporter.handle(
              Event.warn(
                  "The old binary profile format is deprecated."
                      + " Use the JSON trace profile instead."));
        } else {
          try {
            if (dumpMode != null) {
              reporter.handle(
                  Event.warn(
                      "--dump has not been implemented yet for the JSON profile, ignoring."));
            }
            JsonProfile jsonProfile = new JsonProfile(profileFile.getPathFile());

            JsonProfile.BuildMetadata buildMetadata = jsonProfile.getBuildMetadata();
            if (buildMetadata != null) {
              infoListener.info(
                  "Profile created on "
                      + buildMetadata.date()
                      + ", build ID: "
                      + buildMetadata.buildId()
                      + ", output base: "
                      + buildMetadata.outputBase());
            }

            new PhaseText(
                    out,
                    jsonProfile.getPhaseSummaryStatistics(),
                    /* phaseStatistics= */ Optional.absent(),
                    Optional.of(new CriticalPathStatistics(jsonProfile.getTraceEvents())))
                .print();
          } catch (IOException e) {
            reporter.handle(Event.error("Failed to analyze profile file(s): " + e.getMessage()));
            return BlazeCommandResult.exitCode(ExitCode.PARSING_FAILURE);
          }
        }
      }
    }
    return BlazeCommandResult.exitCode(ExitCode.SUCCESS);
  }

  private static BlazeCommandResult handleOldBinaryProfile(
      CommandEnvironment env,
      String dumpMode,
      PrintStream out,
      Path profileFile,
      InfoListener infoListener) {
    try {
      ProfileInfo info = ProfileInfo.loadProfileVerbosely(profileFile, infoListener);

      if (dumpMode != null) {
        dumpProfile(info, out, dumpMode);
        return null;
      }

      ProfileInfo.aggregateProfile(info, infoListener);
      PhaseSummaryStatistics phaseSummaryStatistics = new PhaseSummaryStatistics(info);
      EnumMap<ProfilePhase, PhaseStatistics> phaseStatistics = new EnumMap<>(ProfilePhase.class);

      for (ProfilePhase phase : ProfilePhase.values()) {
        phaseStatistics.put(phase, new PhaseStatistics(phase, info));
      }

      new PhaseText(
              out,
              phaseSummaryStatistics,
              Optional.of(phaseStatistics),
              Optional.of(new CriticalPathStatistics(info)))
          .print();
    } catch (IOException e) {
      System.out.println(e);
      env.getReporter().handle(Event.error("Failed to analyze profile file(s): " + e.getMessage()));
      return BlazeCommandResult.exitCode(ExitCode.PARSING_FAILURE);
    }
    return null;
  }

  private static PrintStream getOutputStream(CommandEnvironment env) {
    return new PrintStream(
        new BufferedOutputStream(env.getReporter().getOutErr().getOutputStream()), false);
  }

  /** Dumps all tasks in the requested format. */
  private static void dumpProfile(ProfileInfo info, PrintStream out, String dumpMode) {
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

  /** Dumps the task information and all subtasks. */
  private static void dumpTask(ProfileInfo.Task task, PrintStream out, int indent) {
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

  private static void dumpRaw(ProfileInfo.Task task, PrintStream out) {
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
