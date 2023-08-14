// Copyright 2023 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import java.time.Duration;
import javax.annotation.Nullable;
import one.profiler.AsyncProfiler;

/** Bazel module to record pprof-compatible profiles for single invocations. */
public class CommandProfilerModule extends BlazeModule {

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private static final Duration PROFILING_INTERVAL = Duration.ofMillis(10);

  /** CommandProfilerModule options. */
  public static final class Options extends OptionsBase {
    @Option(
        name = "experimental_command_profile",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.LOGGING,
        effectTags = {OptionEffectTag.UNKNOWN},
        help =
            "Records a Java Flight Recorder CPU profile into a profile.jfr file in the output base"
                + " directory. The syntax and semantics of this flag might change in the future to"
                + " support different profile types or output formats; use at your own risk.")
    public boolean captureCommandProfile;
  }

  @Override
  public Iterable<Class<? extends OptionsBase>> getCommandOptions(Command command) {
    return ImmutableList.of(Options.class);
  }

  private boolean captureCommandProfile;
  @Nullable private Reporter reporter;
  @Nullable private Path outputBase;
  @Nullable private Path outputPath;

  @Override
  public void beforeCommand(CommandEnvironment env) {
    Options options = env.getOptions().getOptions(Options.class);
    captureCommandProfile = options.captureCommandProfile;
    outputBase = env.getBlazeWorkspace().getOutputBase();
    reporter = env.getReporter();

    if (!captureCommandProfile) {
      // Early exit so we don't attempt to load the JNI unless necessary.
      return;
    }

    AsyncProfiler profiler = getProfiler();
    if (profiler == null) {
      return;
    }

    outputPath = outputBase.getRelative("profile.jfr");

    try {
      profiler.execute(getProfilerCommand(outputPath));
    } catch (Exception e) {
      // This may occur if the user has insufficient privileges to capture performance events.
      reporter.handle(Event.error("Starting JFR CPU profile failed: " + e));
      captureCommandProfile = false;
    }

    if (captureCommandProfile) {
      reporter.handle(Event.info("Writing JFR CPU profile to " + outputPath));
    }
  }

  @Override
  public void afterCommand() {
    if (!captureCommandProfile) {
      // Early exit so we don't attempt to load the JNI unless necessary.
      return;
    }

    AsyncProfiler profiler = getProfiler();
    if (profiler == null) {
      return;
    }

    profiler.stop();

    captureCommandProfile = false;
    outputBase = null;
    reporter = null;
    outputPath = null;
  }

  @Nullable
  private static AsyncProfiler getProfiler() {
    try {
      return AsyncProfiler.getInstance();
    } catch (Throwable t) {
      // Loading the JNI must be allowed to fail, as we might be running on an unsupported platform.
      logger.atWarning().withCause(t).log("Failed to load async_profiler JNI");
    }
    return null;
  }

  private static String getProfilerCommand(Path outputPath) {
    // See https://github.com/async-profiler/async-profiler/blob/master/src/arguments.cpp.
    return String.format(
        "start,event=cpu,interval=%s,file=%s,jfr", PROFILING_INTERVAL.toNanos(), outputPath);
  }
}
