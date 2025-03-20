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
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.common.options.EnumConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import java.time.Duration;
import java.util.Locale;
import javax.annotation.Nullable;
import one.profiler.AsyncProfiler;

/** Bazel module to record a Java Flight Recorder profile for a single command. */
public class CommandProfilerModule extends BlazeModule {

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private static final Duration PROFILING_INTERVAL = Duration.ofMillis(10);

  /** The type of profile to capture. */
  enum ProfileType {
    CPU,
    WALL,
    ALLOC,
    LOCK;

    @Override
    public String toString() {
      return name().toLowerCase(Locale.US);
    }
  }

  /** Options converter for --experimental_command_profile. */
  public static final class ProfileTypeConverter extends EnumConverter<ProfileType> {

    public ProfileTypeConverter() {
      super(ProfileType.class, "--experimental_command_profile setting");
    }
  }

  /** CommandProfilerModule options. */
  public static final class Options extends OptionsBase {

    @Option(
        name = "experimental_command_profile",
        defaultValue = "null",
        converter = ProfileTypeConverter.class,
        documentationCategory = OptionDocumentationCategory.LOGGING,
        effectTags = {OptionEffectTag.UNKNOWN},
        help =
            "Records a Java Flight Recorder profile for the duration of the command. One of the"
                + " supported profiling event types (cpu, wall, alloc or lock) must be given as an"
                + " argument. The profile is written to a file named after the event type under the"
                + " output base directory."
                + " The syntax and semantics of this flag might change in the future to support"
                + " additional profile types or output formats; use at your own risk.")
    public ProfileType profileType;
  }

  @Override
  public Iterable<Class<? extends OptionsBase>> getCommonCommandOptions() {
    return ImmutableList.of(Options.class);
  }

  @Nullable private ProfileType profileType;
  @Nullable private Reporter reporter;
  @Nullable private Path outputBase;
  @Nullable private Path outputPath;

  @Override
  public void beforeCommand(CommandEnvironment env) {
    Options options = env.getOptions().getOptions(Options.class);
    profileType = options.profileType;
    outputBase = env.getBlazeWorkspace().getOutputBase();
    reporter = env.getReporter();

    if (profileType == null) {
      // Early exit so we don't attempt to load the JNI unless necessary.
      return;
    }

    AsyncProfiler profiler = getProfiler();
    if (profiler == null) {
      return;
    }

    outputPath = getProfilerOutputPath(profileType);

    try {
      profiler.execute(getProfilerCommand(profileType, outputPath));
    } catch (Exception e) {
      // This may occur if the user has insufficient privileges to capture performance events.
      reporter.handle(
          Event.error(String.format("Starting JFR %s profile failed: %s", profileType, e)));
      profileType = null;
    }

    if (profileType != null) {
      reporter.handle(
          Event.info(String.format("Writing JFR %s profile to %s", profileType, outputPath)));
    }
  }

  @Override
  public void afterCommand() {
    if (profileType == null) {
      // Early exit so we don't attempt to load the JNI unless necessary.
      return;
    }

    AsyncProfiler profiler = getProfiler();
    if (profiler == null) {
      return;
    }

    profiler.stop();

    profileType = null;
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

  private Path getProfilerOutputPath(ProfileType profileType) {
    return outputBase.getChild(profileType + ".jfr");
  }

  private static String getProfilerCommand(ProfileType profileType, Path outputPath) {
    // See https://github.com/async-profiler/async-profiler/blob/master/src/arguments.cpp.
    return String.format(
        "start,event=%s,interval=%s,file=%s,jfr",
        profileType, PROFILING_INTERVAL.toNanos(), outputPath);
  }
}
