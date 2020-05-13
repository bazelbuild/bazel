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

import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.profiler.JsonProfile;
import com.google.devtools.build.lib.profiler.output.PhaseText;
import com.google.devtools.build.lib.profiler.statistics.CriticalPathStatistics;
import com.google.devtools.build.lib.runtime.BlazeCommand;
import com.google.devtools.build.lib.runtime.BlazeCommandResult;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.util.ExitCode;
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

      reporter.handle(
          Event.warn(
              "This information is intended for consumption by Bazel developers"
                  + " only, and may change at any time. Script against it at your own risk"));
      for (String name : options.getResidue()) {
        Path profileFile = env.getWorkingDirectory().getRelative(name);
        if (isOldBinaryProfile(profileFile.getPathFile())) {
          reporter.handle(
              Event.error(
                  "The old binary profile format is deprecated."
                      + " Use the JSON trace profile instead."));
          return BlazeCommandResult.exitCode(ExitCode.PARSING_FAILURE);
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
              reporter.handle(
                  Event.info(
                      "Profile created on "
                          + buildMetadata.date()
                          + ", build ID: "
                          + buildMetadata.buildId()
                          + ", output base: "
                          + buildMetadata.outputBase()));
            }

            new PhaseText(
                    out,
                    jsonProfile.getPhaseSummaryStatistics(),
                    new CriticalPathStatistics(jsonProfile.getTraceEvents()))
                .print();
          } catch (IOException e) {
            reporter.handle(Event.error("Failed to analyze profile file(s): " + e.getMessage()));
            return BlazeCommandResult.exitCode(ExitCode.PARSING_FAILURE);
          }
        }
      }
    }
    return BlazeCommandResult.success();
  }

  private static PrintStream getOutputStream(CommandEnvironment env) {
    return new PrintStream(
        new BufferedOutputStream(env.getReporter().getOutErr().getOutputStream()), false);
  }
}
