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

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.analysis.BlazeVersionInfo;
import com.google.devtools.build.lib.analysis.NoBuildEvent;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.runtime.BlazeCommand;
import com.google.devtools.build.lib.runtime.BlazeCommandResult;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingResult;
import java.io.IOException;

/**
 * The 'blaze version' command, which informs users about the blaze version
 * information.
 */
@Command(name = "version",
         options = { VersionCommand.VersionOptions.class },
         allowResidue = false,
         mustRunInWorkspace = false,
         help = "resource:version.txt",
         shortDescription = "Prints version information for %{product}.")
public final class VersionCommand implements BlazeCommand {
  /** Options for the "version" command. */
  public static class VersionOptions extends OptionsBase {
    @Option(
      name = "gnu_format",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.EXECUTION},
      help =
          "If set, write the version to stdout using the conventions described in the GNU"
          + " standards."
    )
    public boolean gnuFormat;
  }

  @Override
  public void editOptions(OptionsParser optionsParser) {}

  @Override
  public BlazeCommandResult exec(CommandEnvironment env, OptionsParsingResult options) {
    env.getEventBus().post(new NoBuildEvent());
    try {
      env.getReporter().getOutErr().printOutLn(
          getInfo(
              env.getRuntime().getProductName(),
              BlazeVersionInfo.instance(),
              options.getOptions(VersionOptions.class).gnuFormat));
    } catch (IOException e) {
      env.getReporter().handle(Event.error(e.getMessage()));
      return BlazeCommandResult.exitCode(ExitCode.COMMAND_LINE_ERROR);
    }
    return BlazeCommandResult.success();
  }

  @VisibleForTesting
  static String getInfo(String productName, BlazeVersionInfo info, boolean gnuFormat)
      throws IOException {
    if (info.getSummary() == null) {
      throw new IOException("Version information not available");
    }
    if (gnuFormat) {
      return productName + " " + (info.isReleasedBlaze() ? info.getVersion() : "no_version");
    } else {
      return info.getSummary();
    }
  }
}
