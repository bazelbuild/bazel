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

import com.google.devtools.build.lib.analysis.BlazeVersionInfo;
import com.google.devtools.build.lib.analysis.NoBuildEvent;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.runtime.BlazeCommand;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsProvider;

/**
 * The 'blaze version' command, which informs users about the blaze version
 * information.
 */
@Command(name = "version",
         options = {},
         allowResidue = false,
         mustRunInWorkspace = false,
         help = "resource:version.txt",
         shortDescription = "Prints version information for %{product}.")
public final class VersionCommand implements BlazeCommand {
  @Override
  public void editOptions(OptionsParser optionsParser) {}

  @Override
  public ExitCode exec(CommandEnvironment env, OptionsProvider options) {
    BlazeVersionInfo info = BlazeVersionInfo.instance();
    if (info.getSummary() == null) {
      env.getReporter().handle(Event.error("Version information not available"));
      return ExitCode.COMMAND_LINE_ERROR;
    }
    env.getEventBus().post(new NoBuildEvent());
    env.getReporter().getOutErr().printOutLn(info.getSummary());
    return ExitCode.SUCCESS;
  }
}
