// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.runtime.mobileinstall;

import static com.google.devtools.build.lib.analysis.OutputGroupProvider.INTERNAL_SUFFIX;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.buildtool.BuildTool;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.rules.android.WriteAdbArgsAction;
import com.google.devtools.build.lib.rules.android.WriteAdbArgsAction.StartType;
import com.google.devtools.build.lib.runtime.BlazeCommand;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.commands.BuildCommand;
import com.google.devtools.build.lib.runtime.commands.ProjectFileSupport;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionPriority;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParser.OptionUsageRestrictions;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.devtools.common.options.OptionsProvider;
import java.util.List;

/**
 * Implementation of the 'mobile-install' command.
 */
 @Command(name = "mobile-install",
         builds = true,
         options = { MobileInstallCommand.Options.class, WriteAdbArgsAction.Options.class },
         inherits = { BuildCommand.class },
         shortDescription = "Installs targets to mobile devices.",
         completion = "label",
         allowResidue = true,
         help = "resource:mobile-install.txt")
public class MobileInstallCommand implements BlazeCommand {
  /**
   * Command line options for the 'mobile-install' command.
   */
  public static final class Options extends OptionsBase {
    @Option(name = "split_apks",
        defaultValue = "false",
        category = "mobile-install",
        help = "Whether to use split apks to install and update the "
            + "application on the device. Works only with devices with "
            + "Marshmallow or later")
    public boolean splitApks;

    @Option(name = "incremental",
        category = "mobile-install",
        defaultValue = "false",
        help = "Whether to do an incremental install. If true, try to avoid unnecessary additional "
            + "work by reading the state of the device the code is to be installed on and using "
            + "that information to avoid unnecessary work. If false (the default), always do a "
            + "full install.")
    public boolean incremental;

    @Option(
      name = "v2",
      category = "mobile-install",
      defaultValue = "false",
      help = "Whether to use the v2 mobile-install. If true, rather than using the current "
          + "version of mobile-install, use version 2.",
      optionUsageRestrictions = OptionUsageRestrictions.UNDOCUMENTED
    )
    public boolean v2;

    @Option(
      name = "mobile_install_aspects",
      category = "mobile-install",
      defaultValue =
          "@android_test_support//tools/android/mobile_install:mobile-install.bzl%MIASPECT",
      help = "The aspect to use for mobile-install.",
      optionUsageRestrictions = OptionUsageRestrictions.UNDOCUMENTED
    )
    public String mobileInstallAspects;
  }

  @Override
  public ExitCode exec(CommandEnvironment env, OptionsProvider options) {
    BlazeRuntime runtime = env.getRuntime();
    Options mobileInstallOptions = options.getOptions(Options.class);
    WriteAdbArgsAction.Options adbOptions = options.getOptions(WriteAdbArgsAction.Options.class);
    if (adbOptions.start == StartType.WARM && !mobileInstallOptions.incremental) {
      env.getReporter().handle(Event.warn(
         "Warm start is enabled, but will have no effect on a non-incremental build"));
    }

    List<String> targets = ProjectFileSupport.getTargets(runtime, options);
    BuildRequest request = BuildRequest.create(
        this.getClass().getAnnotation(Command.class).name(), options,
        runtime.getStartupOptionsProvider(), targets,
        env.getReporter().getOutErr(), env.getCommandId(), env.getCommandStartTime());
    return new BuildTool(env).processRequest(request, null).getExitCondition();
  }

  @Override
  public void editOptions(CommandEnvironment env, OptionsParser optionsParser)
      throws AbruptExitException {
    try {
      if (optionsParser.getOptions(Options.class).v2) {
        optionsParser.parse(
            OptionPriority.COMMAND_LINE,
            "Options required by the mobile-install v2 command",
            ImmutableList.of(
                "--aspects=" + optionsParser.getOptions(Options.class).mobileInstallAspects,
                "--output_groups=mobile_install_v2" + INTERNAL_SUFFIX));
      } else {
        String outputGroup =
            optionsParser.getOptions(Options.class).splitApks
                ? "mobile_install_split" + INTERNAL_SUFFIX
                : optionsParser.getOptions(Options.class).incremental
                    ? "mobile_install_incremental" + INTERNAL_SUFFIX
                    : "mobile_install_full" + INTERNAL_SUFFIX;
        optionsParser.parse(
            OptionPriority.COMMAND_LINE,
            "Options required by the mobile-install command",
            ImmutableList.of("--output_groups=" + outputGroup));
      }
    } catch (OptionsParsingException e) {
      throw new IllegalStateException(e);
    }
  }
}
