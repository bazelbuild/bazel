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
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.buildtool.BuildResult;
import com.google.devtools.build.lib.buildtool.BuildTool;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.rules.android.WriteAdbArgsAction;
import com.google.devtools.build.lib.rules.android.WriteAdbArgsAction.StartType;
import com.google.devtools.build.lib.runtime.BlazeCommand;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.CommonCommandOptions;
import com.google.devtools.build.lib.runtime.commands.BuildCommand;
import com.google.devtools.build.lib.runtime.commands.ProjectFileSupport;
import com.google.devtools.build.lib.shell.AbnormalTerminationException;
import com.google.devtools.build.lib.shell.BadExitStatusException;
import com.google.devtools.build.lib.shell.CommandException;
import com.google.devtools.build.lib.util.CommandBuilder;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.common.options.EnumConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionPriority;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.devtools.common.options.OptionsProvider;
import com.google.devtools.common.options.proto.OptionFilters.OptionEffectTag;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/** Implementation of the 'mobile-install' command. */
@Command(
  name = "mobile-install",
  builds = true,
  options = {MobileInstallCommand.Options.class, WriteAdbArgsAction.Options.class},
  inherits = {BuildCommand.class},
  shortDescription = "Installs targets to mobile devices.",
  completion = "label",
  allowResidue = true,
  help = "resource:mobile-install.txt"
)
public class MobileInstallCommand implements BlazeCommand {

  /**
   * An enumeration of all the modes that mobile-install supports.
   */
  public enum Mode {
    CLASSIC("classic", null),
    SKYLARK("skylark", "MIASPECT"),
    SKYLARK_INCREMENTAL_RES("skylark_incremental_res", "MIRESASPECT");

    private final String mode;
    private final String aspectName;

    private Mode(String mode, String aspectName) {
      this.mode = mode;
      this.aspectName = aspectName;
    }

    public String getAspectName() {
      return aspectName;
    }

    @Override
    public String toString() {
      return mode;
    }
  }

  /**
   * Converter for the --mode option.
   */
  public static class ModeConverter extends EnumConverter<Mode> {
    public ModeConverter() {
      super(Mode.class, "mode");
    }
  }

  /**
   * Command line options for the 'mobile-install' command.
   */
  public static final class Options extends OptionsBase {
    @Option(
      name = "split_apks",
      defaultValue = "false",
      category = "mobile-install",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "Whether to use split apks to install and update the "
              + "application on the device. Works only with devices with "
              + "Marshmallow or later"
    )
    public boolean splitApks;

    @Option(
      name = "incremental",
      category = "mobile-install",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "Whether to do an incremental install. If true, try to avoid unnecessary additional "
              + "work by reading the state of the device the code is to be installed on and using "
              + "that information to avoid unnecessary work. If false (the default), always do a "
              + "full install."
    )
    public boolean incremental;

    @Option(
      name = "mode",
      category = "mobile-install",
      defaultValue = "classic",
      converter = ModeConverter.class,
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "Select how to run mobile-install. \"classic\" runs the current version of "
              + "mobile-install. \"skylark\" uses the new skylark version, which has support for "
              + "android_test. \"skylark_incremental_res\" is the same as \"skylark\" plus "
              + "incremental resource processing. \"skylark_incremental_res\" requires a device "
              + "with root access."
    )
    public Mode mode;

    @Option(
      name = "mobile_install_aspect",
      category = "mobile-install",
      defaultValue = "@android_test_support//tools/android/mobile_install:mobile-install.bzl",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "The aspect to use for mobile-install."
    )
    public String mobileInstallAspect;
  }

  private static final String SINGLE_TARGET_MESSAGE =
      "Can only run a single target. Do not use wildcards that match more than one target";
  private static final String NO_TARGET_MESSAGE = "No targets found to run";

  @Override
  public ExitCode exec(CommandEnvironment env, OptionsProvider options) {
    Options mobileInstallOptions = options.getOptions(Options.class);
    WriteAdbArgsAction.Options adbOptions = options.getOptions(WriteAdbArgsAction.Options.class);

    if (mobileInstallOptions.mode == Mode.CLASSIC) {
      if (adbOptions.start == StartType.WARM && !mobileInstallOptions.incremental) {
        env.getReporter().handle(Event.warn(
           "Warm start is enabled, but will have no effect on a non-incremental build"));
      }
      List<String> targets =
          ProjectFileSupport.getTargets(env.getRuntime().getProjectFileProvider(), options);
      BuildRequest request =
          BuildRequest.create(
              this.getClass().getAnnotation(Command.class).name(),
              options,
              env.getRuntime().getStartupOptionsProvider(),
              targets,
              env.getReporter().getOutErr(),
              env.getCommandId(),
              env.getCommandStartTime());
      return new BuildTool(env).processRequest(request, null).getExitCondition();
    }

    // This list should look like: ["//executable:target", "arg1", "arg2"]
    List<String> targetAndArgs = options.getResidue();

    // The user must at least specify an executable target.
    if (targetAndArgs.isEmpty()) {
      env.getReporter().handle(Event.error("Must specify a target to run"));
      return ExitCode.COMMAND_LINE_ERROR;
    }

    List<String> targets = ImmutableList.of(targetAndArgs.get(0));
    List<String> runTargetArgs = targetAndArgs.subList(1, targetAndArgs.size());

    OutErr outErr = env.getReporter().getOutErr();
    BuildRequest request =
        BuildRequest.create(
            this.getClass().getAnnotation(Command.class).name(),
            options,
            env.getRuntime().getStartupOptionsProvider(),
            targets,
            outErr,
            env.getCommandId(),
            env.getCommandStartTime());
    BuildResult result = new BuildTool(env).processRequest(request, null);

    if (!result.getSuccess()) {
      env.getReporter().handle(Event.error("Build failed. Not running target"));
      return result.getExitCondition();
    }

    Collection<ConfiguredTarget> targetsBuilt = result.getSuccessfulTargets();
    if (targetsBuilt == null) {
      env.getReporter().handle(Event.error(NO_TARGET_MESSAGE));
      return ExitCode.COMMAND_LINE_ERROR;
    }
    if (targetsBuilt.size() != 1) {
      env.getReporter().handle(Event.error(SINGLE_TARGET_MESSAGE));
      return ExitCode.COMMAND_LINE_ERROR;
    }
    ConfiguredTarget targetToRun = Iterables.getOnlyElement(targetsBuilt);

    List<String> cmdLine = new ArrayList<>();
    // TODO(bazel-team): Get the executable path from the filesToRun provider from the aspect.
    cmdLine.add(
        targetToRun.getConfiguration().getBinFragment().getPathString()
            + "/"
            + targetToRun.getLabel().toPathFragment().getPathString()
            + "_launcher");
    cmdLine.addAll(runTargetArgs);

    // Make mobile-install v2 understand relevant v1 flags for ASwB compatibility.
    CommonCommandOptions commonCommandOptions = options.getOptions(CommonCommandOptions.class);
    if (commonCommandOptions != null && !"".equals(commonCommandOptions.toolTag)) {
      cmdLine.add("--tool_tag");
      cmdLine.add(commonCommandOptions.toolTag);
    }
    cmdLine.add("--start_type");
    cmdLine.add(adbOptions.start.toString());
    if (!"".equals(adbOptions.adb)) {
      cmdLine.add("--adb_path");
      cmdLine.add(adbOptions.adb);
    }
    for (String adbArg : adbOptions.adbArgs) {
      cmdLine.add("--adb_arg");
      cmdLine.add(adbArg);
    }

    Path workingDir = env.getBlazeWorkspace().getOutputPath().getParentDirectory();
    com.google.devtools.build.lib.shell.Command command =
        new CommandBuilder()
            .addArgs(cmdLine)
            .setEnv(env.getClientEnv())
            .setWorkingDir(workingDir)
            .build();

    try {
      // Restore a raw EventHandler if it is registered. This allows for blaze run to produce the
      // actual output of the command being run even if --color=no is specified.
      env.getReporter().switchToAnsiAllowingHandler();

      // The command API is a little strange in that the following statement
      // will return normally only if the program exits with exit code 0.
      // If it ends with any other code, we have to catch BadExitStatusException.
      command
          .execute(
              com.google.devtools.build.lib.shell.Command.NO_INPUT,
              com.google.devtools.build.lib.shell.Command.NO_OBSERVER,
              outErr.getOutputStream(),
              outErr.getErrorStream(),
              true /* interruptible */)
          .getTerminationStatus()
          .getExitCode();
      return ExitCode.SUCCESS;
    } catch (BadExitStatusException e) {
      String message =
          "Non-zero return code '"
              + e.getResult().getTerminationStatus().getExitCode()
              + "' from command: "
              + e.getMessage();
      env.getReporter().handle(Event.error(message));
      return ExitCode.RUN_FAILURE;
    } catch (AbnormalTerminationException e) {
      // The process was likely terminated by a signal in this case.
      return ExitCode.INTERRUPTED;
    } catch (CommandException e) {
      env.getReporter().handle(Event.error("Error running program: " + e.getMessage()));
      return ExitCode.RUN_FAILURE;
    }
  }

  @Override
  public void editOptions(OptionsParser optionsParser) {
    Options options = optionsParser.getOptions(Options.class);
    try {
      if (options.mode == Mode.CLASSIC) {
        String outputGroup =
            options.splitApks
                ? "mobile_install_split" + INTERNAL_SUFFIX
                : options.incremental
                    ? "mobile_install_incremental" + INTERNAL_SUFFIX
                    : "mobile_install_full" + INTERNAL_SUFFIX;
        optionsParser.parse(
            OptionPriority.COMMAND_LINE,
            "Options required by the mobile-install command",
            ImmutableList.of("--output_groups=" + outputGroup));
      } else {
        optionsParser.parse(
            OptionPriority.COMMAND_LINE,
            "Options required by the skylark implementation of mobile-install command",
            ImmutableList.of(
                "--aspects=" + options.mobileInstallAspect + "%" + options.mode.getAspectName(),
                "--output_groups=android_incremental_deploy_info",
                "--output_groups=mobile_install" + INTERNAL_SUFFIX,
                "--output_groups=mobile_install_launcher" + INTERNAL_SUFFIX));
      }
    } catch (OptionsParsingException e) {
      throw new IllegalStateException(e);
    }
  }
}
