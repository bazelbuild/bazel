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

import static com.google.devtools.build.lib.analysis.OutputGroupInfo.INTERNAL_SUFFIX;

import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.configuredtargets.AbstractConfiguredTarget;
import com.google.devtools.build.lib.analysis.test.TestConfiguration.TestOptions;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.buildtool.BuildResult;
import com.google.devtools.build.lib.buildtool.BuildTool;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.rules.android.WriteAdbArgsAction;
import com.google.devtools.build.lib.rules.android.WriteAdbArgsAction.StartType;
import com.google.devtools.build.lib.runtime.BlazeCommand;
import com.google.devtools.build.lib.runtime.BlazeCommandResult;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.CommonCommandOptions;
import com.google.devtools.build.lib.runtime.ProjectFileSupport;
import com.google.devtools.build.lib.runtime.commands.BuildCommand;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.Interrupted;
import com.google.devtools.build.lib.server.FailureDetails.MobileInstall;
import com.google.devtools.build.lib.server.FailureDetails.MobileInstall.Code;
import com.google.devtools.build.lib.shell.BadExitStatusException;
import com.google.devtools.build.lib.shell.CommandException;
import com.google.devtools.build.lib.util.CommandBuilder;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.InterruptedFailureDetails;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.common.options.Converters;
import com.google.devtools.common.options.EnumConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;
import com.google.devtools.common.options.OptionPriority.PriorityCategory;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.devtools.common.options.OptionsParsingResult;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import javax.annotation.Nullable;

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

  /** An enumeration of all the modes that mobile-install supports. */
  public enum Mode {
    CLASSIC,
    CLASSIC_INTERNAL_TEST_DO_NOT_USE,
    SKYLARK
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
      documentationCategory = OptionDocumentationCategory.OUTPUT_SELECTION,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS, OptionEffectTag.AFFECTS_OUTPUTS},
      help =
          "Whether to use split apks to install and update the "
              + "application on the device. Works only with devices with "
              + "Marshmallow or later"
    )
    public boolean splitApks;

    @Option(
      name = "incremental",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.OUTPUT_SELECTION,
      effectTags = OptionEffectTag.LOADING_AND_ANALYSIS,
      help =
          "Whether to do an incremental install. If true, try to avoid unnecessary additional "
              + "work by reading the state of the device the code is to be installed on and using "
              + "that information to avoid unnecessary work. If false (the default), always do a "
              + "full install."
    )
    public boolean incremental;

    @Option(
        name = "mode",
        defaultValue = "classic",
        converter = ModeConverter.class,
        documentationCategory = OptionDocumentationCategory.EXECUTION_STRATEGY,
        effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS, OptionEffectTag.EXECUTION},
        metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
        help =
            "Select how to run mobile-install. \"classic\" runs the current version of"
                + " mobile-install. \"skylark\" uses the new Starlark version, which has support"
                + " for android_test.")
    public Mode mode;

    @Option(
      name = "mobile_install_aspect",
      defaultValue = "@android_test_support//tools/android/mobile_install:mobile-install.bzl",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS, OptionEffectTag.CHANGES_INPUTS},
      help = "The aspect to use for mobile-install."
    )
    public String mobileInstallAspect;

    @Option(
        name = "mobile_install_supported_rules",
        defaultValue = "",
        converter = Converters.CommaSeparatedOptionListConverter.class,
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
        help = "The supported rules for mobile-install.")
    public List<String> mobileInstallSupportedRules;
  }

  private static final String SINGLE_TARGET_MESSAGE =
      "Can only run a single target. Do not use wildcards that match more than one target";
  private static final String NO_TARGET_MESSAGE = "No targets found to run";

  @Override
  public BlazeCommandResult exec(CommandEnvironment env, OptionsParsingResult options) {
    Options mobileInstallOptions = options.getOptions(Options.class);
    WriteAdbArgsAction.Options adbOptions = options.getOptions(WriteAdbArgsAction.Options.class);

    if (mobileInstallOptions.mode == Mode.CLASSIC
        || mobileInstallOptions.mode == Mode.CLASSIC_INTERNAL_TEST_DO_NOT_USE) {
      // Notify internal users that classic mode is no longer supported.
      if (mobileInstallOptions.mode == Mode.CLASSIC
          && !mobileInstallOptions.mobileInstallAspect.startsWith("@")) {
        String message = "mobile-install --mode=classic is no longer supported";
        env.getReporter().handle(Event.error(message));
        return createFailureResult(message, Code.CLASSIC_UNSUPPORTED);
      }
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
      DetailedExitCode detailedExitCode =
          new BuildTool(env).processRequest(request, null).getDetailedExitCode();
      return BlazeCommandResult.detailedExitCode(detailedExitCode);
    }

    // This list should look like: ["//executable:target", "arg1", "arg2"]
    List<String> targetAndArgs = options.getResidue();

    // The user must at least specify an executable target.
    if (targetAndArgs.isEmpty()) {
      String message = "Must specify a target to run";
      env.getReporter().handle(Event.error(message));
      return createFailureResult(message, Code.NO_TARGET_SPECIFIED);
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
      return BlazeCommandResult.detailedExitCode(result.getDetailedExitCode());
    }

    Collection<ConfiguredTarget> targetsBuilt = result.getSuccessfulTargets();
    if (targetsBuilt == null) {
      env.getReporter().handle(Event.warn(NO_TARGET_MESSAGE));
      return BlazeCommandResult.success();
    }
    if (targetsBuilt.size() != 1) {
      env.getReporter().handle(Event.error(SINGLE_TARGET_MESSAGE));
      return createFailureResult(SINGLE_TARGET_MESSAGE, Code.MULTIPLE_TARGETS_SPECIFIED);
    }
    ConfiguredTarget targetToRun = Iterables.getOnlyElement(targetsBuilt);

    if (!mobileInstallOptions.mobileInstallSupportedRules.isEmpty()) {
      String message =
          errorMessageIfNotSupported(targetToRun, mobileInstallOptions.mobileInstallSupportedRules);
      if (message != null) {
        env.getReporter().handle(Event.error(message));
        return createFailureResult(message, Code.TARGET_TYPE_INVALID);
      }
    }

    List<String> cmdLine = new ArrayList<>();
    // TODO(bazel-team): Get the executable path from the filesToRun provider from the aspect.
    BuildConfiguration configuration =
        env.getSkyframeExecutor()
            .getConfiguration(env.getReporter(), targetToRun.getConfigurationKey());
    cmdLine.add(
        configuration.getBinFragment(targetToRun.getLabel().getRepository()).getPathString()
            + "/"
            + targetToRun.getLabel().toPathFragment().getPathString()
            + "_mi/launcher");
    cmdLine.addAll(runTargetArgs);

    cmdLine.add("--build_id=" + env.getCommandId());

    // Collect relevant common command options
    CommonCommandOptions commonCommandOptions = options.getOptions(CommonCommandOptions.class);
    if (!commonCommandOptions.toolTag.isEmpty()) {
      cmdLine.add("--tool_tag=" + commonCommandOptions.toolTag);
    }

    // Collect relevant adb options
    cmdLine.add("--start=" + adbOptions.start);
    if (!adbOptions.adb.isEmpty()) {
      cmdLine.add("--adb=" + adbOptions.adb);
    }
    for (String adbArg : adbOptions.adbArgs) {
      if (!adbArg.isEmpty()) {
        cmdLine.add("--adb_arg=" + adbArg);
      }
    }
    if (!adbOptions.device.isEmpty()) {
      cmdLine.add("--device=" + adbOptions.device);
    }

    // Collect relevant test options
    TestOptions testOptions = options.getOptions(TestOptions.class);
    // Default value of testFilter is null.
    if (!Strings.isNullOrEmpty(testOptions.testFilter)){
      cmdLine.add("--test_filter=" + testOptions.testFilter);
    }
    for (String arg : testOptions.testArguments) {
      if (!arg.isEmpty()) {
        cmdLine.add("--test_arg=" + arg);
      }
    }

    Path workingDir =
        env.getDirectories().getOutputPath(env.getWorkspaceName()).getParentDirectory();
    com.google.devtools.build.lib.shell.Command command =
        new CommandBuilder()
            .addArgs(cmdLine)
            .setEnv(env.getClientEnv())
            .setWorkingDir(workingDir)
            .build();

    try (SilentCloseable c = Profiler.instance().profile("mobile install")) {
      // Restore a raw EventHandler if it is registered. This allows for blaze run to produce the
      // actual output of the command being run even if --color=no is specified.
      env.getReporter().switchToAnsiAllowingHandler();

      // The command API is a little strange in that the following statement will return normally
      // only if the program exits with exit code 0. If it ends with any other code, we have to
      // catch BadExitStatusException.
      command
          .execute(outErr.getOutputStream(), outErr.getErrorStream())
          .getTerminationStatus()
          .getExitCode();
      return BlazeCommandResult.success();
    } catch (BadExitStatusException e) {
      String message =
          "Non-zero return code '"
              + e.getResult().getTerminationStatus().getExitCode()
              + "' from command: "
              + e.getMessage();
      env.getReporter().handle(Event.error(message));
      return createFailureResult(message, Code.NON_ZERO_EXIT);
    } catch (CommandException e) {
      String message = "Error running program: " + e.getMessage();
      env.getReporter().handle(Event.error(message));
      return createFailureResult(message, Code.ERROR_RUNNING_PROGRAM);
    } catch (InterruptedException e) {
      return BlazeCommandResult.detailedExitCode(
          InterruptedFailureDetails.detailedExitCode(
              "mobile install interrupted", Interrupted.Code.MOBILE_INSTALL_COMMAND));
    }
  }

  @Override
  public void editOptions(OptionsParser optionsParser) {
    Options options = optionsParser.getOptions(Options.class);
    try {
      if (options.mode == Mode.CLASSIC || options.mode == Mode.CLASSIC_INTERNAL_TEST_DO_NOT_USE) {
        String outputGroup =
            options.splitApks
                ? "mobile_install_split" + INTERNAL_SUFFIX
                : options.incremental
                    ? "mobile_install_incremental" + INTERNAL_SUFFIX
                    : "mobile_install_full" + INTERNAL_SUFFIX;
        optionsParser.parse(
            PriorityCategory.COMMAND_LINE,
            "Options required by the mobile-install command",
            ImmutableList.of("--output_groups=" + outputGroup));
      } else {
        optionsParser.parse(
            PriorityCategory.COMMAND_LINE,
            "Options required by the Starlark implementation of mobile-install command",
            ImmutableList.of(
                "--aspects=" + options.mobileInstallAspect + "%MIASPECT",
                "--output_groups=mobile_install" + INTERNAL_SUFFIX,
                "--output_groups=mobile_install_launcher" + INTERNAL_SUFFIX));
      }
    } catch (OptionsParsingException e) {
      throw new IllegalStateException(e);
    }
  }

  @Nullable
  private static String errorMessageIfNotSupported(
      ConfiguredTarget target, List<String> mobileInstallSupportedRules) {
    // Dereference any aliases that might be present.
    target = target.getActual();

    if (target instanceof AbstractConfiguredTarget) {
      String ruleType = ((AbstractConfiguredTarget) target).getRuleClassString();
      if (!mobileInstallSupportedRules.contains(ruleType)) {
        return String.format(
            "mobile-install can only be run on %s targets. Got: %s",
            mobileInstallSupportedRules, ruleType);
      } else {
        return null;
      }
    }
    return "Invalid target";
  }

  private static BlazeCommandResult createFailureResult(String message, Code detailedCode) {
    return BlazeCommandResult.failureDetail(
        FailureDetail.newBuilder()
            .setMessage(message)
            .setMobileInstall(MobileInstall.newBuilder().setCode(detailedCode))
            .build());
  }
}
