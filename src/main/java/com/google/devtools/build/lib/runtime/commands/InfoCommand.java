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

import com.google.common.base.Strings;
import com.google.common.base.Supplier;
import com.google.common.base.Suppliers;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.NoBuildEvent;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.runtime.BlazeCommand;
import com.google.devtools.build.lib.runtime.BlazeCommandResult;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.server.FailureDetails.BuildConfiguration.Code;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.Interrupted;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.InterruptedFailureDetails;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParsingResult;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * Implementation of 'blaze info'.
 */
@Command(name = "info",
         // TODO(bazel-team): this is not really a build command, but needs access to the
         // configuration options to do its job
         builds = true,
         allowResidue = true,
         binaryStdOut = true,
         help = "resource:info.txt",
         shortDescription = "Displays runtime info about the %{product} server.",
         options = { InfoCommand.Options.class },
         completion = "info-key",
         // We have InfoCommand inherit from {@link BuildCommand} because we want all
         // configuration defaults specified in ~/.blazerc for {@code build} to apply to
         // {@code info} too, even though it doesn't actually do a build.
         //
         // (Ideally there would be a way to make {@code info} inherit just the bare
         // minimum of relevant options from {@code build}, i.e. those that affect the
         // values it prints.  But there's no such mechanism.)
         inherits = { BuildCommand.class })
public class InfoCommand implements BlazeCommand {

  /** Options for the info command. */
  public static class Options extends OptionsBase {
    @Option(
      name = "show_make_env",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.LOGGING,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.TERMINAL_OUTPUT},
      help = "Include the \"Make\" environment in the output."
    )
    public boolean showMakeEnvironment;

    @Option(
        name = "experimental_supports_info_crosstool_configuration",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.UNKNOWN},
        metadataTags = {OptionMetadataTag.HIDDEN},
        help = "Noop.")
    public boolean experimentalSupportsInfoCrosstoolConfiguration;
  }

  /**
   * Unchecked variant of {@link AbruptExitException}. Below, we need to throw from the Supplier
   * interface, which does not allow checked exceptions.
   */
  private static class AbruptExitRuntimeException extends RuntimeException {

    private final DetailedExitCode detailedExitCode;

    private AbruptExitRuntimeException(DetailedExitCode exitCode) {
      this.detailedExitCode = exitCode;
    }

    private DetailedExitCode getDetailedExitCode() {
      return detailedExitCode;
    }
  }

  @Override
  public BlazeCommandResult exec(
      final CommandEnvironment env, final OptionsParsingResult optionsParsingResult) {
    final BlazeRuntime runtime = env.getRuntime();
    env.getReporter().switchToAnsiAllowingHandler();
    Options infoOptions = optionsParsingResult.getOptions(Options.class);
    OutErr outErr = env.getReporter().getOutErr();
    // Creating a BuildConfiguration is expensive and often unnecessary. Delay the creation until
    // it is needed. We memoize so that it's cached intra-command (it's still created freshly on
    // every command since the configuration can change across commands).
    Supplier<BuildConfiguration> configurationSupplier =
        Suppliers.memoize(
            () -> {
              try {
                // In order to be able to answer configuration-specific queries, we need to set up
                // the package path. Since info inherits all the build options, all the necessary
                // information is available here.
                env.syncPackageLoading(optionsParsingResult);
                // TODO(bazel-team): What if there are multiple configurations? [multi-config]
                return env.getSkyframeExecutor()
                    .getConfiguration(
                        env.getReporter(),
                        runtime.createBuildOptions(optionsParsingResult),
                        /*keepGoing=*/ true);
              } catch (InvalidConfigurationException e) {
                env.getReporter().handle(Event.error(e.getMessage()));
                throw new AbruptExitRuntimeException(
                    DetailedExitCode.of(
                        ExitCode.COMMAND_LINE_ERROR,
                        FailureDetail.newBuilder()
                            .setMessage(Strings.nullToEmpty(e.getMessage()))
                            .setBuildConfiguration(
                                FailureDetails.BuildConfiguration.newBuilder()
                                    .setCode(
                                        e.getDetailedCode() == null
                                            ? Code.BUILD_CONFIGURATION_UNKNOWN
                                            : e.getDetailedCode()))
                            .build()));
              } catch (AbruptExitException e) {
                throw new AbruptExitRuntimeException(e.getDetailedExitCode());
              } catch (InterruptedException e) {
                env.getReporter().handle(Event.error("interrupted"));
                throw new AbruptExitRuntimeException(
                    InterruptedFailureDetails.detailedExitCode(
                        "command interrupted while syncing package loading",
                        Interrupted.Code.PACKAGE_LOADING_SYNC));
              }
            });

    Map<String, InfoItem> items = getInfoItemMap(env, optionsParsingResult);

    try {
      if (infoOptions.showMakeEnvironment) {
        Map<String, String> makeEnv = configurationSupplier.get().getMakeEnvironment();
        for (Map.Entry<String, String> entry : makeEnv.entrySet()) {
          InfoItem item = new InfoItem.MakeInfoItem(entry.getKey(), entry.getValue());
          items.put(item.getName(), item);
        }
      }

      List<String> residue = optionsParsingResult.getResidue();
      if (residue.size() > 1) {
        String message = "at most one key may be specified";
        env.getReporter().handle(Event.error(message));
        return createFailureResult(
            message, ExitCode.COMMAND_LINE_ERROR, FailureDetails.InfoCommand.Code.TOO_MANY_KEYS);
      }

      String key = residue.size() == 1 ? residue.get(0) : null;
      env.getEventBus().post(new NoBuildEvent());
      if (key != null) { // print just the value for the specified key:
        byte[] value;
        if (items.containsKey(key)) {
          value = items.get(key).get(configurationSupplier, env);
        } else {
          String message = "unknown key: '" + key + "'";
          env.getReporter().handle(Event.error(message));
          return createFailureResult(
              message,
              ExitCode.COMMAND_LINE_ERROR,
              FailureDetails.InfoCommand.Code.KEY_NOT_RECOGNIZED);
        }
        try {
          outErr.getOutputStream().write(value);
          outErr.getOutputStream().flush();
        } catch (IOException e) {
          String message = "Cannot write info block: " + e.getMessage();
          env.getReporter().handle(Event.error(message));
          return createFailureResult(
              message,
              ExitCode.ANALYSIS_FAILURE,
              FailureDetails.InfoCommand.Code.INFO_BLOCK_WRITE_FAILURE);
        }
      } else { // print them all
        configurationSupplier.get();  // We'll need this later anyway
        for (InfoItem infoItem : items.values()) {
          if (infoItem.isHidden()) {
            continue;
          }
          outErr.getOutputStream().write(
              (infoItem.getName() + ": ").getBytes(StandardCharsets.UTF_8));
          outErr.getOutputStream().write(infoItem.get(configurationSupplier, env));
        }
      }
    } catch (AbruptExitException e) {
      return BlazeCommandResult.detailedExitCode(e.getDetailedExitCode());
    } catch (AbruptExitRuntimeException e) {
      return BlazeCommandResult.detailedExitCode(e.getDetailedExitCode());
    } catch (IOException e) {
      return createFailureResult(
          "Cannot write info block: " + e.getMessage(),
          ExitCode.LOCAL_ENVIRONMENTAL_ERROR,
          FailureDetails.InfoCommand.Code.ALL_INFO_WRITE_FAILURE);
    } catch (InterruptedException e) {
      return BlazeCommandResult.detailedExitCode(
          InterruptedFailureDetails.detailedExitCode(
              "info interrupted", Interrupted.Code.INFO_ITEM));
    }
    return BlazeCommandResult.success();
  }

  private static BlazeCommandResult createFailureResult(
      String message, ExitCode exitCode, FailureDetails.InfoCommand.Code detailedCode) {
    return BlazeCommandResult.detailedExitCode(
        DetailedExitCode.of(
            exitCode,
            FailureDetail.newBuilder()
                .setMessage(message)
                .setInfoCommand(FailureDetails.InfoCommand.newBuilder().setCode(detailedCode))
                .build()));
  }

  private static Map<String, InfoItem> getHardwiredInfoItemMap(
      OptionsParsingResult commandOptions, String productName) {
    List<InfoItem> hardwiredInfoItems =
        ImmutableList.<InfoItem>of(
            new InfoItem.WorkspaceInfoItem(),
            new InfoItem.InstallBaseInfoItem(),
            new InfoItem.OutputBaseInfoItem(productName),
            new InfoItem.ExecutionRootInfoItem(),
            new InfoItem.OutputPathInfoItem(),
            new InfoItem.ClientEnv(),
            new InfoItem.BlazeBinInfoItem(productName),
            new InfoItem.BlazeGenfilesInfoItem(productName),
            new InfoItem.BlazeTestlogsInfoItem(productName),
            new InfoItem.ReleaseInfoItem(productName),
            new InfoItem.ServerPidInfoItem(productName),
            new InfoItem.ServerLogInfoItem(productName),
            new InfoItem.PackagePathInfoItem(commandOptions),
            new InfoItem.UsedHeapSizeInfoItem(),
            new InfoItem.UsedHeapSizeAfterGcInfoItem(),
            new InfoItem.CommitedHeapSizeInfoItem(),
            new InfoItem.MaxHeapSizeInfoItem(),
            new InfoItem.GcTimeInfoItem(),
            new InfoItem.GcCountInfoItem(),
            new InfoItem.JavaRuntimeInfoItem(),
            new InfoItem.JavaVirtualMachineInfoItem(),
            new InfoItem.JavaHomeInfoItem(),
            new InfoItem.CharacterEncodingInfoItem(),
            new InfoItem.DefaultsPackageInfoItem(),
            new InfoItem.BuildLanguageInfoItem(),
            new InfoItem.DefaultPackagePathInfoItem(commandOptions),
            new InfoItem.StarlarkSemanticsInfoItem(commandOptions));
    ImmutableMap.Builder<String, InfoItem> result = new ImmutableMap.Builder<>();
    for (InfoItem item : hardwiredInfoItems) {
      result.put(item.getName(), item);
    }
    return result.build();
  }

  public static List<String> getHardwiredInfoItemNames(String productName) {
    ImmutableList.Builder<String> result = new ImmutableList.Builder<>();
    for (String name : InfoCommand.getHardwiredInfoItemMap(null, productName).keySet()) {
      result.add(name);
    }
    return result.build();
  }

  static Map<String, InfoItem> getInfoItemMap(
      CommandEnvironment env, OptionsParsingResult optionsParsingResult) {
    Map<String, InfoItem> items = new TreeMap<>(env.getRuntime().getInfoItems());
    items.putAll(getHardwiredInfoItemMap(optionsParsingResult, env.getRuntime().getProductName()));
    return items;
  }
}
