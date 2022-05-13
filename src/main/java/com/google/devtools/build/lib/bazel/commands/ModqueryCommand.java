// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.bazel.commands;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.bazel.bzlmod.BazelModuleInspectorValue;
import com.google.devtools.build.lib.bazel.bzlmod.BazelModuleResolutionValue;
import com.google.devtools.build.lib.bazel.bzlmod.ModqueryExecutor;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleKey;
import com.google.devtools.build.lib.bazel.bzlmod.Version;
import com.google.devtools.build.lib.bazel.commands.ModqueryOptions.QueryType;
import com.google.devtools.build.lib.bazel.commands.ModqueryOptions.QueryTypeConverter;
import com.google.devtools.build.lib.bazel.commands.ModqueryOptions.TargetModule;
import com.google.devtools.build.lib.bazel.commands.ModqueryOptions.TargetModuleListConverter;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.pkgcache.PackageOptions;
import com.google.devtools.build.lib.runtime.BlazeCommand;
import com.google.devtools.build.lib.runtime.BlazeCommandResult;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.KeepGoingOption;
import com.google.devtools.build.lib.runtime.LoadingPhaseThreadsOption;
import com.google.devtools.build.lib.runtime.UiOptions;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.ModqueryCommand.Code;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.InterruptedFailureDetails;
import com.google.devtools.build.lib.util.io.AnsiTerminalPrinter;
import com.google.devtools.build.skyframe.EvaluationContext;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.devtools.common.options.OptionsParsingResult;
import java.util.List;

/** Queries the Bzlmod external dependency graph. */
@Command(
    name = ModqueryCommand.NAME,
    options = {
      ModqueryOptions.class,
      // Don't know what these do but were used in fetch
      PackageOptions.class,
      KeepGoingOption.class,
      LoadingPhaseThreadsOption.class
    },
    // TODO(andreisolo): figure out which extra options are really needed
    help = "resource:modquery.txt",
    shortDescription = "Queries the Bzlmod external dependency graph",
    allowResidue = true)
public final class ModqueryCommand implements BlazeCommand {

  public static final String NAME = "modquery";

  @Override
  public BlazeCommandResult exec(CommandEnvironment env, OptionsParsingResult options) {
    BazelModuleResolutionValue moduleResolution;
    BazelModuleInspectorValue moduleInspector;

    try {
      // Don't know what it does but was used in fetch
      env.syncPackageLoading(options);

      SkyframeExecutor skyframeExecutor = env.getSkyframeExecutor();
      LoadingPhaseThreadsOption threadsOption = options.getOptions(LoadingPhaseThreadsOption.class);

      EvaluationContext evaluationContext =
          EvaluationContext.newBuilder()
              // Don't know what it does but was used in fetch
              .setNumThreads(threadsOption.threads)
              .setEventHandler(env.getReporter())
              .build();

      moduleResolution =
          (BazelModuleResolutionValue)
              skyframeExecutor
                  .prepareAndGet(ImmutableSet.of(BazelModuleResolutionValue.KEY), evaluationContext)
                  .get(BazelModuleResolutionValue.KEY);

      moduleInspector =
          (BazelModuleInspectorValue)
              skyframeExecutor
                  .prepareAndGet(ImmutableSet.of(BazelModuleInspectorValue.KEY), evaluationContext)
                  .get(BazelModuleInspectorValue.KEY);

      // Don't know what it does but was used in fetch
    } catch (InterruptedException e) {
      String errorMessage = "Modquery interrupted: " + e.getMessage();
      env.getReporter().handle(Event.error(errorMessage));
      return BlazeCommandResult.detailedExitCode(
          InterruptedFailureDetails.detailedExitCode(errorMessage));
      // Don't know what it does but was used in fetch
    } catch (AbruptExitException e) {
      env.getReporter().handle(Event.error(null, "Unknown error: " + e.getMessage()));
      return BlazeCommandResult.detailedExitCode(e.getDetailedExitCode());
    }

    AnsiTerminalPrinter printer =
        new AnsiTerminalPrinter(
            env.getReporter().getOutErr().getOutputStream(),
            options.getOptions(UiOptions.class).useColor());

    ModqueryExecutor modqueryExecutor =
        new ModqueryExecutor(
            moduleResolution.getDepGraph(),
            moduleInspector.getDepGraph(),
            moduleInspector.getModulesIndex(),
            printer);

    ModqueryOptions modqueryOptions = options.getOptions(ModqueryOptions.class);
    Preconditions.checkArgument(modqueryOptions != null);

    if (options.getResidue().isEmpty()) {
      String errorMessage =
          String.format("No query type specified, choose one from : %s.", QueryType.printValues());
      return reportAndCreateFailureResult(env, errorMessage, Code.MODQUERY_COMMAND_UNKNOWN);
    }

    String queryInput = options.getResidue().get(0);
    QueryType query;
    try {
      query = new QueryTypeConverter().convert(queryInput);
    } catch (OptionsParsingException e) {
      String errorMessage =
          String.format("Invalid query type, choose one from : %s.", QueryType.printValues());
      return reportAndCreateFailureResult(env, errorMessage, Code.MODQUERY_COMMAND_UNKNOWN);
    }

    List<String> args = options.getResidue().subList(1, options.getResidue().size());
    if (query.getArgNumber() != args.size()) {
      return reportAndCreateFailureResult(
          env,
          String.format(
              "invalid number of arguments (provided %d, required %d).",
              args.size(), query.getArgNumber()),
          Code.MISSING_ARGUMENTS);
    }

    TargetModuleListConverter converter = new TargetModuleListConverter();
    ImmutableList.Builder<ImmutableSet<ModuleKey>> argsKeysListBuilder =
        new ImmutableList.Builder<>();

    for (String arg : args) {
      try {
        ImmutableList<TargetModule> targetList = converter.convert(arg);
        ImmutableSet<ModuleKey> argModuleKeys =
            targetListToModuleKeySet(targetList, moduleInspector.getModulesIndex());
        argsKeysListBuilder.add(argModuleKeys);
      } catch (OptionsParsingException | InvalidArgumentException e) {
        return reportAndCreateFailureResult(env, e.getMessage(), Code.INVALID_ARGUMENTS);
      }
    }
    ImmutableList<ImmutableSet<ModuleKey>> argsKeysList = argsKeysListBuilder.build();

    ImmutableSet<ModuleKey> fromKeys;
    try {
      fromKeys =
          targetListToModuleKeySet(modqueryOptions.modulesFrom, moduleInspector.getModulesIndex());
    } catch (InvalidArgumentException e) {
      return reportAndCreateFailureResult(env, e.getMessage(), Code.INVALID_ARGUMENTS);
    }

    if (query == QueryType.TREE) {
      modqueryExecutor.tree(fromKeys);
    } else if (query == QueryType.DEPS) {
      modqueryExecutor.deps(argsKeysList.get(0));
    } else if (query == QueryType.PATH) {
      modqueryExecutor.path(fromKeys, argsKeysList.get(0));
    } else if (query == QueryType.ALL_PATHS) {
      modqueryExecutor.allPaths(fromKeys, argsKeysList.get(0));
    } else if (query == QueryType.EXPLAIN) {
      modqueryExecutor.explain(argsKeysList.get(0));
    } else if (query == QueryType.SHOW) {
      modqueryExecutor.show(argsKeysList.get(0));
    }

    return BlazeCommandResult.success();
  }

  private static ImmutableSet<ModuleKey> targetListToModuleKeySet(
      ImmutableList<TargetModule> targetList,
      ImmutableMap<String, ImmutableSet<ModuleKey>> modulesIndex)
      throws InvalidArgumentException {
    ImmutableSet.Builder<ModuleKey> allTargetKeys = new ImmutableSet.Builder<>();
    for (TargetModule targetModule : targetList) {
      allTargetKeys.addAll(targetToModuleKeySet(targetModule, modulesIndex));
    }
    return allTargetKeys.build();
  }

  // Helper to check the module-version argument exists and retrieve its present version(s) if not
  // specified
  private static ImmutableSet<ModuleKey> targetToModuleKeySet(
      TargetModule target, ImmutableMap<String, ImmutableSet<ModuleKey>> modulesIndex)
      throws InvalidArgumentException {
    if (target.getName().equals("") && target.getVersion() == Version.EMPTY) {
      return ImmutableSet.of(ModuleKey.ROOT);
    }
    ImmutableSet<ModuleKey> existingKeys = modulesIndex.get(target.getName());

    if (existingKeys == null) {
      throw new InvalidArgumentException(
          String.format("Module %s does not exist in the dependency graph.", target.getName()));
    }

    if (target.getVersion() == null) {
      return existingKeys;
    }
    ModuleKey key = ModuleKey.create(target.getName(), target.getVersion());
    if (!existingKeys.contains(key)) {
      throw new InvalidArgumentException(
          String.format(
              "Module version %s@%s does not exist, available versions: %s.",
              target.getName(), target.getVersion(), existingKeys));
    }
    return ImmutableSet.of(key);
  }

  private static BlazeCommandResult reportAndCreateFailureResult(
      CommandEnvironment env, String message, Code detailedCode) {
    if (message.charAt(message.length() - 1) != '.') {
      message = message.concat(".");
    }
    String fullMessage =
        String.format(
            message.concat(" Type '%s help modquery' for syntax and help."),
            env.getRuntime().getProductName());
    env.getReporter().handle(Event.error(fullMessage));
    return createFailureResult(fullMessage, detailedCode);
  }

  private static BlazeCommandResult createFailureResult(String message, Code detailedCode) {
    return BlazeCommandResult.detailedExitCode(
        DetailedExitCode.of(
            FailureDetail.newBuilder()
                .setModqueryCommand(
                    FailureDetails.ModqueryCommand.newBuilder().setCode(detailedCode).build())
                .setMessage(message)
                .build()));
  }

  /** Exception thrown when a user-input argument is invalid */
  private static class InvalidArgumentException extends Exception {
    private InvalidArgumentException(String message) {
      super(message);
    }
  }
}
