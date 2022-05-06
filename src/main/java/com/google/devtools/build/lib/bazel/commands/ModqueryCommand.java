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

import com.google.auto.value.AutoValue;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.bazel.bzlmod.BazelModuleInspectorValue;
import com.google.devtools.build.lib.bazel.bzlmod.BazelModuleResolutionValue;
import com.google.devtools.build.lib.bazel.bzlmod.ModqueryExecutor;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleKey;
import com.google.devtools.build.lib.bazel.bzlmod.Version;
import com.google.devtools.build.lib.bazel.commands.ModqueryOptions.QueryType;
import com.google.devtools.build.lib.bazel.commands.ModqueryOptions.TargetModuleConverter;
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
import com.google.devtools.common.options.OptionsParsingResult;
import java.util.Arrays;
import java.util.List;
import javax.annotation.Nullable;

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
      modqueryExecutor.transitiveDeps(ModuleKey.ROOT);
      return BlazeCommandResult.success();
    }

    String input = options.getResidue().get(0);
    QueryType query;
    try {
      query = QueryType.valueOf(input.toUpperCase());
    } catch (IllegalArgumentException e) {
      String errorMessage =
          String.format(
              "Invalid query type, choose one from : (%s).", Arrays.toString(QueryType.values()));
      return reportAndCreateFailureResult(env, errorMessage, Code.MODQUERY_COMMAND_UNKNOWN);
    }

    List<String> args = options.getResidue().subList(1, options.getResidue().size());
    TargetModuleConverter converter = new TargetModuleConverter();

    if (query == QueryType.DEPS) {
      if (args.size() != 1) {
        return reportAndCreateFailureResult(
            env,
            String.format("invalid number of arguments (provided %d, required 1).", args.size()),
            Code.MISSING_ARGUMENTS);
      }
      try {
        ModuleKey target =
            targetToModuleKey(converter.convert(args.get(0)), moduleInspector.getModulesIndex());
        modqueryExecutor.deps(target);
        return BlazeCommandResult.success();
      } catch (Exception e) {
        return reportAndCreateFailureResult(env, e.getMessage(), Code.INVALID_ARGUMENTS);
      }
    } else if (query == QueryType.TRANSITIVE_DEPS) {
      if (args.size() != 1) {
        return reportAndCreateFailureResult(
            env,
            String.format("invalid number of arguments (provided %d, required 1).", args.size()),
            Code.MISSING_ARGUMENTS);
      }
      try {
        ModuleKey target =
            targetToModuleKey(converter.convert(args.get(0)), moduleInspector.getModulesIndex());
        modqueryExecutor.transitiveDeps(target);
        return BlazeCommandResult.success();
      } catch (Exception e) {
        return reportAndCreateFailureResult(env, e.getMessage(), Code.INVALID_ARGUMENTS);
      }
    } else if (query == QueryType.ALL_PATHS) {
      if (args.size() != 2) {
        return reportAndCreateFailureResult(
            env,
            String.format("invalid number of arguments (provided %d, required 2).", args.size()),
            Code.MISSING_ARGUMENTS);
      }
      try {
        ModuleKey from =
            targetToModuleKey(converter.convert(args.get(0)), moduleInspector.getModulesIndex());
        ModuleKey to =
            targetToModuleKey(converter.convert(args.get(1)), moduleInspector.getModulesIndex());
        modqueryExecutor.allPaths(from, to);
        return BlazeCommandResult.success();
      } catch (Exception e) {
        return reportAndCreateFailureResult(env, e.getMessage(), Code.INVALID_ARGUMENTS);
      }
    } else if (query == QueryType.PATH) {
      if (args.size() != 2) {
        return reportAndCreateFailureResult(
            env,
            String.format("invalid number of arguments (provided %d, required 2).", args.size()),
            Code.MISSING_ARGUMENTS);
      }
      try {
        ModuleKey from =
            targetToModuleKey(converter.convert(args.get(0)), moduleInspector.getModulesIndex());
        ModuleKey to =
            targetToModuleKey(converter.convert(args.get(1)), moduleInspector.getModulesIndex());
        modqueryExecutor.allPaths(from, to);
        return BlazeCommandResult.success();
      } catch (Exception e) {
        return reportAndCreateFailureResult(env, e.getMessage(), Code.INVALID_ARGUMENTS);
      }
    } else {
      // else if (query == QueryType.EXPLAIN)
      Preconditions.checkArgument(query == QueryType.EXPLAIN);
      if (args.size() != 1) {
        return reportAndCreateFailureResult(
            env,
            String.format("invalid number of arguments (provided %d, required 2).", args.size()),
            Code.MISSING_ARGUMENTS);
      }
      try {
        ImmutableSet<ModuleKey> targets =
            targetToModuleKeySet(
                converter.convert(args.get(0)), moduleInspector.getModulesIndex(), true);
        modqueryExecutor.explain(targets);
        return BlazeCommandResult.success();
      } catch (Exception e) {
        return reportAndCreateFailureResult(env, e.getMessage(), Code.INVALID_ARGUMENTS);
      }
    }
  }

  private static BlazeCommandResult reportAndCreateFailureResult(
      CommandEnvironment env, String message, Code detailedCode) {
    env.getReporter().handle(Event.error(message));
    return createFailureResult(env, message, detailedCode);
  }

  private static BlazeCommandResult createFailureResult(
      CommandEnvironment env, String message, Code detailedCode) {
    return BlazeCommandResult.detailedExitCode(
        DetailedExitCode.of(
            FailureDetail.newBuilder()
                .setModqueryCommand(
                    FailureDetails.ModqueryCommand.newBuilder().setCode(detailedCode).build())
                .setMessage(
                    String.format(
                        message + " Type '%s help modquery' for syntax and help.",
                        env.getRuntime().getProductName()))
                .build()));
  }

  /** Exception thrown when a user-input argument is invalid */
  private static class InvalidArgumentException extends Exception {
    private InvalidArgumentException(String message) {
      super(message);
    }
  }

  // Helper to check the module-version argument exists and retrieve its present version(s) if not
  // specified
  private static ImmutableSet<ModuleKey> targetToModuleKeySet(
      TargetModule target,
      ImmutableMap<String, ImmutableSet<ModuleKey>> modulesIndex,
      boolean allowMore)
      throws InvalidArgumentException {
    ImmutableSet<ModuleKey> existingKeys = modulesIndex.get(target.getName());

    if (existingKeys == null) {
      throw new InvalidArgumentException(
          String.format("Module %s does not exist in the dependency graph.", target.getName()));
    }
    if (target.getVersion() == null) {
      if (!allowMore && existingKeys.size() > 1) {
        throw new InvalidArgumentException(
            String.format(
                "The argument %s can only specify a single module version, please select one of:"
                    + " (%s).",
                target.getName(), existingKeys));
      }
      return ImmutableSet.copyOf(existingKeys);
    } else {
      ModuleKey key = ModuleKey.create(target.getName(), target.getVersion());
      if (!existingKeys.contains(key)) {
        throw new InvalidArgumentException(
            String.format(
                "Module version %s@%s does not exist, available versions: (%s).",
                target.getName(), target.getVersion(), existingKeys));
      }
      return ImmutableSet.of(key);
    }
  }

  // Shortcut for targetToModuleKey which should return a single ModuleKey and throw an exception if
  // more keys with the same name are found
  private static ModuleKey targetToModuleKey(
      TargetModule target, ImmutableMap<String, ImmutableSet<ModuleKey>> modulesIndex)
      throws InvalidArgumentException {
    return targetToModuleKeySet(target, modulesIndex, false).asList().get(0);
  }

  /**
   * Argument of a modquery converted from the form &lt;name&gt;@&lt;version&gt; or &lt;name&gt;.
   */
  @AutoValue
  abstract static class TargetModule {
    abstract String getName();

    @Nullable
    /* If it is null, it represents any (one or multiple) present versions of the module in the dep
    graph, which is different from the empty version */
    abstract Version getVersion();

    static TargetModule create(String name, Version version) {
      return new AutoValue_ModqueryCommand_TargetModule(name, version);
    }
  }
}
