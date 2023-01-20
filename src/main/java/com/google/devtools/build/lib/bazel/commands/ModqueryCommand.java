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

import static com.google.common.collect.ImmutableMap.toImmutableMap;
import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static java.nio.charset.StandardCharsets.US_ASCII;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableBiMap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.bazel.bzlmod.BazelModuleInspectorValue;
import com.google.devtools.build.lib.bazel.bzlmod.BazelModuleInspectorValue.AugmentedModule;
import com.google.devtools.build.lib.bazel.bzlmod.BzlmodRepoRuleValue;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleKey;
import com.google.devtools.build.lib.bazel.bzlmod.Version;
import com.google.devtools.build.lib.bazel.commands.ModqueryOptions.Charset;
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
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.ModqueryCommand.Code;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.InterruptedFailureDetails;
import com.google.devtools.build.skyframe.EvaluationContext;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.devtools.common.options.OptionsParsingResult;
import java.io.OutputStreamWriter;
import java.util.List;
import java.util.Objects;

/** Queries the Bzlmod external dependency graph. */
@Command(
    name = ModqueryCommand.NAME,
    // TODO(andreisolo): figure out which extra options are really needed
    options = {
      ModqueryOptions.class,
      // Don't know what these do but were used in fetch
      PackageOptions.class,
      KeepGoingOption.class,
      LoadingPhaseThreadsOption.class
    },
    help = "resource:modquery.txt",
    shortDescription = "Queries the Bzlmod external dependency graph",
    allowResidue = true)
public final class ModqueryCommand implements BlazeCommand {

  public static final String NAME = "modquery";

  @Override
  public BlazeCommandResult exec(CommandEnvironment env, OptionsParsingResult options) {
    BazelModuleInspectorValue moduleInspector;

    SkyframeExecutor skyframeExecutor = env.getSkyframeExecutor();
    LoadingPhaseThreadsOption threadsOption = options.getOptions(LoadingPhaseThreadsOption.class);

    EvaluationContext evaluationContext =
        EvaluationContext.newBuilder()
            .setParallelism(threadsOption.threads)
            .setEventHandler(env.getReporter())
            .build();

    try {
      env.syncPackageLoading(options);

      EvaluationResult<SkyValue> evaluationResult =
          skyframeExecutor.prepareAndGet(
              ImmutableSet.of(BazelModuleInspectorValue.KEY), evaluationContext);

      if (evaluationResult.hasError()) {
        Exception e = evaluationResult.getError().getException();
        String message = "Unexpected error during repository rule evaluation.";
        if (e != null) {
          message = e.getMessage();
        }
        return reportAndCreateFailureResult(env, message, Code.INVALID_ARGUMENTS);
      }

      moduleInspector =
          (BazelModuleInspectorValue) evaluationResult.get(BazelModuleInspectorValue.KEY);

    } catch (InterruptedException e) {
      String errorMessage = "Modquery interrupted: " + e.getMessage();
      env.getReporter().handle(Event.error(errorMessage));
      return BlazeCommandResult.detailedExitCode(
          InterruptedFailureDetails.detailedExitCode(errorMessage));
    } catch (AbruptExitException e) {
      env.getReporter().handle(Event.error(null, "Unknown error: " + e.getMessage()));
      return BlazeCommandResult.detailedExitCode(e.getDetailedExitCode());
    }

    ModqueryOptions modqueryOptions = options.getOptions(ModqueryOptions.class);
    Preconditions.checkArgument(modqueryOptions != null);

    if (options.getResidue().isEmpty()) {
      String errorMessage =
          String.format("No query type specified, choose one from : %s.", QueryType.printValues());
      return reportAndCreateFailureResult(env, errorMessage, Code.MODQUERY_COMMAND_UNKNOWN);
    }

    // The first keyword in the residue must be the QueryType, and then comes a list of "arguments".
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

    // Arguments are structured as a list of comma-separated target lists for generality,
    // even though there can only be 0 or 1 args so far.
    ImmutableList<ImmutableSet<ModuleKey>> argsKeysList;
    AugmentedModule rootModule = moduleInspector.getDepGraph().get(ModuleKey.ROOT);
    try {
      argsKeysList =
          parseTargetArgs(
              query.getArgNumber(),
              moduleInspector.getModulesIndex(),
              args,
              rootModule.getDeps(),
              rootModule.getUnusedDeps(),
              modqueryOptions.includeUnused);
    } catch (InvalidArgumentException e) {
      return reportAndCreateFailureResult(env, e.getMessage(), e.getCode());
    } catch (OptionsParsingException e) {
      return reportAndCreateFailureResult(env, e.getMessage(), Code.INVALID_ARGUMENTS);
    }
    /* Extract and check the --from argument */
    ImmutableSet<ModuleKey> fromKeys;
    if (modqueryOptions.modulesFrom == null) {
      fromKeys = ImmutableSet.of(ModuleKey.ROOT);
    } else {
      try {
        fromKeys =
            targetListToModuleKeySet(
                modqueryOptions.modulesFrom,
                moduleInspector.getModulesIndex(),
                rootModule.getDeps(),
                rootModule.getUnusedDeps(),
                modqueryOptions.includeUnused);
      } catch (InvalidArgumentException e) {
        return reportAndCreateFailureResult(env, e.getMessage(), e.getCode());
      }
    }

    ImmutableMap<ModuleKey, BzlmodRepoRuleValue> repoRuleValues = null;
    // If the query is a SHOW, also request the BzlmodRepoRuleValues from SkyFrame.
    // Unused modules do not have a BzlmodRepoRuleValue, so they are filtered out.
    if (query == QueryType.SHOW) {
      try {
        ImmutableSet<ModuleKey> keys =
            argsKeysList.get(0).stream()
                .filter(
                    k ->
                        ModqueryExecutor.filterUnused(
                            k, modqueryOptions.includeUnused, false, moduleInspector.getDepGraph()))
                .collect(toImmutableSet());
        ImmutableSet<SkyKey> skyKeys =
            keys.stream()
                .map(k -> BzlmodRepoRuleValue.key(k.getCanonicalRepoName()))
                .collect(toImmutableSet());
        EvaluationResult<SkyValue> result =
            env.getSkyframeExecutor().prepareAndGet(skyKeys, evaluationContext);
        repoRuleValues =
            keys.stream()
                .collect(
                    toImmutableMap(
                        k -> k,
                        k ->
                            (BzlmodRepoRuleValue)
                                result.get(BzlmodRepoRuleValue.key(k.getCanonicalRepoName()))));
      } catch (InterruptedException e) {
        String errorMessage = "Modquery interrupted: " + e.getMessage();
        env.getReporter().handle(Event.error(errorMessage));
        return BlazeCommandResult.detailedExitCode(
            InterruptedFailureDetails.detailedExitCode(errorMessage));
      }
    }

    // Workaround to allow different default value for DEPS and EXPLAIN, and also use
    // Integer.MAX_VALUE instead of the exact number string.
    if (modqueryOptions.depth < 1) {
      if (query == QueryType.EXPLAIN || query == QueryType.DEPS) {
        modqueryOptions.depth = 1;
      } else {
        modqueryOptions.depth = Integer.MAX_VALUE;
      }
    }

    ModqueryExecutor modqueryExecutor =
        new ModqueryExecutor(
            moduleInspector.getDepGraph(),
            modqueryOptions,
            new OutputStreamWriter(
                env.getReporter().getOutErr().getOutputStream(),
                modqueryOptions.charset == Charset.UTF8 ? UTF_8 : US_ASCII));

    switch (query) {
      case TREE:
        modqueryExecutor.tree(fromKeys);
        break;
      case DEPS:
        modqueryExecutor.tree(argsKeysList.get(0));
        break;
      case PATH:
        modqueryExecutor.path(fromKeys, argsKeysList.get(0));
        break;
      case ALL_PATHS:
        modqueryExecutor.allPaths(fromKeys, argsKeysList.get(0));
        break;
      case EXPLAIN:
        modqueryExecutor.allPaths(fromKeys, argsKeysList.get(0));
        break;
      case SHOW:
        Preconditions.checkArgument(repoRuleValues != null);
        modqueryExecutor.show(repoRuleValues);
        break;
    }

    return BlazeCommandResult.success();
  }

  /**
   * A general parser for an undefined number of arguments. The arguments are comma-separated lists
   * of {@link TargetModule}s. Each target {@link TargetModule} can represent a {@code repo_name},
   * as defined in the {@code MODULE.bazel} file of the root project, a specific version of a
   * module, or all present versions of a module. The root module can only be specified by the
   * {@code root} keyword, which takes precedence over the other above (in case of modules named
   * root).
   */
  @VisibleForTesting
  public static ImmutableList<ImmutableSet<ModuleKey>> parseTargetArgs(
      int requiredArgNum,
      ImmutableMap<String, ImmutableSet<ModuleKey>> modulesIndex,
      List<String> args,
      ImmutableBiMap<String, ModuleKey> rootDeps,
      ImmutableBiMap<String, ModuleKey> rootUnusedDeps,
      boolean includeUnused)
      throws OptionsParsingException, InvalidArgumentException {
    if (requiredArgNum != args.size()) {
      throw new InvalidArgumentException(
          String.format(
              "Invalid number of arguments (provided %d, required %d).",
              args.size(), requiredArgNum),
          requiredArgNum > args.size() ? Code.MISSING_ARGUMENTS : Code.TOO_MANY_ARGUMENTS);
    }

    TargetModuleListConverter converter = new TargetModuleListConverter();
    ImmutableList.Builder<ImmutableSet<ModuleKey>> argsKeysListBuilder =
        new ImmutableList.Builder<>();

    for (String arg : args) {
      ImmutableList<TargetModule> targetList = converter.convert(arg);
      ImmutableSet<ModuleKey> argModuleKeys =
          targetListToModuleKeySet(
              targetList, modulesIndex, rootDeps, rootUnusedDeps, includeUnused);
      argsKeysListBuilder.add(argModuleKeys);
    }
    return argsKeysListBuilder.build();
  }

  /** Collects a list of {@link TargetModule} into a set of {@link ModuleKey}s. */
  private static ImmutableSet<ModuleKey> targetListToModuleKeySet(
      ImmutableList<TargetModule> targetList,
      ImmutableMap<String, ImmutableSet<ModuleKey>> modulesIndex,
      ImmutableBiMap<String, ModuleKey> rootDeps,
      ImmutableBiMap<String, ModuleKey> rootUnusedDeps,
      boolean includeUnused)
      throws InvalidArgumentException {
    ImmutableSet.Builder<ModuleKey> allTargetKeys = new ImmutableSet.Builder<>();
    for (TargetModule targetModule : targetList) {
      allTargetKeys.addAll(
          targetToModuleKeySet(
              targetModule, modulesIndex, rootDeps, rootUnusedDeps, includeUnused));
    }
    return allTargetKeys.build();
  }

  /**
   * Helper to check the module (and version) of the given {@link TargetModule} exists and retrieve
   * it, (or retrieve all present versions if it's semantic specifies it, i.e. when <code>
   * Version == null</code>).
   */
  private static ImmutableSet<ModuleKey> targetToModuleKeySet(
      TargetModule target,
      ImmutableMap<String, ImmutableSet<ModuleKey>> modulesIndex,
      ImmutableBiMap<String, ModuleKey> rootDeps,
      ImmutableBiMap<String, ModuleKey> rootUnusedDeps,
      boolean includeUnused)
      throws InvalidArgumentException {
    if (target.getName().isEmpty() && Objects.equals(target.getVersion(), Version.EMPTY)) {
      return ImmutableSet.of(ModuleKey.ROOT);
    }
    if (rootDeps.containsKey(target.getName())) {
      if (includeUnused && rootUnusedDeps.containsKey(target.getName())) {
        return ImmutableSet.of(
            rootDeps.get(target.getName()), rootUnusedDeps.get(target.getName()));
      }
      return ImmutableSet.of(rootDeps.get(target.getName()));
    }
    ImmutableSet<ModuleKey> existingKeys = modulesIndex.get(target.getName());

    if (existingKeys == null) {
      throw new InvalidArgumentException(
          String.format("Module %s does not exist in the dependency graph.", target.getName()),
          Code.INVALID_ARGUMENTS);
    }

    if (target.getVersion() == null) {
      return existingKeys;
    }
    ModuleKey key = ModuleKey.create(target.getName(), target.getVersion());
    if (!existingKeys.contains(key)) {
      throw new InvalidArgumentException(
          String.format(
              "Module version %s@%s does not exist, available versions: %s.",
              target.getName(), key, existingKeys),
          Code.INVALID_ARGUMENTS);
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

  /**
   * Exception thrown when a user-input argument is invalid (wrong number of arguments or the
   * specified modules do not exist).
   */
  @VisibleForTesting
  public static class InvalidArgumentException extends Exception {
    private final Code code;

    private InvalidArgumentException(String message, Code code) {
      super(message);
      this.code = code;
    }

    public Code getCode() {
      return code;
    }
  }
}
