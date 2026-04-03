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

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.collect.ImmutableListMultimap.toImmutableListMultimap;
import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static com.google.devtools.build.lib.bazel.bzlmod.modcommand.ModOptions.Charset.UTF8;
import static com.google.devtools.build.lib.runtime.Command.BuildPhase.LOADS;
import static java.nio.charset.StandardCharsets.ISO_8859_1;
import static java.nio.charset.StandardCharsets.US_ASCII;
import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.stream.Collectors.joining;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableBiMap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Ordering;
import com.google.common.io.CharSource;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.devtools.build.lib.analysis.NoBuildEvent;
import com.google.devtools.build.lib.analysis.NoBuildRequestFinishedEvent;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.bazel.bzlmod.BazelDepGraphValue;
import com.google.devtools.build.lib.bazel.bzlmod.BazelModTidyValue;
import com.google.devtools.build.lib.bazel.bzlmod.BazelModuleInspectorValue;
import com.google.devtools.build.lib.bazel.bzlmod.BazelModuleInspectorValue.AugmentedModule;
import com.google.devtools.build.lib.bazel.bzlmod.BazelModuleResolutionValue;
import com.google.devtools.build.lib.bazel.bzlmod.InterimModule;
import com.google.devtools.build.lib.bazel.bzlmod.Module;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleExtensionId;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleKey;
import com.google.devtools.build.lib.bazel.bzlmod.Registry;
import com.google.devtools.build.lib.bazel.bzlmod.RootModuleFileFixup;
import com.google.devtools.build.lib.bazel.bzlmod.SingleExtensionValue;
import com.google.devtools.build.lib.bazel.bzlmod.Version;
import com.google.devtools.build.lib.bazel.bzlmod.modcommand.ExtensionArg;
import com.google.devtools.build.lib.bazel.bzlmod.modcommand.ExtensionArg.ExtensionArgConverter;
import com.google.devtools.build.lib.bazel.bzlmod.modcommand.InvalidArgumentException;
import com.google.devtools.build.lib.bazel.bzlmod.modcommand.ModExecutor;
import com.google.devtools.build.lib.bazel.bzlmod.modcommand.ModOptions;
import com.google.devtools.build.lib.bazel.bzlmod.modcommand.ModOptions.ModSubcommand;
import com.google.devtools.build.lib.bazel.bzlmod.modcommand.ModOptions.ModSubcommandConverter;
import com.google.devtools.build.lib.bazel.bzlmod.modcommand.ModuleArg;
import com.google.devtools.build.lib.bazel.bzlmod.modcommand.ModuleArg.ModuleArgConverter;
import com.google.devtools.build.lib.bazel.bzlmod.modcommand.VersionsRenderer;
import com.google.devtools.build.lib.bazel.bzlmod.modcommand.VersionsRenderer.ModuleVersionEntry;
import com.google.devtools.build.lib.bazel.repository.RepoDefinitionValue;
import com.google.devtools.build.lib.bazel.repository.downloader.DownloadManager;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.pkgcache.PackageOptions;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.runtime.BlazeCommand;
import com.google.devtools.build.lib.runtime.BlazeCommandResult;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.LoadingPhaseThreadsOption;
import com.google.devtools.build.lib.runtime.UiOptions;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.ModCommand.Code;
import com.google.devtools.build.lib.shell.AbnormalTerminationException;
import com.google.devtools.build.lib.shell.CommandException;
import com.google.devtools.build.lib.shell.CommandResult;
import com.google.devtools.build.lib.skyframe.BzlLoadCycleReporter;
import com.google.devtools.build.lib.skyframe.BzlmodRepoCycleReporter;
import com.google.devtools.build.lib.skyframe.RepositoryMappingValue;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.CommandBuilder;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.InterruptedFailureDetails;
import com.google.devtools.build.lib.util.MaybeCompleteSet;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.CyclesReporter;
import com.google.devtools.build.skyframe.EvaluationContext;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.devtools.common.options.OptionsParsingResult;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.stream.JsonWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.stream.IntStream;
import javax.annotation.Nullable;

/** Queries the Bzlmod external dependency graph. */
@Command(
    name = ModCommand.NAME,
    buildPhase = LOADS,
    options = {
      CoreOptions.class, // for --action_env, which affects the repo env
      ModOptions.class,
      PackageOptions.class,
      LoadingPhaseThreadsOption.class
    },
    help = "resource:mod.txt",
    shortDescription = "Queries the Bzlmod external dependency graph",
    allowResidue = true)
public final class ModCommand implements BlazeCommand {
  @Nullable private DownloadManager downloadManager;

  public void setDownloadManager(DownloadManager downloadManager) {
    this.downloadManager = downloadManager;
  }

  public static final String NAME = "mod";

  @Override
  public BlazeCommandResult exec(CommandEnvironment env, OptionsParsingResult options) {
    env.getEventBus()
        .post(
            new NoBuildEvent(
                env.getCommandName(),
                env.getCommandStartTime(),
                /* separateFinishedEvent= */ true,
                /* showProgress= */ true,
                /* id= */ null));
    BlazeCommandResult result = execInternal(env, options);
    env.getEventBus()
        .post(
            new NoBuildRequestFinishedEvent(
                result.getExitCode(), env.getRuntime().getClock().currentTimeMillis()));
    return result;
  }

  private void validateArgs(ModSubcommand subcommand, ModOptions modOptions, List<String> args)
      throws InvalidArgumentException {

    // Validate output format.
    switch (subcommand) {
      case SHOW_REPO -> {
        switch (modOptions.outputFormat) {
          case TEXT, STREAMED_JSONPROTO, STREAMED_PROTO -> {} // supported
          default ->
              throw new InvalidArgumentException(
                  String.format(
                      "Invalid --output '%s' for the 'show_repo' subcommand. Only 'text',"
                          + " 'streamed_jsonproto', and 'streamed_proto' are supported.",
                      modOptions.outputFormat),
                  Code.INVALID_ARGUMENTS);
        }
      }
      case SHOW_EXTENSION -> {
        if (modOptions.outputFormat != ModOptions.OutputFormat.TEXT) {
          throw new InvalidArgumentException(
              String.format(
                  "Invalid --output '%s' for the 'show_extension' subcommand. Only 'text' is"
                      + " supported.",
                  modOptions.outputFormat),
              Code.INVALID_ARGUMENTS);
        }
      }
      case ModSubcommand sub when sub.isGraph() -> {
        switch (modOptions.outputFormat) {
          case TEXT, JSON, GRAPH -> {} // supported
          default ->
              throw new InvalidArgumentException(
                  String.format(
                      "Invalid --output '%s' for the '%s' subcommand. "
                          + "Only 'text', 'json', and 'graph' are supported.",
                      modOptions.outputFormat, sub),
                  Code.INVALID_ARGUMENTS);
        }
      }
      // We don't validate other subcommands yet since they are less confusing.
      default -> {}
    }

    if (subcommand == ModSubcommand.SHOW_REPO) {
      int selectedModes = 0;
      if (modOptions.allRepos) {
        selectedModes++;
      }
      if (modOptions.allVisibleRepos) {
        selectedModes++;
      }
      if (!args.isEmpty()) {
        selectedModes++;
      }
      if (selectedModes > 1) {
        throw new InvalidArgumentException(
            "the 'show_repo' command requires exactly one of --all_repos, --all_visible_repos, or a"
                + " list of repo arguments",
            Code.TOO_MANY_ARGUMENTS);
      }
    } else {
      if (modOptions.allRepos) {
        throw new InvalidArgumentException(
            String.format("the '%s' command doesn't take the --all_repos option", subcommand),
            Code.INVALID_ARGUMENTS);
      }
      if (modOptions.allVisibleRepos) {
        throw new InvalidArgumentException(
            String.format(
                "the '%s' command doesn't take the --all_visible_repos option", subcommand),
            Code.INVALID_ARGUMENTS);
      }
    }

    if (subcommand == ModSubcommand.UPGRADE) {
      if (modOptions.all && !args.isEmpty()) {
        throw new InvalidArgumentException(
            "the 'upgrade' command doesn't accept both --all and a list of module names",
            Code.TOO_MANY_ARGUMENTS);
      }
    } else {
      if (modOptions.all) {
        throw new InvalidArgumentException(
            String.format("the '%s' command doesn't take the --all option", subcommand),
            Code.INVALID_ARGUMENTS);
      }
    }
  }

  private BlazeCommandResult execInternal(CommandEnvironment env, OptionsParsingResult options) {
    ModOptions modOptions = options.getOptions(ModOptions.class);
    Preconditions.checkArgument(modOptions != null);

    if (options.getResidue().isEmpty()) {
      String errorMessage =
          String.format(
              "No subcommand specified, choose one of : %s.", ModSubcommand.printValues());
      return reportAndCreateFailureResult(env, errorMessage, Code.MOD_COMMAND_UNKNOWN);
    }

    // The first element in the residue must be the subcommand, and then comes a list of arguments.
    String subcommandStr = options.getResidue().get(0);
    ModSubcommand subcommand;
    try {
      subcommand = new ModSubcommandConverter().convert(subcommandStr);
    } catch (OptionsParsingException e) {
      String errorMessage =
          String.format("Invalid subcommand, choose one from : %s.", ModSubcommand.printValues());
      return reportAndCreateFailureResult(env, errorMessage, Code.MOD_COMMAND_UNKNOWN);
    }
    List<String> args = options.getResidue().subList(1, options.getResidue().size());

    // Validate and parse args as early as possible, so we don't have to
    // wait for Skyframe evaluations to happen before failing due to a simple error.
    try {
      validateArgs(subcommand, modOptions, args);
    } catch (InvalidArgumentException e) {
      return reportAndCreateFailureResult(env, e.getMessage(), e.getCode());
    }

    ImmutableList.Builder<RepositoryMappingValue.Key> repositoryMappingKeysBuilder =
        ImmutableList.builder();
    if (subcommand.equals(ModSubcommand.DUMP_REPO_MAPPING)) {
      if (args.isEmpty()) {
        // Make this case an error so that we are free to add a mode that emits all mappings in a
        // single JSON object later.
        return reportAndCreateFailureResult(
            env, "No repository name(s) specified", Code.INVALID_ARGUMENTS);
      }
      for (String arg : args) {
        try {
          repositoryMappingKeysBuilder.add(RepositoryMappingValue.key(RepositoryName.create(arg)));
        } catch (LabelSyntaxException e) {
          return reportAndCreateFailureResult(env, e.getMessage(), Code.INVALID_ARGUMENTS);
        }
      }
    }
    ImmutableList<RepositoryMappingValue.Key> repoMappingKeys =
        repositoryMappingKeysBuilder.build();

    BazelDepGraphValue depGraphValue;
    @Nullable BazelModuleInspectorValue moduleInspector;
    @Nullable BazelModTidyValue modTidyValue;
    @Nullable BazelModuleResolutionValue resolutionValue;
    ImmutableList<RepositoryMappingValue> repoMappingValues;

    SkyframeExecutor skyframeExecutor = env.getSkyframeExecutor();
    LoadingPhaseThreadsOption threadsOption = options.getOptions(LoadingPhaseThreadsOption.class);

    EvaluationContext evaluationContext =
        EvaluationContext.newBuilder()
            .setParallelism(threadsOption.threads)
            .setEventHandler(env.getReporter())
            .build();

    try {
      env.syncPackageLoading(options);

      ImmutableSet.Builder<SkyKey> keys = ImmutableSet.builder();
      if (subcommand.equals(ModSubcommand.DUMP_REPO_MAPPING)) {
        keys.addAll(repoMappingKeys);
      } else if (subcommand.equals(ModSubcommand.TIDY)) {
        keys.add(BazelModTidyValue.KEY);
      } else {
        keys.add(BazelDepGraphValue.KEY, BazelModuleInspectorValue.KEY);
        if (subcommand.equals(ModSubcommand.UPGRADE)) {
          keys.add(BazelModuleResolutionValue.KEY);
          // Only request BazelModTidyValue when we'll actually modify files.
          if (!args.isEmpty() || modOptions.all) {
            keys.add(BazelModTidyValue.KEY);
          }
        }
      }
      EvaluationResult<SkyValue> evaluationResult =
          skyframeExecutor.prepareAndGet(keys.build(), evaluationContext);

      if (evaluationResult.hasError()) {
        var cycleInfo = evaluationResult.getError().getCycleInfo();
        if (!cycleInfo.isEmpty()) {
          // We don't expect target-level cycles here, so restrict to the subset of reporters that
          // are relevant for the (conceptual) loading phase.
          new CyclesReporter(new BzlmodRepoCycleReporter(), new BzlLoadCycleReporter())
              .reportCycles(cycleInfo, cycleInfo.getFirst().getTopKey(), env.getReporter());
        }
        Exception e = evaluationResult.getError().getException();
        String message = "Unexpected error during module graph evaluation.";
        if (e != null) {
          message = e.getMessage();
        }
        return reportAndCreateFailureResult(env, message, Code.MOD_COMMAND_UNKNOWN);
      }

      depGraphValue = (BazelDepGraphValue) evaluationResult.get(BazelDepGraphValue.KEY);

      moduleInspector =
          (BazelModuleInspectorValue) evaluationResult.get(BazelModuleInspectorValue.KEY);

      modTidyValue = (BazelModTidyValue) evaluationResult.get(BazelModTidyValue.KEY);

      resolutionValue =
          (BazelModuleResolutionValue) evaluationResult.get(BazelModuleResolutionValue.KEY);

      repoMappingValues =
          repoMappingKeys.stream()
              .map(evaluationResult::get)
              .map(RepositoryMappingValue.class::cast)
              .collect(toImmutableList());
    } catch (InterruptedException e) {
      return handleInterrupted(env, e);
    } catch (AbruptExitException e) {
      env.getReporter().handle(Event.error(null, "Unknown error: " + e.getMessage()));
      return BlazeCommandResult.detailedExitCode(e.getDetailedExitCode());
    }

    // Handle commands that do not require BazelModuleInspectorValue.
    if (subcommand.equals(ModSubcommand.DUMP_REPO_MAPPING)) {
      String missingRepos =
          IntStream.range(0, repoMappingKeys.size())
              .filter(i -> repoMappingValues.get(i) == RepositoryMappingValue.NOT_FOUND_VALUE)
              .mapToObj(repoMappingKeys::get)
              .map(RepositoryMappingValue.Key::repoName)
              .map(RepositoryName::getName)
              .collect(joining(", "));
      if (!missingRepos.isEmpty()) {
        return reportAndCreateFailureResult(
            env, "Repositories not found: " + missingRepos, Code.INVALID_ARGUMENTS);
      }
      try (SilentCloseable c =
          Profiler.instance().profile(ProfilerTask.BZLMOD, "execute mod " + subcommand)) {
        dumpRepoMappings(
            repoMappingValues,
            new OutputStreamWriter(
                env.getReporter().getOutErr().getOutputStream(),
                modOptions.charset == UTF8 ? UTF_8 : US_ASCII));
      } catch (IOException e) {
        throw new IllegalStateException(e);
      }
      return BlazeCommandResult.success();
    } else if (subcommand == ModSubcommand.TIDY) {
      // tidy doesn't take extra arguments.
      if (!args.isEmpty()) {
        return reportAndCreateFailureResult(
            env, "the 'tidy' command doesn't take extra arguments", Code.TOO_MANY_ARGUMENTS);
      }
      try (SilentCloseable c =
          Profiler.instance().profile(ProfilerTask.BZLMOD, "execute mod " + subcommand)) {
        return runTidy(env, modTidyValue);
      }
    } else if (subcommand == ModSubcommand.UPGRADE) {
      if (resolutionValue == null) {
        return reportAndCreateFailureResult(
            env, "Failed to compute module resolution.", Code.MOD_COMMAND_UNKNOWN);
      }
      boolean willModify = !args.isEmpty() || modOptions.all;
      if (willModify && modTidyValue == null) {
        return reportAndCreateFailureResult(
            env,
            "Failed to compute buildozer path for MODULE.bazel modification.",
            Code.MOD_COMMAND_UNKNOWN);
      }
      try (SilentCloseable c =
          Profiler.instance().profile(ProfilerTask.BZLMOD, "execute mod " + subcommand)) {
        return runUpgrade(
            env, args, depGraphValue, moduleInspector, resolutionValue, modOptions, modTidyValue);
      }
    }

    // Extract and check the --base_module argument first to use it when parsing the other args.
    // Can only be a TargetModule or a repoName relative to the ROOT.
    ModuleKey baseModuleKey;
    AugmentedModule rootModule = moduleInspector.depGraph().get(ModuleKey.ROOT);
    try {
      ImmutableSet<ModuleKey> keys =
          modOptions.baseModule.resolveToModuleKeys(
              moduleInspector.modulesIndex(),
              moduleInspector.depGraph(),
              moduleInspector.moduleKeyToCanonicalNames(),
              rootModule.deps(),
              rootModule.unusedDeps(),
              false,
              false);
      if (keys.size() > 1) {
        throw new InvalidArgumentException(
            String.format(
                "The --base_module option can only specify exactly one module version, choose one"
                    + " of: %s.",
                keys.stream().map(ModuleKey::toString).collect(joining(", "))),
            Code.INVALID_ARGUMENTS);
      }
      baseModuleKey = Iterables.getOnlyElement(keys);
    } catch (InvalidArgumentException e) {
      return reportAndCreateFailureResult(
          env,
          String.format(
              "In --base_module %s option: %s (Note that unused modules cannot be used here)",
              modOptions.baseModule, e.getMessage()),
          Code.INVALID_ARGUMENTS);
    }

    // The args can have different types depending on the subcommand, so create multiple containers
    // which can be filled accordingly.
    ImmutableSet<ModuleKey> argsAsModules = null;
    ImmutableSortedSet<ModuleExtensionId> argsAsExtensions = null;
    ImmutableMap<String, RepositoryName> argsAsRepos = null;

    AugmentedModule baseModule =
        Objects.requireNonNull(moduleInspector.depGraph().get(baseModuleKey));
    RepositoryMapping baseModuleMapping = depGraphValue.getFullRepoMapping(baseModuleKey);
    try {
      switch (subcommand) {
        case GRAPH -> {
          // GRAPH doesn't take extra arguments.
          if (!args.isEmpty()) {
            throw new InvalidArgumentException(
                "the 'graph' command doesn't take extra arguments", Code.TOO_MANY_ARGUMENTS);
          }
        }
        case SHOW_REPO -> {
          argsAsRepos =
              getReposToShow(modOptions, moduleInspector, depGraphValue, baseModuleMapping, args);
        }
        case SHOW_EXTENSION -> {
          ImmutableSortedSet.Builder<ModuleExtensionId> extensionsBuilder =
              new ImmutableSortedSet.Builder<>(ModuleExtensionId.LEXICOGRAPHIC_COMPARATOR);
          for (String arg : args) {
            try {
              extensionsBuilder.add(
                  ExtensionArgConverter.INSTANCE
                      .convert(arg)
                      .resolveToExtensionId(
                          moduleInspector.modulesIndex(),
                          moduleInspector.depGraph(),
                          moduleInspector.moduleKeyToCanonicalNames(),
                          baseModule.deps(),
                          baseModule.unusedDeps()));
            } catch (InvalidArgumentException | OptionsParsingException e) {
              throw new InvalidArgumentException(
                  String.format("In extension argument %s: %s", arg, e.getMessage()),
                  Code.INVALID_ARGUMENTS,
                  e);
            }
          }
          argsAsExtensions = extensionsBuilder.build();
        }
        default -> {
          ImmutableSet.Builder<ModuleKey> keysBuilder = new ImmutableSet.Builder<>();
          for (String arg : args) {
            try {
              keysBuilder.addAll(
                  ModuleArgConverter.INSTANCE
                      .convert(arg)
                      .resolveToModuleKeys(
                          moduleInspector.modulesIndex(),
                          moduleInspector.depGraph(),
                          moduleInspector.moduleKeyToCanonicalNames(),
                          baseModule.deps(),
                          baseModule.unusedDeps(),
                          modOptions.includeUnused,
                          /* warnUnused= */ true));
            } catch (InvalidArgumentException | OptionsParsingException e) {
              throw new InvalidArgumentException(
                  String.format("In module argument %s: %s", arg, e.getMessage()),
                  Code.INVALID_ARGUMENTS);
            }
          }
          argsAsModules = keysBuilder.build();
        }
      }
    } catch (InvalidArgumentException e) {
      return reportAndCreateFailureResult(env, e.getMessage(), e.getCode());
    }
    /* Extract and check the --from and --extension_usages argument */
    ImmutableSet<ModuleKey> fromKeys;
    ImmutableSet<ModuleKey> usageKeys;
    try {
      fromKeys =
          moduleArgListToKeys(
              modOptions.modulesFrom,
              moduleInspector.modulesIndex(),
              moduleInspector.depGraph(),
              moduleInspector.moduleKeyToCanonicalNames(),
              baseModule.deps(),
              baseModule.unusedDeps(),
              modOptions.includeUnused);
    } catch (InvalidArgumentException e) {
      return reportAndCreateFailureResult(
          env,
          String.format("In --from %s option: %s", modOptions.modulesFrom, e.getMessage()),
          Code.INVALID_ARGUMENTS);
    }

    try {
      usageKeys =
          moduleArgListToKeys(
              modOptions.extensionUsages,
              moduleInspector.modulesIndex(),
              moduleInspector.depGraph(),
              moduleInspector.moduleKeyToCanonicalNames(),
              baseModule.deps(),
              baseModule.unusedDeps(),
              modOptions.includeUnused);
    } catch (InvalidArgumentException e) {
      return reportAndCreateFailureResult(
          env,
          String.format(
              "In --extension_usages %s option: %s (Note that unused modules cannot be used"
                  + " here)",
              modOptions.extensionUsages, e.getMessage()),
          Code.INVALID_ARGUMENTS);
    }

    /* Extract and check the --extension_filter argument */
    Optional<MaybeCompleteSet<ModuleExtensionId>> filterExtensions = Optional.empty();
    if (subcommand.isGraph() && modOptions.extensionFilter != null) {
      if (modOptions.extensionFilter.isEmpty()) {
        filterExtensions = Optional.of(MaybeCompleteSet.completeSet());
      } else {
        try {
          filterExtensions =
              Optional.of(
                  MaybeCompleteSet.copyOf(
                      extensionArgListToIds(
                          modOptions.extensionFilter,
                          moduleInspector.modulesIndex(),
                          moduleInspector.depGraph(),
                          moduleInspector.moduleKeyToCanonicalNames(),
                          baseModule.deps(),
                          baseModule.unusedDeps())));
        } catch (InvalidArgumentException e) {
          return reportAndCreateFailureResult(
              env,
              String.format(
                  "In --extension_filter %s option: %s",
                  modOptions.extensionFilter, e.getMessage()),
              Code.INVALID_ARGUMENTS);
        }
      }
    }

    ImmutableMap<String, RepoDefinitionValue> targetRepoDefinitions = null;
    try {
      if (subcommand == ModSubcommand.SHOW_REPO) {
        ImmutableSet<SkyKey> skyKeys =
            argsAsRepos.values().stream().map(RepoDefinitionValue::key).collect(toImmutableSet());
        EvaluationResult<SkyValue> result =
            env.getSkyframeExecutor().prepareAndGet(skyKeys, evaluationContext);
        if (result.hasError()) {
          Exception e = result.getError().getException();
          String message = "Unexpected error during repository rule evaluation.";
          if (e != null) {
            message = e.getMessage();
          }
          return reportAndCreateFailureResult(env, message, Code.INVALID_ARGUMENTS);
        }
        var resultBuilder =
            ImmutableMap.<String, RepoDefinitionValue>builderWithExpectedSize(argsAsRepos.size());
        for (Map.Entry<String, RepositoryName> e : argsAsRepos.entrySet()) {
          SkyValue value = result.get(RepoDefinitionValue.key(e.getValue()));
          if (value == RepoDefinitionValue.NOT_FOUND) {
            return reportAndCreateFailureResult(
                env,
                String.format("In repo argument %s: no such repo", e.getKey()),
                Code.INVALID_ARGUMENTS);
          }
          resultBuilder.put(e.getKey(), (RepoDefinitionValue) value);
        }
        targetRepoDefinitions = resultBuilder.buildOrThrow();
      }
    } catch (InterruptedException e) {
      return handleInterrupted(env, e);
    }

    // Workaround to allow different default value for DEPS and EXPLAIN, and also use
    // Integer.MAX_VALUE instead of the exact number string.
    if (modOptions.depth < 1) {
      modOptions.depth =
          switch (subcommand) {
            case EXPLAIN -> 1;
            case DEPS -> 2;
            default -> Integer.MAX_VALUE;
          };
    }

    ModExecutor modExecutor =
        new ModExecutor(
            moduleInspector.depGraph(),
            depGraphValue.getExtensionUsagesTable(),
            moduleInspector.extensionToRepoInternalNames(),
            filterExtensions,
            modOptions,
            env.getReporter().getOutErr().getOutputStream());

    try (SilentCloseable c =
        Profiler.instance().profile(ProfilerTask.BZLMOD, "execute mod " + subcommand)) {
      switch (subcommand) {
        case GRAPH -> modExecutor.graph(fromKeys);
        case DEPS -> modExecutor.graph(argsAsModules);
        case PATH -> modExecutor.path(fromKeys, argsAsModules);
        case ALL_PATHS, EXPLAIN -> modExecutor.allPaths(fromKeys, argsAsModules);
        case SHOW_REPO -> modExecutor.showRepo(targetRepoDefinitions);
        case SHOW_EXTENSION -> modExecutor.showExtension(argsAsExtensions, usageKeys);
        default -> throw new IllegalStateException("Unexpected subcommand: " + subcommand);
      }
    } catch (InvalidArgumentException e) {
      return reportAndCreateFailureResult(env, e.getMessage(), Code.INVALID_ARGUMENTS);
    }

    if (moduleInspector.errors().isEmpty()) {
      return BlazeCommandResult.success();
    } else {
      return reportAndCreateFailureResult(
          env,
          String.format(
              "Results may be incomplete as %d extension%s failed.",
              moduleInspector.errors().size(), moduleInspector.errors().size() == 1 ? "" : "s"),
          Code.ERROR_DURING_GRAPH_INSPECTION);
    }
  }

  private ImmutableMap<String, RepositoryName> getReposToShow(
      ModOptions modOptions,
      BazelModuleInspectorValue moduleInspector,
      BazelDepGraphValue depGraphValue,
      RepositoryMapping baseModuleMapping,
      List<String> args)
      throws InvalidArgumentException {

    ImmutableMap.Builder<String, RepositoryName> targetToRepoName = new ImmutableMap.Builder<>();

    if (modOptions.allRepos) {
      // Module repos.
      for (RepositoryName repoName : moduleInspector.moduleKeyToCanonicalNames().values()) {
        if (repoName.isMain()) {
          // The main repo can't be inspected.
          continue;
        }
        targetToRepoName.put(repoName.getNameWithAt(), repoName);
      }

      // Extension repos.
      for (Map.Entry<ModuleExtensionId, Collection<String>> extensionRepos :
          moduleInspector.extensionToRepoInternalNames().asMap().entrySet()) {
        String extensionUniqueName =
            depGraphValue.getExtensionUniqueNames().get(extensionRepos.getKey());

        for (String internalName : extensionRepos.getValue()) {
          RepositoryName repoName =
              SingleExtensionValue.repositoryName(extensionUniqueName, internalName);
          targetToRepoName.put(repoName.getNameWithAt(), repoName);
        }
      }
    } else if (modOptions.allVisibleRepos) {
      for (Entry<String, RepositoryName> entry : baseModuleMapping.entries().entrySet()) {
        if (entry.getValue().isMain()) {
          // The main repo can't be inspected.
          continue;
        }
        targetToRepoName.put("@" + entry.getKey(), entry.getValue());
      }
    } else {
      // Resolve explicitly specified repos.
      for (String arg : args) {
        try {
          targetToRepoName.putAll(
              ModuleArgConverter.INSTANCE
                  .convert(arg)
                  .resolveToRepoNames(
                      moduleInspector.modulesIndex(),
                      moduleInspector.depGraph(),
                      moduleInspector.moduleKeyToCanonicalNames(),
                      baseModuleMapping));
        } catch (InvalidArgumentException | OptionsParsingException e) {
          throw new InvalidArgumentException(
              String.format(
                  "In repo argument %s: %s (Note that unused modules cannot be used here)",
                  arg, e.getMessage()),
              Code.INVALID_ARGUMENTS,
              e);
        }
      }
    }
    return targetToRepoName.buildKeepingLast();
  }

  private BlazeCommandResult runTidy(CommandEnvironment env, BazelModTidyValue modTidyValue) {
    ImmutableListMultimap<PathFragment, String> allCommandsPerFile =
        modTidyValue.fixups().stream()
            .flatMap(fixup -> fixup.moduleFilePathToBuildozerCommands().entries().stream())
            .collect(toImmutableListMultimap(Entry::getKey, Entry::getValue));
    StringBuilder buildozerInput = new StringBuilder();
    for (PathFragment moduleFilePath : modTidyValue.moduleFilePaths()) {
      buildozerInput.append("//").append(moduleFilePath).append(":all|");
      for (String command : allCommandsPerFile.get(moduleFilePath)) {
        buildozerInput.append(command).append('|');
      }
      buildozerInput.append("format\n");
    }

    boolean changesMade;
    try {
      changesMade = executeBuildozer(env, modTidyValue, buildozerInput);
    } catch (InterruptedException e) {
      return handleInterrupted(env, e);
    } catch (CommandException | IOException e) {
      return reportAndCreateFailureResult(env, formatBuildozerError(e), Code.BUILDOZER_FAILED);
    }

    if (changesMade) {
      for (RootModuleFileFixup fixupEvent : modTidyValue.fixups()) {
        env.getReporter().handle(Event.info(fixupEvent.getSuccessMessage()));
      }
    }

    return reportAndCreateTidyResult(env, modTidyValue);
  }

  private BlazeCommandResult runUpgrade(
      CommandEnvironment env,
      List<String> args,
      BazelDepGraphValue depGraphValue,
      BazelModuleInspectorValue moduleInspector,
      BazelModuleResolutionValue resolutionValue,
      ModOptions modOptions,
      @Nullable BazelModTidyValue modTidyValue) {
    ImmutableMap<ModuleKey, Module> resolvedGraph = depGraphValue.getDepGraph();
    ImmutableMap<ModuleKey, InterimModule> unprunedGraph = resolutionValue.getUnprunedDepGraph();

    // Identify direct dependency module keys from the root module.
    AugmentedModule rootModule = moduleInspector.depGraph().get(ModuleKey.ROOT);
    ImmutableSet<ModuleKey> directDepKeys = ImmutableSet.copyOf(rootModule.deps().values());

    // Collect registries for each non-root module in the resolved graph.
    // Track modules that are skipped due to overrides (single_version_override,
    // multiple_version_override, non-registry overrides like local_path_override, etc.).
    LinkedHashMap<String, Registry> registryByModule = new LinkedHashMap<>();
    ImmutableSet.Builder<String> overriddenModulesBuilder = ImmutableSet.builder();
    HashSet<String> seenModuleNames = new HashSet<>();
    for (ModuleKey moduleKey : resolvedGraph.keySet()) {
      if (moduleKey.equals(ModuleKey.ROOT)) {
        continue;
      }
      if (!seenModuleNames.add(moduleKey.name())) {
        // Multiple versions of the same module (multiple_version_override), skip.
        overriddenModulesBuilder.add(moduleKey.name());
        registryByModule.remove(moduleKey.name());
        continue;
      }
      InterimModule interim = unprunedGraph.get(moduleKey);
      if (interim == null) {
        // Module has a single_version_override or non-registry override, skip.
        overriddenModulesBuilder.add(moduleKey.name());
        continue;
      }
      Registry registry = interim.getRegistry();
      if (registry == null) {
        // Module has a non-registry override (e.g. local_path_override), skip.
        overriddenModulesBuilder.add(moduleKey.name());
        continue;
      }
      registryByModule.putIfAbsent(moduleKey.name(), registry);
    }
    ImmutableSet<String> overriddenModules = overriddenModulesBuilder.build();

    // Fetch available versions from each module's registry in parallel.
    ImmutableMap<String, Optional<ImmutableList<Version>>> availableVersionsMap;
    int threads = env.getOptions().getOptions(LoadingPhaseThreadsOption.class).threads;
    try (var executor =
        Executors.newFixedThreadPool(
            Math.min(registryByModule.size(), threads),
            new ThreadFactoryBuilder().setNameFormat("mod-upgrade-version-fetch-%d").build())) {
      availableVersionsMap =
          fetchAllAvailableVersions(registryByModule, executor, env.getReporter());
    } catch (InterruptedException e) {
      return handleInterrupted(env, e);
    }

    // Build version entries, grouped by direct/transitive.
    ImmutableList.Builder<ModuleVersionEntry> directDepsBuilder = ImmutableList.builder();
    ImmutableList.Builder<ModuleVersionEntry> transitiveDepsBuilder = ImmutableList.builder();

    for (ModuleKey moduleKey : resolvedGraph.keySet()) {
      if (moduleKey.equals(ModuleKey.ROOT)) {
        continue;
      }
      if (!registryByModule.containsKey(moduleKey.name())) {
        continue;
      }

      Version latest = null;
      Optional<ImmutableList<Version>> availableVersions =
          availableVersionsMap.get(moduleKey.name());
      if (availableVersions != null && availableVersions.isPresent()) {
        latest = VersionsRenderer.findLatestStable(availableVersions.get());
      }

      boolean isDirect = directDepKeys.contains(moduleKey);
      ModuleVersionEntry entry =
          new ModuleVersionEntry(moduleKey.name(), moduleKey.version(), latest, isDirect);
      if (isDirect) {
        directDepsBuilder.add(entry);
      } else {
        transitiveDepsBuilder.add(entry);
      }
    }

    ImmutableList<ModuleVersionEntry> directDeps = directDepsBuilder.build();
    ImmutableList<ModuleVersionEntry> transitiveDeps = transitiveDepsBuilder.build();

    boolean willModify = !args.isEmpty() || modOptions.all;

    if (!willModify) {
      // Display-only mode: show the versions table with a hint message.
      boolean useColor = env.getOptions().getOptions(UiOptions.class).useColor();
      VersionsRenderer renderer =
          new VersionsRenderer(
              env.getReporter().getOutErr().getOutputStream(),
              useColor,
              modOptions.charset == UTF8);
      renderer.render(directDeps, transitiveDeps);
      renderer.renderHint();
      return BlazeCommandResult.success();
    }

    // Upgrade mode: determine which modules to upgrade.
    ImmutableMap<String, ModuleVersionEntry> directDepsByName =
        directDeps.stream()
            .collect(ImmutableMap.toImmutableMap(ModuleVersionEntry::name, entry -> entry));
    ImmutableMap<String, ModuleVersionEntry> transitiveDepsByName =
        transitiveDeps.stream()
            .collect(ImmutableMap.toImmutableMap(ModuleVersionEntry::name, entry -> entry));

    // Direct deps whose version will be updated in-place.
    ImmutableList.Builder<ModuleVersionEntry> toUpgradeBuilder = ImmutableList.builder();
    // Indirect deps that will be promoted to direct deps with the latest version.
    ImmutableList.Builder<ModuleVersionEntry> toPromoteBuilder = ImmutableList.builder();

    if (modOptions.all) {
      // Upgrade all direct deps that have a newer version available.
      for (ModuleVersionEntry entry : directDeps) {
        if (entry.latest() != null && entry.latest().compareTo(entry.installed()) > 0) {
          toUpgradeBuilder.add(entry);
        }
      }
    } else {
      // Upgrade specific named modules.
      for (String moduleName : args) {
        ModuleVersionEntry entry = directDepsByName.get(moduleName);
        if (entry != null) {
          // Direct dependency: upgrade version in-place.
          if (entry.latest() == null) {
            String msg =
                String.format("Skipping %s: could not determine latest version.", moduleName);
            env.getReporter().handle(Event.warn(msg));
            continue;
          }
          if (entry.latest().compareTo(entry.installed()) <= 0) {
            String msg =
                String.format(
                    "%s is already at the latest version %s.", moduleName, entry.installed());
            env.getReporter().handle(Event.info(msg));
            continue;
          }
          toUpgradeBuilder.add(entry);
          continue;
        }

        // Check transitive deps: promote to direct dependency with the latest version.
        entry = transitiveDepsByName.get(moduleName);
        if (entry == null) {
          if (overriddenModules.contains(moduleName)) {
            String msg =
                String.format(
                    "Skipping %s: module has an override and won't be upgraded automatically.",
                    moduleName);
            env.getReporter().handle(Event.warn(msg));
            continue;
          }
          return reportAndCreateFailureResult(
              env,
              String.format(
                  "Module '%s' is not in the dependency graph. Run '%s mod upgrade' to see"
                      + " all dependencies.",
                  moduleName, env.getRuntime().getProductName()),
              Code.INVALID_ARGUMENTS);
        }
        if (entry.latest() == null) {
          String msg =
              String.format("Skipping %s: could not determine latest version.", moduleName);
          env.getReporter().handle(Event.warn(msg));
          continue;
        }
        if (entry.latest().compareTo(entry.installed()) <= 0) {
          String msg =
              String.format(
                  "%s is already at the latest version %s.", moduleName, entry.installed());
          env.getReporter().handle(Event.info(msg));
          continue;
        }
        toPromoteBuilder.add(entry);
      }
    }

    ImmutableList<ModuleVersionEntry> toUpgrade = toUpgradeBuilder.build();
    ImmutableList<ModuleVersionEntry> toPromote = toPromoteBuilder.build();

    if (toUpgrade.isEmpty() && toPromote.isEmpty()) {
      env.getReporter().handle(Event.info("All specified modules are already up to date."));
      return BlazeCommandResult.success();
    }

    // Generate buildozer commands for all MODULE.bazel modifications.
    StringBuilder buildozerInput = new StringBuilder();

    // Direct dep upgrades: update version in-place.
    for (ModuleVersionEntry entry : toUpgrade) {
      buildozerInput
          .append("//MODULE.bazel:")
          .append(entry.name())
          .append("|set version \"")
          .append(entry.latest())
          .append("\"\n");
    }

    // Indirect dep promotions: create new bazel_dep entries with repo_name = None,
    // placed at the correct position relative to existing nodep entries.
    boolean changesMade;
    try {
      if (!toPromote.isEmpty()) {
        generatePromoteCommands(env, modTidyValue, toPromote, buildozerInput);
      }
      // Run a format pass on all module file paths.
      for (PathFragment moduleFilePath : modTidyValue.moduleFilePaths()) {
        buildozerInput.append("//").append(moduleFilePath).append(":all|format\n");
      }

      changesMade = executeBuildozer(env, modTidyValue, buildozerInput);
    } catch (InterruptedException e) {
      return handleInterrupted(env, e);
    } catch (CommandException | IOException e) {
      return reportAndCreateFailureResult(env, formatBuildozerError(e), Code.BUILDOZER_FAILED);
    }

    if (!changesMade) {
      env.getReporter().handle(Event.info("No changes were made to MODULE.bazel."));
      return BlazeCommandResult.success();
    }

    // Report results.
    for (ModuleVersionEntry entry : toUpgrade) {
      env.getReporter()
          .handle(
              Event.info(
                  String.format(
                      "Upgraded %s from %s to %s.",
                      entry.name(), entry.installed(), entry.latest())));
    }
    for (ModuleVersionEntry entry : toPromote) {
      env.getReporter()
          .handle(
              Event.info(
                  String.format(
                      "Upgraded %s from %s to %s (indirect dependency).",
                      entry.name(), entry.installed(), entry.latest())));
    }
    int totalChanges = toUpgrade.size() + toPromote.size();
    env.getReporter()
        .handle(
            Event.info(
                String.format(
                    "%d module%s upgraded.", totalChanges, totalChanges == 1 ? "" : "s")));

    return BlazeCommandResult.success();
  }

  /**
   * Queries MODULE.bazel via buildozer to find existing {@code bazel_dep(..., repo_name = None)}
   * entries and generates the buildozer commands to insert each promoted indirect dep at the
   * correct position. Appends commands to {@code buildozerInput}.
   *
   * <p>If the existing indirect deps are in an alphabetically sorted group, new entries are
   * inserted maintaining the alphabetical order. Otherwise, new entries are appended after the last
   * existing entry.
   */
  private static void generatePromoteCommands(
      CommandEnvironment env,
      BazelModTidyValue modTidyValue,
      List<ModuleVersionEntry> toPromote,
      StringBuilder buildozerInput)
      throws InterruptedException {
    // Run: buildozer 'print name repo_name' //MODULE.bazel:%bazel_dep
    ImmutableList<String> indirectDeps;
    try {
      CommandResult result =
          new CommandBuilder(env.getClientEnv())
              .setWorkingDir(env.getWorkspace())
              .addArg(modTidyValue.buildozer().getPathString())
              .addArg("print name repo_name")
              .addArg("//MODULE.bazel:%bazel_dep")
              .build()
              .execute();
      String stdout = new String(result.getStdout(), UTF_8).trim();
      indirectDeps =
          stdout
              .lines()
              .map(line -> line.trim().split("\\s+"))
              .filter(parts -> parts.length >= 2 && parts[parts.length - 1].equals("None"))
              .map(parts -> parts[0])
              .collect(toImmutableList());
    } catch (CommandException e) {
      for (ModuleVersionEntry entry : toPromote) {
        appendNewBazelDep(buildozerInput, entry, "");
      }
      return;
    }

    // Separate entries that already exist (just need version update) from truly new ones.
    ImmutableSet<String> existingIndirectDeps = ImmutableSet.copyOf(indirectDeps);
    ImmutableList.Builder<ModuleVersionEntry> newEntriesBuilder = ImmutableList.builder();
    for (ModuleVersionEntry entry : toPromote) {
      if (existingIndirectDeps.contains(entry.name())) {
        // Already has a bazel_dep(..., repo_name = None) — just update version in-place.
        buildozerInput
            .append("//MODULE.bazel:")
            .append(entry.name())
            .append("|set version \"")
            .append(entry.latest())
            .append("\"\n");
      } else {
        newEntriesBuilder.add(entry);
      }
    }
    ImmutableList<ModuleVersionEntry> newEntries = newEntriesBuilder.build();

    // If all upgrades affect only existing indirect deps, we're already done.
    if (newEntries.isEmpty()) {
      return;
    }

    // If no indirect deps, just let buildozer append new entries at the end of the file.
    if (indirectDeps.isEmpty()) {
      for (ModuleVersionEntry entry : newEntries) {
        appendNewBazelDep(buildozerInput, entry, "");
      }
      return;
    }

    // If the existing group is not sorted alphabetically, just append new entries after the last
    // one.
    if (!Ordering.natural().isOrdered(indirectDeps)) {
      String lastEntry = indirectDeps.getLast();
      for (ModuleVersionEntry entry : newEntries) {
        appendNewBazelDep(buildozerInput, entry, " after " + lastEntry);
        lastEntry = entry.name();
      }
      return;
    }

    // Sorted — merge new entries into the sorted order.
    List<String> allNames = new ArrayList<>(indirectDeps);
    for (ModuleVersionEntry entry : newEntries) {
      allNames.add(entry.name());
    }
    allNames.sort(String::compareTo);

    ImmutableMap.Builder<String, Integer> nameToIndexBuilder = ImmutableMap.builder();
    for (int i = 0; i < allNames.size(); i++) {
      nameToIndexBuilder.put(allNames.get(i), i);
    }
    ImmutableMap<String, Integer> nameToIndex = nameToIndexBuilder.buildOrThrow();

    // If any new entry would be first, we need to read the comment from the current first entry
    // so we can move it to the new first entry after insertion.
    String currentFirst = indirectDeps.getFirst();
    String firstEntryComment = readComment(env, modTidyValue, currentFirst);

    for (ModuleVersionEntry entry : newEntries) {
      int idx = nameToIndex.get(entry.name());
      if (idx > 0) {
        appendNewBazelDep(buildozerInput, entry, " after " + allNames.get(idx - 1));
      } else {
        // First in sorted order. Move the comment from the current first entry to the new one.
        if (firstEntryComment != null && !firstEntryComment.isEmpty()) {
          buildozerInput.append("//MODULE.bazel:").append(currentFirst).append("|remove_comment\n");
        }
        appendNewBazelDep(buildozerInput, entry, " before " + currentFirst);
        if (firstEntryComment != null && !firstEntryComment.isEmpty()) {
          String escaped = escapeBuildozerString(firstEntryComment);
          buildozerInput
              .append("//MODULE.bazel:")
              .append(entry.name())
              .append("|comment ")
              .append(escaped)
              .append("\n");
        }
      }
    }
  }

  /**
   * Reads the comment attached to a rule via {@code buildozer 'print_comment'}. Returns the comment
   * text (without the leading {@code #}), or {@code null} if none.
   */
  @Nullable
  private static String readComment(
      CommandEnvironment env, BazelModTidyValue modTidyValue, String ruleName)
      throws InterruptedException {
    try {
      CommandResult result =
          new CommandBuilder(env.getClientEnv())
              .setWorkingDir(env.getWorkspace())
              .addArg(modTidyValue.buildozer().getPathString())
              .addArg("print_comment")
              .addArg("//MODULE.bazel:" + ruleName)
              .build()
              .execute();
      String comment = new String(result.getStdout(), UTF_8).trim();
      return comment.isEmpty() ? null : comment;
    } catch (CommandException e) {
      return null;
    }
  }

  /** Appends buildozer commands to create a new {@code bazel_dep(..., repo_name = None)} entry. */
  private static void appendNewBazelDep(
      StringBuilder buildozerInput, ModuleVersionEntry entry, String position) {
    buildozerInput
        .append("//MODULE.bazel:__pkg__|new bazel_dep ")
        .append(entry.name())
        .append(position)
        .append("\n");
    buildozerInput
        .append("//MODULE.bazel:")
        .append(entry.name())
        .append("|set version \"")
        .append(entry.latest())
        .append("\"|set repo_name None\n");
  }

  /**
   * Escapes a string for use in a buildozer command file. Handles characters that have special
   * meaning in buildozer's command syntax: backslashes, pipes (command separator), newlines
   * (command boundary), double quotes (string delimiter), and spaces.
   */
  private static String escapeBuildozerString(String s) {
    return s.replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\n", "\\n")
        .replace("\"", "\\\"")
        .replace(" ", "\\ ");
  }

  private static BlazeCommandResult handleInterrupted(
      CommandEnvironment env, InterruptedException e) {
    String errorMessage = "mod command interrupted: " + e.getMessage();
    env.getReporter().handle(Event.error(errorMessage));
    return BlazeCommandResult.detailedExitCode(
        InterruptedFailureDetails.detailedExitCode(errorMessage));
  }

  /**
   * Executes buildozer with the given commands piped to stdin. Returns {@code true} if buildozer
   * made changes, {@code false} if no changes were needed (buildozer exit code 3).
   */
  private static boolean executeBuildozer(
      CommandEnvironment env, BazelModTidyValue modTidyValue, CharSequence commands)
      throws InterruptedException, CommandException, IOException {
    try (var stdin = CharSource.wrap(commands).asByteSource(ISO_8859_1).openStream()) {
      new CommandBuilder(env.getClientEnv())
          .setWorkingDir(env.getWorkspace())
          .addArg(modTidyValue.buildozer().getPathString())
          .addArg("-f")
          .addArg("-")
          .build()
          .executeAsync(stdin, /* killSubprocessOnInterrupt= */ true)
          .get();
      return true;
    } catch (AbnormalTerminationException e) {
      if (e.getResult().terminationStatus().getRawExitCode() == 3) {
        // Buildozer exits with exit code 3 if it didn't make any changes.
        return false;
      }
      throw e;
    }
  }

  /**
   * Returns a user-friendly error message for an exception thrown during {@link executeBuildozer}.
   */
  private static String formatBuildozerError(Exception e) {
    String suffix = "";
    if (e instanceof AbnormalTerminationException ate) {
      suffix = ":\n" + new String(ate.getResult().getStderr(), ISO_8859_1);
    }
    return "Unexpected error while running buildozer: " + e.getMessage() + suffix;
  }

  private static BlazeCommandResult reportAndCreateTidyResult(
      CommandEnvironment env, BazelModTidyValue modTidyValue) {
    if (modTidyValue.errors().isEmpty()) {
      return BlazeCommandResult.success();
    } else {
      return reportAndCreateFailureResult(
          env,
          String.format(
              "Failed to process %d extension%s due to errors.",
              modTidyValue.errors().size(), modTidyValue.errors().size() == 1 ? "" : "s"),
          Code.ERROR_DURING_GRAPH_INSPECTION);
    }
  }

  /** Collects a list of {@link ModuleArg} into a set of {@link ModuleKey}s. */
  private static ImmutableSet<ModuleKey> moduleArgListToKeys(
      ImmutableList<ModuleArg> argList,
      ImmutableMap<String, ImmutableSet<ModuleKey>> modulesIndex,
      ImmutableMap<ModuleKey, AugmentedModule> depGraph,
      ImmutableMap<ModuleKey, RepositoryName> moduleKeyToCanonicalNames,
      ImmutableBiMap<String, ModuleKey> baseModuleDeps,
      ImmutableBiMap<String, ModuleKey> baseModuleUnusedDeps,
      boolean includeUnused)
      throws InvalidArgumentException {
    ImmutableSet.Builder<ModuleKey> allTargetKeys = new ImmutableSet.Builder<>();
    for (ModuleArg moduleArg : argList) {
      allTargetKeys.addAll(
          moduleArg.resolveToModuleKeys(
              modulesIndex,
              depGraph,
              moduleKeyToCanonicalNames,
              baseModuleDeps,
              baseModuleUnusedDeps,
              includeUnused,
              true));
    }
    return allTargetKeys.build();
  }

  private static ImmutableSortedSet<ModuleExtensionId> extensionArgListToIds(
      ImmutableList<ExtensionArg> args,
      ImmutableMap<String, ImmutableSet<ModuleKey>> modulesIndex,
      ImmutableMap<ModuleKey, AugmentedModule> depGraph,
      ImmutableMap<ModuleKey, RepositoryName> moduleKeyToCanonicalNames,
      ImmutableBiMap<String, ModuleKey> baseModuleDeps,
      ImmutableBiMap<String, ModuleKey> baseModuleUnusedDeps)
      throws InvalidArgumentException {
    ImmutableSortedSet.Builder<ModuleExtensionId> extensionsBuilder =
        new ImmutableSortedSet.Builder<>(ModuleExtensionId.LEXICOGRAPHIC_COMPARATOR);
    for (ExtensionArg arg : args) {
      extensionsBuilder.add(
          arg.resolveToExtensionId(
              modulesIndex,
              depGraph,
              moduleKeyToCanonicalNames,
              baseModuleDeps,
              baseModuleUnusedDeps));
    }
    return extensionsBuilder.build();
  }

  private ImmutableMap<String, Optional<ImmutableList<Version>>> fetchAllAvailableVersions(
      Map<String, Registry> registryByModule,
      ExecutorService executor,
      ExtendedEventHandler reporter)
      throws InterruptedException {
    var futures = ImmutableMap.<String, Future<Optional<ImmutableList<Version>>>>builder();
    for (var entry : registryByModule.entrySet()) {
      futures.put(
          entry.getKey(),
          executor.submit(
              () -> fetchAvailableVersions(entry.getKey(), entry.getValue(), reporter)));
    }
    var builder = ImmutableMap.<String, Optional<ImmutableList<Version>>>builder();
    for (var entry : futures.buildOrThrow().entrySet()) {
      try {
        builder.put(entry.getKey(), entry.getValue().get());
      } catch (ExecutionException e) {
        String msg =
            String.format(
                "Could not read metadata file for module %s: %s",
                entry.getKey(), e.getCause().getMessage());
        reporter.handle(Event.warn(msg));
        builder.put(entry.getKey(), Optional.empty());
      }
    }
    return builder.buildOrThrow();
  }

  private Optional<ImmutableList<Version>> fetchAvailableVersions(
      String moduleName, Registry registry, ExtendedEventHandler eventHandler)
      throws IOException, InterruptedException {
    try (SilentCloseable c =
        Profiler.instance()
            .profile(ProfilerTask.BZLMOD, () -> "getting available versions: " + moduleName)) {
      return registry.getAvailableVersions(moduleName, eventHandler, downloadManager);
    }
  }

  private static BlazeCommandResult reportAndCreateFailureResult(
      CommandEnvironment env, String message, Code detailedCode) {
    String fullMessage =
        switch (detailedCode) {
          case MISSING_ARGUMENTS, TOO_MANY_ARGUMENTS, INVALID_ARGUMENTS ->
              String.format(
                  "%s%s Type '%s help mod' for syntax and help.",
                  message, message.endsWith(".") ? "" : ".", env.getRuntime().getProductName());
          default -> message;
        };
    env.getReporter().handle(Event.error(fullMessage));
    return createFailureResult(fullMessage, detailedCode);
  }

  private static BlazeCommandResult createFailureResult(String message, Code detailedCode) {
    return BlazeCommandResult.detailedExitCode(
        DetailedExitCode.of(
            FailureDetail.newBuilder()
                .setModCommand(FailureDetails.ModCommand.newBuilder().setCode(detailedCode).build())
                .setMessage(message)
                .build()));
  }

  public static void dumpRepoMappings(List<RepositoryMappingValue> repoMappings, Writer writer)
      throws IOException {
    Gson gson = new GsonBuilder().disableHtmlEscaping().create();
    for (RepositoryMappingValue repoMapping : repoMappings) {
      JsonWriter jsonWriter = gson.newJsonWriter(writer);
      jsonWriter.beginObject();
      for (Entry<String, RepositoryName> entry :
          repoMapping.repositoryMapping().entries().entrySet()) {
        jsonWriter.name(entry.getKey());
        jsonWriter.value(entry.getValue().getName());
      }
      jsonWriter.endObject();
      writer.write('\n');
    }
    writer.flush();
  }
}
