// Copyright 2023 The Bazel Authors. All rights reserved.
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
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.NoBuildEvent;
import com.google.devtools.build.lib.analysis.NoBuildRequestFinishedEvent;
import com.google.devtools.build.lib.bazel.bzlmod.BazelFetchAllValue;
import com.google.devtools.build.lib.bazel.commands.RepositoryFetcher.RepositoryFetcherException;
import com.google.devtools.build.lib.bazel.commands.TargetFetcher.TargetFetcherException;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions;
import com.google.devtools.build.lib.buildtool.BuildResult;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.pkgcache.PackageOptions;
import com.google.devtools.build.lib.rules.repository.RepositoryDelegatorFunction;
import com.google.devtools.build.lib.rules.repository.RepositoryDirectoryValue;
import com.google.devtools.build.lib.runtime.BlazeCommand;
import com.google.devtools.build.lib.runtime.BlazeCommandResult;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.KeepGoingOption;
import com.google.devtools.build.lib.runtime.LoadingPhaseThreadsOption;
import com.google.devtools.build.lib.runtime.commands.TestCommand;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.FetchCommand.Code;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.skyframe.RepositoryMappingValue.RepositoryMappingResolutionException;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.InterruptedFailureDetails;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.skyframe.EvaluationContext;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.InMemoryGraph;
import com.google.devtools.build.skyframe.NodeEntry;
import com.google.devtools.build.skyframe.QueryableGraph.Reason;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingResult;
import java.io.IOException;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map.Entry;
import java.util.Objects;
import java.util.Queue;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * Fetches external repositories into a specified directory.
 *
 * <p>This command is used to fetch external repositories into a specified directory. It can be used
 * to fetch all external repositories, a specific list of repositories or the repositories needed to
 * build a specific list of targets.
 *
 * <p>The command is used to create a vendor directory that can be used to build the project
 * offline.
 */
@Command(
    name = VendorCommand.NAME,
    builds = true,
    inherits = {TestCommand.class},
    options = {
      VendorOptions.class,
      PackageOptions.class,
      KeepGoingOption.class,
      LoadingPhaseThreadsOption.class
    },
    allowResidue = true,
    usesConfigurationOptions = true,
    help = "resource:vendor.txt",
    shortDescription =
        "Fetches external repositories into a specific folder specified by the flag "
            + "--vendor_dir.")
public final class VendorCommand implements BlazeCommand {
  public static final String NAME = "vendor";

  @Override
  public void editOptions(OptionsParser optionsParser) {
    // We only need to inject these options with fetch target (when there is a residue)
    if (!optionsParser.getResidue().isEmpty()) {
      TargetFetcher.injectNoBuildOption(optionsParser);
    }
  }

  @Override
  public BlazeCommandResult exec(CommandEnvironment env, OptionsParsingResult options) {
    BlazeCommandResult invalidResult = validateOptions(env, options);
    if (invalidResult != null) {
      return invalidResult;
    }

    env.getEventBus()
        .post(
            new NoBuildEvent(
                env.getCommandName(),
                env.getCommandStartTime(),
                /* separateFinishedEvent= */ true,
                /* showProgress= */ true,
                env.getCommandId().toString()));

    // IS_VENDOR_COMMAND & VENDOR_DIR is already injected in "BazelRepositoryModule", we just need
    // to update this value for the delegator function to recognize this call is from VendorCommand
    env.getSkyframeExecutor()
        .injectExtraPrecomputedValues(
            ImmutableList.of(
                PrecomputedValue.injected(RepositoryDelegatorFunction.IS_VENDOR_COMMAND, true)));

    BlazeCommandResult result;
    VendorOptions vendorOptions = options.getOptions(VendorOptions.class);
    PathFragment vendorDirectory = options.getOptions(RepositoryOptions.class).vendorDirectory;
    LoadingPhaseThreadsOption threadsOption = options.getOptions(LoadingPhaseThreadsOption.class);
    try {
      if (!options.getResidue().isEmpty()) {
        result = vendorTargets(env, options, options.getResidue(), vendorDirectory);
      } else if (!vendorOptions.repos.isEmpty()) {
        result = vendorRepos(env, threadsOption, vendorOptions.repos, vendorDirectory);
      } else {
        result = vendorAll(env, threadsOption, vendorDirectory);
      }
    } catch (InterruptedException e) {
      return createFailedBlazeCommandResult(
          env.getReporter(), "Vendor interrupted: " + e.getMessage());
    } catch (IOException e) {
      return createFailedBlazeCommandResult(
          env.getReporter(), "Error while vendoring repos: " + e.getMessage());
    }

    env.getEventBus()
        .post(
            new NoBuildRequestFinishedEvent(
                result.getExitCode(), env.getRuntime().getClock().currentTimeMillis()));
    return result;
  }

  @Nullable
  private BlazeCommandResult validateOptions(CommandEnvironment env, OptionsParsingResult options) {
    if (!options.getOptions(BuildLanguageOptions.class).enableBzlmod) {
      return createFailedBlazeCommandResult(
          env.getReporter(),
          "Bzlmod has to be enabled for vendoring to work, run with --enable_bzlmod");
    }
    if (options.getOptions(RepositoryOptions.class).vendorDirectory == null) {
      return createFailedBlazeCommandResult(
          env.getReporter(),
          Code.OPTIONS_INVALID,
          "You cannot run vendor without specifying --vendor_dir");
    }
    if (!options.getOptions(PackageOptions.class).fetch) {
      return createFailedBlazeCommandResult(
          env.getReporter(), Code.OPTIONS_INVALID, "You cannot run vendor with --nofetch");
    }
    return null;
  }

  private BlazeCommandResult vendorAll(
      CommandEnvironment env, LoadingPhaseThreadsOption threadsOption, PathFragment vendorDirectory)
      throws InterruptedException, IOException {
    EvaluationContext evaluationContext =
        EvaluationContext.newBuilder()
            .setParallelism(threadsOption.threads)
            .setEventHandler(env.getReporter())
            .build();

    SkyKey fetchKey = BazelFetchAllValue.key(/* configureEnabled= */ false);
    EvaluationResult<SkyValue> evaluationResult =
        env.getSkyframeExecutor().prepareAndGet(ImmutableSet.of(fetchKey), evaluationContext);
    if (evaluationResult.hasError()) {
      Exception e = evaluationResult.getError().getException();
      return createFailedBlazeCommandResult(
          env.getReporter(),
          e != null ? e.getMessage() : "Unexpected error during fetching all external deps.");
    }

    BazelFetchAllValue fetchAllValue = (BazelFetchAllValue) evaluationResult.get(fetchKey);
    vendor(env, vendorDirectory, fetchAllValue.getReposToVendor());
    env.getReporter().handle(Event.info("All external dependencies vendored successfully."));
    return BlazeCommandResult.success();
  }

  private BlazeCommandResult vendorRepos(
      CommandEnvironment env,
      LoadingPhaseThreadsOption threadsOption,
      List<String> repos,
      PathFragment vendorDirectory)
      throws InterruptedException, IOException {
    ImmutableMap<RepositoryName, RepositoryDirectoryValue> repositoryNamesAndValues;
    try {
      repositoryNamesAndValues = RepositoryFetcher.fetchRepos(repos, env, threadsOption);
    } catch (RepositoryMappingResolutionException e) {
      return createFailedBlazeCommandResult(
          env.getReporter(), "Invalid repo name: " + e.getMessage(), e.getDetailedExitCode());
    } catch (RepositoryFetcherException e) {
      return createFailedBlazeCommandResult(env.getReporter(), e.getMessage());
    }

    // Split repos to found and not found, vendor found ones and report others
    ImmutableList.Builder<RepositoryName> reposToVendor = ImmutableList.builder();
    List<String> notFoundRepoErrors = new ArrayList<>();
    for (Entry<RepositoryName, RepositoryDirectoryValue> entry :
        repositoryNamesAndValues.entrySet()) {
      if (entry.getValue().repositoryExists()) {
        if (!entry.getValue().excludeFromVendoring()) {
          reposToVendor.add(entry.getKey());
        }
      } else {
        notFoundRepoErrors.add(entry.getValue().getErrorMsg());
      }
    }

    vendor(env, vendorDirectory, reposToVendor.build());
    if (!notFoundRepoErrors.isEmpty()) {
      return createFailedBlazeCommandResult(
          env.getReporter(), "Vendoring some repos failed with errors: " + notFoundRepoErrors);
    }
    env.getReporter().handle(Event.info("All requested repos vendored successfully."));
    return BlazeCommandResult.success();
  }

  private BlazeCommandResult vendorTargets(
      CommandEnvironment env,
      OptionsParsingResult options,
      List<String> targets,
      PathFragment vendorDirectory)
      throws InterruptedException, IOException {
    // Call fetch which runs build to have the targets graph and configuration set
    BuildResult buildResult;
    try {
      buildResult = TargetFetcher.fetchTargets(env, options, targets);
    } catch (TargetFetcherException e) {
      return createFailedBlazeCommandResult(
          env.getReporter(), Code.QUERY_EVALUATION_ERROR, e.getMessage());
    }

    // Traverse the graph created from build to collect repos and vendor them
    ImmutableList<SkyKey> targetKeys =
        buildResult.getActualTargets().stream()
            .map(
                target ->
                    ConfiguredTargetKey.builder()
                        .setConfigurationKey(target.getConfigurationKey())
                        .setLabel(target.getLabel())
                        .build())
            .collect(toImmutableList());
    InMemoryGraph inMemoryGraph = env.getSkyframeExecutor().getEvaluator().getInMemoryGraph();
    ImmutableSet<RepositoryName> reposToVendor = collectReposFromTargets(inMemoryGraph, targetKeys);

    vendor(env, vendorDirectory, reposToVendor.asList());
    env.getReporter()
        .handle(
            Event.info(
                "All external dependencies for the requested targets vendored successfully."));
    return BlazeCommandResult.success();
  }

  private ImmutableSet<RepositoryName> collectReposFromTargets(
      InMemoryGraph inMemoryGraph, ImmutableList<SkyKey> targetKeys) throws InterruptedException {
    ImmutableSet.Builder<RepositoryName> repos = ImmutableSet.builder();
    Queue<SkyKey> nodes = new ArrayDeque<>(targetKeys);
    Set<SkyKey> visited = new HashSet<>();
    while (!nodes.isEmpty()) {
      SkyKey key = nodes.remove();
      visited.add(key);
      NodeEntry nodeEntry = inMemoryGraph.get(null, Reason.VENDOR_EXTERNAL_REPOS, key);
      if (nodeEntry.getValue() instanceof RepositoryDirectoryValue repoDirValue
          && repoDirValue.repositoryExists()
          && !repoDirValue.excludeFromVendoring()) {
        repos.add((RepositoryName) key.argument());
      }
      for (SkyKey depKey : nodeEntry.getDirectDeps()) {
        if (!visited.contains(depKey)) {
          nodes.add(depKey);
        }
      }
    }
    return repos.build();
  }

  /**
   * Copies the fetched repos from the external cache into the vendor directory, unless the repo is
   * ignored or was already vendored and up-to-date
   */
  private void vendor(
      CommandEnvironment env,
      PathFragment vendorDirectory,
      ImmutableList<RepositoryName> reposToVendor)
      throws IOException {
    Path vendorPath =
        vendorDirectory.isAbsolute()
            ? env.getRuntime().getFileSystem().getPath(vendorDirectory)
            : env.getWorkspace().getRelative(vendorDirectory);
    Path externalPath =
        env.getDirectories()
            .getOutputBase()
            .getRelative(LabelConstants.EXTERNAL_REPOSITORY_LOCATION);

    if (!vendorPath.exists()) {
      vendorPath.createDirectory();
    }

    env.getReporter().handle(Event.info("Vendoring ..."));

    // Update "out-of-date" repos under the vendor directory
    for (RepositoryName repo : reposToVendor) {
      if (!isRepoUpToDate(repo.getName(), vendorPath, externalPath)) {
        Path repoUnderVendor = vendorPath.getRelative(repo.getName());
        if (!repoUnderVendor.exists()) {
          repoUnderVendor.createDirectory();
        }
        FileSystemUtils.copyTreesBelow(
            externalPath.getRelative(repo.getName()), repoUnderVendor, Symlinks.NOFOLLOW);
        FileSystemUtils.copyFile(
            externalPath.getChild("@" + repo.getName() + ".marker"),
            vendorPath.getChild("@" + repo.getName() + ".marker"));
      }
    }
  }

  /**
   * Returns whether the repo under vendor needs to be updated by comparing its marker file with the
   * one under /external
   */
  private boolean isRepoUpToDate(String repoName, Path vendorPath, Path externalPath)
      throws IOException {
    Path vendorMarkerFile = vendorPath.getChild("@" + repoName + ".marker");
    if (!vendorMarkerFile.exists()) {
      return false;
    }

    // Since this runs after fetching repos, its guaranteed that the marker files
    // under $OUTPUT_BASE/external are up-to-date. We just need to compare it against the marker
    // under vendor.
    Path externalMarkerFile = externalPath.getChild("@" + repoName + ".marker");
    String vendorMarkerContent = FileSystemUtils.readContent(vendorMarkerFile, UTF_8);
    String externalMarkerContent = FileSystemUtils.readContent(externalMarkerFile, UTF_8);
    return Objects.equals(vendorMarkerContent, externalMarkerContent);
  }

  private static BlazeCommandResult createFailedBlazeCommandResult(
      Reporter reporter, Code fetchCommandCode, String message) {
    return createFailedBlazeCommandResult(
        reporter,
        message,
        DetailedExitCode.of(
            FailureDetail.newBuilder()
                .setMessage(message)
                .setFetchCommand(
                    FailureDetails.FetchCommand.newBuilder().setCode(fetchCommandCode).build())
                .build()));
  }

  private static BlazeCommandResult createFailedBlazeCommandResult(
      Reporter reporter, String errorMessage) {
    return createFailedBlazeCommandResult(
        reporter, errorMessage, InterruptedFailureDetails.detailedExitCode(errorMessage));
  }

  private static BlazeCommandResult createFailedBlazeCommandResult(
      Reporter reporter, String message, DetailedExitCode exitCode) {
    reporter.handle(Event.error(message));
    return BlazeCommandResult.detailedExitCode(exitCode);
  }
}
