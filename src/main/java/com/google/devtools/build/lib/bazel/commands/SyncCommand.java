// Copyright 2018 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.NoBuildEvent;
import com.google.devtools.build.lib.analysis.NoBuildRequestFinishedEvent;
import com.google.devtools.build.lib.bazel.repository.RepositoryOrderEvent;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.pkgcache.PackageCacheOptions;
import com.google.devtools.build.lib.rules.repository.RepositoryDelegatorFunction;
import com.google.devtools.build.lib.rules.repository.RepositoryDirectoryValue;
import com.google.devtools.build.lib.runtime.BlazeCommand;
import com.google.devtools.build.lib.runtime.BlazeCommandResult;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.KeepGoingOption;
import com.google.devtools.build.lib.runtime.LoadingPhaseThreadsOption;
import com.google.devtools.build.lib.skyframe.PackageLookupValue;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.skyframe.WorkspaceFileValue;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.EvaluationContext;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingResult;
import java.util.HashSet;
import java.util.Set;

/** Syncs all repositories specified in the workspace file */
@Command(
    name = SyncCommand.NAME,
    options = {PackageCacheOptions.class, KeepGoingOption.class, LoadingPhaseThreadsOption.class},
    help = "resource:sync.txt",
    shortDescription = "Syncs all repositories specified in the workspace file",
    allowResidue = false)
public final class SyncCommand implements BlazeCommand {
  public static final String NAME = "sync";

  static final ImmutableSet<String> WHITELISTED_NATIVE_RULES =
      ImmutableSet.<String>of("local_repository");

  @Override
  public void editOptions(OptionsParser optionsParser) {}

  private static void reportError(CommandEnvironment env, EvaluationResult<SkyValue> value) {
    if (value.getError().getException() != null) {
      env.getReporter().handle(Event.error(value.getError().getException().getMessage()));
    } else {
      env.getReporter().handle(Event.error(value.getError().toString()));
    }
  }

  @Override
  public BlazeCommandResult exec(CommandEnvironment env, OptionsParsingResult options) {
    ExitCode exitCode = ExitCode.SUCCESS;

    try {
      env.getReporter()
          .post(
              new NoBuildEvent(
                  env.getCommandName(),
                  env.getCommandStartTime(),
                  true,
                  true,
                  env.getCommandId().toString()));
      env.setupPackageCache(options, env.getRuntime().getDefaultsPackageContent());
      SkyframeExecutor skyframeExecutor = env.getSkyframeExecutor();
      skyframeExecutor.injectExtraPrecomputedValues(
          ImmutableList.of(
              PrecomputedValue.injected(
                  RepositoryDelegatorFunction.DEPENDENCY_FOR_UNCONDITIONAL_FETCHING,
                  env.getCommandId().toString())));

      // Obtain the key for the top-level WORKSPACE file
      SkyKey packageLookupKey = PackageLookupValue.key(Label.EXTERNAL_PACKAGE_IDENTIFIER);
      LoadingPhaseThreadsOption threadsOption = options.getOptions(LoadingPhaseThreadsOption.class);
      EvaluationContext evaluationContext =
          EvaluationContext.newBuilder()
              .setNumThreads(threadsOption.threads)
              .setEventHander(env.getReporter())
              .build();
      EvaluationResult<SkyValue> packageLookupValue =
          skyframeExecutor.prepareAndGet(ImmutableSet.of(packageLookupKey), evaluationContext);
      if (packageLookupValue.hasError()) {
        reportError(env, packageLookupValue);
        env.getReporter()
            .post(
                new NoBuildRequestFinishedEvent(
                    ExitCode.ANALYSIS_FAILURE, env.getRuntime().getClock().currentTimeMillis()));
        return BlazeCommandResult.exitCode(ExitCode.ANALYSIS_FAILURE);
      }
      RootedPath workspacePath =
          ((PackageLookupValue) packageLookupValue.get(packageLookupKey))
              .getRootedPath(Label.EXTERNAL_PACKAGE_IDENTIFIER);
      SkyKey workspace = WorkspaceFileValue.key(workspacePath);

      // read and evaluate the WORKSPACE file to its end
      ImmutableList.Builder<String> repositoryOrder = new ImmutableList.Builder<>();
      Set<String> namesSeen = new HashSet<>();
      WorkspaceFileValue fileValue = null;
      while (workspace != null) {
        EvaluationResult<SkyValue> value =
            skyframeExecutor.prepareAndGet(ImmutableSet.of(workspace), evaluationContext);
        if (value.hasError()) {
          reportError(env, value);
          env.getReporter()
              .post(
                  new NoBuildRequestFinishedEvent(
                      ExitCode.ANALYSIS_FAILURE, env.getRuntime().getClock().currentTimeMillis()));
          return BlazeCommandResult.exitCode(ExitCode.ANALYSIS_FAILURE);
        }
        fileValue = (WorkspaceFileValue) value.get(workspace);
        for (Rule rule : fileValue.getPackage().getTargets(Rule.class)) {
          String name = rule.getName();
          if (!namesSeen.contains(name)) {
            repositoryOrder.add(name);
            namesSeen.add(name);
          }
        }
        workspace = fileValue.next();
      }
      env.getReporter().post(new RepositoryOrderEvent(repositoryOrder.build()));

      // take all skylark workspace rules and get their values
      ImmutableSet.Builder<SkyKey> repositoriesToFetch = new ImmutableSet.Builder<>();
      for (Rule rule : fileValue.getPackage().getTargets(Rule.class)) {
        if (shouldSync(rule)) {
          // TODO(aehlig): avoid the detour of serializing and then parsing the repository name
          try {
            repositoriesToFetch.add(
                RepositoryDirectoryValue.key(RepositoryName.create("@" + rule.getName())));
          } catch (LabelSyntaxException e) {
            env.getReporter()
                .handle(
                    Event.error(
                        "Internal error queuing "
                            + rule.getName()
                            + " to fetch: "
                            + e.getMessage()));
            env.getReporter()
                .post(
                    new NoBuildRequestFinishedEvent(
                        ExitCode.BLAZE_INTERNAL_ERROR,
                        env.getRuntime().getClock().currentTimeMillis()));
            return BlazeCommandResult.exitCode(ExitCode.BLAZE_INTERNAL_ERROR);
          }
        }
      }
      EvaluationResult<SkyValue> fetchValue;
      fetchValue = skyframeExecutor.prepareAndGet(repositoriesToFetch.build(), evaluationContext);
      if (fetchValue.hasError()) {
        reportError(env, fetchValue);
        exitCode = ExitCode.ANALYSIS_FAILURE;
      }
    } catch (InterruptedException e) {
      exitCode = ExitCode.INTERRUPTED;
    } catch (AbruptExitException e) {
      env.getReporter().handle(Event.error(e.getMessage()));
      exitCode = ExitCode.LOCAL_ENVIRONMENTAL_ERROR;
    }
    env.getReporter()
        .post(
            new NoBuildRequestFinishedEvent(
                exitCode, env.getRuntime().getClock().currentTimeMillis()));
    return BlazeCommandResult.exitCode(exitCode);
  }

  private static boolean shouldSync(Rule rule) {
    if (!rule.getRuleClassObject().getWorkspaceOnly()) {
      // We should only sync workspace rules
      return false;
    }
    if (rule.getRuleClassObject().isSkylark()) {
      // Skylark rules are all whitelisted
      return true;
    }
    return WHITELISTED_NATIVE_RULES.contains(rule.getRuleClassObject().getName());
  }
}
