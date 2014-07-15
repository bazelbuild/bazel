// Copyright 2014 Google Inc. All rights reserved.
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

package com.google.devtools.build.lib.view.test;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.NotifyOnActionCacheHit;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.view.FileProvider;
import com.google.devtools.build.lib.view.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.view.RuleContext;
import com.google.devtools.build.lib.view.TransitiveInfoCollection;
import com.google.devtools.build.lib.view.Util;
import com.google.devtools.build.lib.view.actions.CommandLine;
import com.google.devtools.build.lib.view.actions.SpawnAction;
import com.google.devtools.build.lib.view.config.BuildConfiguration;

import java.util.Collection;

import javax.annotation.Nullable;

/**
 * Generates baseline (empty) coverage for the given non-test target.
 */
public class BaselineCoverageAction extends SpawnAction implements NotifyOnActionCacheHit {

  /**
   * Helper interface for language specific processing hooks.
   */
  public interface LanguageHelper {
    /**
     * A hook function that implements language specific customizations. Such an implementation can
     * add extra artifacts to the inputs and set language specific args for lcov_merger.
     */
    void languageHook(
        RuleContext ruleContext,
        Collection<Artifact> metadataFiles,
        NestedSetBuilder<Artifact> inputs,
        StringBuilder langArgs);
  }

  /**
   * Helper interface to construct the tool command line that generates coverage data.
   */
  public interface CommandConstructor {
    /**
     * Returns a command that will generate baseline coverage data.
     *
     * @param configuration BuildConfiguration, giving access to the shell.
     * @param output Artifact for baseline_coverage.data
     * @param manifest Artifact for the input source file.
     * @param langArgs Arguments to append to the command-line.
     */
    CommandLine get(BuildConfiguration configuration, Artifact output,
      Artifact manifest, String langArgs);
  };

  private BaselineCoverageAction(ActionOwner owner, Iterable<Artifact> inputs, Artifact output,
      BuildConfiguration configuration, CommandLine commandLine) {
    super(owner, inputs, ImmutableList.of(output), configuration, DEFAULT_RESOURCE_SET,
        commandLine, configuration.getDefaultShellEnvironment(),
        "Generating baseline coverage data for " + owner.getLabel(), "BaselineCoverage");
  }

  @Override
  protected void internalExecute(
      ActionExecutionContext actionExecutionContext) throws ExecException, InterruptedException {
    super.internalExecute(actionExecutionContext);
    notifyAboutBaselineCoverage(actionExecutionContext.getExecutor().getEventBus());
  }

  @Override
  public void actionCacheHit(Executor executor) {
    notifyAboutBaselineCoverage(executor.getEventBus());
  }

  /**
   * Notify interested parties about new baseline coverage data.
   */
  private void notifyAboutBaselineCoverage(EventBus eventBus) {
    Artifact output = Iterables.getOnlyElement(getOutputs());
    String ownerString = Label.print(getOwner().getLabel());
    eventBus.post(new BaselineCoverageResult(output, ownerString));
  }

  /**
   * Returns collection of baseline coverage artifacts associated with the given target.
   * Will always return 0 or 1 elements.
   */
  public static ImmutableList<Artifact> getBaselineCoverageArtifacts(RuleContext ruleContext,
      Iterable<Artifact> instrumentedFiles, Iterable<Artifact> instrumentationMetadataFiles,
      Iterable<Artifact> filestoRun,
      CommandConstructor commandConstructor, @Nullable LanguageHelper helper) {
    // Create instrumented file manifest.
    final Collection<Artifact> metadataFiles =
        ImmutableList.copyOf(instrumentationMetadataFiles);

    if (metadataFiles.isEmpty()) {
      // Since there are no instrumentation metadata files, baseline coverage will be empty.
      // So skip it altogether.
      return ImmutableList.of();
    }
    BuildConfiguration configuration = ruleContext.getConfiguration();

    // Baseline coverage artifacts will still go into "testlogs" directory.
    final Artifact coverageData = ruleContext.getAnalysisEnvironment().getDerivedArtifact(
        Util.getWorkspaceRelativePath(ruleContext.getTarget()).getChild("baseline_coverage.dat"),
        configuration.getTestLogsDirectory());

    Artifact instrumentedFileManifest =
        InstrumentedFileManifestAction.getInstrumentedFileManifest(ruleContext, filestoRun,
            ImmutableList.copyOf(instrumentedFiles), metadataFiles);

    // Create input list for our action. Basically it should contain everything that lcov_merger
    // might need to generate empty LCOV file with proper line information inside it.
    NestedSetBuilder<Artifact> inputs = NestedSetBuilder.stableOrder();
    for (TransitiveInfoCollection dep :
        ruleContext.getPrerequisites(":baseline_coverage_common", Mode.HOST)) {
      inputs.addTransitive(dep.getProvider(FileProvider.class).getFilesToBuild());
    }

    // Add language-specific parts, if necessary.
    StringBuilder langArgs = new StringBuilder();
    if (helper != null) {
      helper.languageHook(ruleContext, metadataFiles, inputs, langArgs);
    }

    inputs.add(instrumentedFileManifest);
    inputs.addAll(metadataFiles);

    CommandLine commandLine = commandConstructor.get(
        configuration, coverageData, instrumentedFileManifest, langArgs.toString());
    ruleContext.getAnalysisEnvironment().registerAction(new BaselineCoverageAction(
        ruleContext.getActionOwner(), inputs.build(), coverageData,
        ruleContext.getConfiguration(), commandLine));

    return ImmutableList.of(coverageData);
  }

}
