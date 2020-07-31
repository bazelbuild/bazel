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

package com.google.devtools.build.lib.rules.filegroup;

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.devtools.build.lib.analysis.OutputGroupInfo.INTERNAL_SUFFIX;

import com.google.devtools.build.lib.actions.Actions;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.CompilationHelper;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.MiddlemanProvider;
import com.google.devtools.build.lib.analysis.OutputGroupInfo;
import com.google.devtools.build.lib.analysis.PrerequisiteArtifacts;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.TransitionMode;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesCollector;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesCollector.InstrumentationSpec;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesInfo;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.util.FileTypeSet;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.List;

/**
 * ConfiguredTarget for "filegroup".
 */
public class Filegroup implements RuleConfiguredTargetFactory {

  /** Error message for output groups that are explicitly blacklisted for filegroup reference. */
  public static final String ILLEGAL_OUTPUT_GROUP_ERROR =
      "Output group %s is not permitted for " + "reference in filegroups.";

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException, ActionConflictException {
    String outputGroupName = ruleContext.attributes().get("output_group", Type.STRING);
    BuildConfiguration configuration = checkNotNull(ruleContext.getConfiguration());
    if (outputGroupName.endsWith(INTERNAL_SUFFIX)) {
      ruleContext.throwWithAttributeError(
          "output_group", String.format(ILLEGAL_OUTPUT_GROUP_ERROR, outputGroupName));
    }

    NestedSet<Artifact> filesToBuild =
        outputGroupName.isEmpty()
            ? PrerequisiteArtifacts.nestedSet(ruleContext, "srcs", TransitionMode.TARGET)
            : getArtifactsForOutputGroup(
                outputGroupName, ruleContext.getPrerequisites("srcs", TransitionMode.TARGET));

    InstrumentedFilesInfo instrumentedFilesProvider =
        InstrumentedFilesCollector.collectTransitive(
            ruleContext,
            // Seems strange to have "srcs" in "dependency attributes" instead of "source
            // attributes", but that's correct behavior here because:
            // 1. This rule is essentially forwarding, it has no idea how the stuff in srcs is used.
            //    Thus, it needs to look at any dependencies transitively via
            //    InstrumentedFilesProvider.
            // 2. This rule doesn't _process_ any source files. The rule which does process the
            //    source files in filegroup.srcs will include those files in its inputs and in its
            //    InstrumentedFileProvider the same way, via FileProvider. This ensures that when
            //    --instrumentation_filter says a rule's sources should be instrumented for coverage
            //    data collection, it also says all of those sources should be included in the
            //    coverage manifest.
            // Previously, this would have needed to include "srcs" in "source attributes" anyways,
            // since it might have been _consumed_ by a rule using the legacy InstrumentationSpec.
            // In that case, since filegroup provided InstrumentedFilesProvider, the legacy
            // consumer would never try to gather filegroup's instrumented sources via FileProvider.
            new InstrumentationSpec(FileTypeSet.ANY_FILE)
                .withDeprecatedSourceOrDependencyAttributes("srcs", "deps", "data")
                .withDependencyAttributes("srcs", "data"),
            /* reportedToActualSources= */ NestedSetBuilder.create(Order.STABLE_ORDER));

    RunfilesProvider runfilesProvider =
        RunfilesProvider.withData(
            new Runfiles.Builder(
                    ruleContext.getWorkspaceName(), configuration.legacyExternalRunfiles())
                .addRunfiles(ruleContext, RunfilesProvider.DEFAULT_RUNFILES)
                .build(),
            // If you're visiting a filegroup as data, then we also visit its data as data.
            new Runfiles.Builder(
                    ruleContext.getWorkspaceName(), configuration.legacyExternalRunfiles())
                .addTransitiveArtifacts(filesToBuild)
                .addDataDeps(ruleContext)
                .build());

    RuleConfiguredTargetBuilder builder =
        new RuleConfiguredTargetBuilder(ruleContext)
            .addProvider(RunfilesProvider.class, runfilesProvider)
            .setFilesToBuild(filesToBuild)
            .setRunfilesSupport(null, getExecutable(filesToBuild))
            .addNativeDeclaredProvider(instrumentedFilesProvider)
            .addProvider(
                FilegroupPathProvider.class,
                new FilegroupPathProvider(getFilegroupPath(ruleContext)));

    if (configuration.enableAggregatingMiddleman()) {
      builder.addProvider(
          MiddlemanProvider.class,
          new MiddlemanProvider(
              CompilationHelper.getAggregatingMiddleman(
                  ruleContext, Actions.escapeLabel(ruleContext.getLabel()), filesToBuild)));
    }
    return builder.build();
  }

  /**
   * Returns the single Artifact from filesToBuild or {@code null} if there are multiple elements.
   */
  private Artifact getExecutable(NestedSet<Artifact> filesToBuild) {
    return filesToBuild.isSingleton() ? filesToBuild.getSingleton() : null;
  }

  private PathFragment getFilegroupPath(RuleContext ruleContext) {
    String attr = ruleContext.attributes().get("path", Type.STRING);
    if (attr.isEmpty()) {
      return PathFragment.EMPTY_FRAGMENT;
    } else {
      return ruleContext.getLabel().getPackageIdentifier().getSourceRoot().getRelative(attr);
    }
  }

  /** Returns the artifacts from the given targets that are members of the given output group. */
  private static NestedSet<Artifact> getArtifactsForOutputGroup(
      String outputGroupName, List<? extends TransitiveInfoCollection> deps) {
    NestedSetBuilder<Artifact> result = NestedSetBuilder.stableOrder();

    for (TransitiveInfoCollection dep : deps) {
      OutputGroupInfo outputGroupInfo = OutputGroupInfo.get(dep);
      if (outputGroupInfo != null) {
        result.addTransitive(outputGroupInfo.getOutputGroup(outputGroupName));
      }
    }

    return result.build();
  }
}
