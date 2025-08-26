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

import static com.google.devtools.build.lib.analysis.OutputGroupInfo.INTERNAL_SUFFIX;

import com.google.devtools.build.lib.actions.ActionConflictException;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.OutputGroupInfo;
import com.google.devtools.build.lib.analysis.PrerequisiteArtifacts;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesCollector;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesCollector.InstrumentationSpec;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesInfo;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.util.FileTypeSet;
import java.util.List;
import javax.annotation.Nullable;

/** ConfiguredTarget for "filegroup". */
public class Filegroup implements RuleConfiguredTargetFactory {

  /** Error message for output groups that are explicitly forbidden from filegroup reference. */
  public static final String ILLEGAL_OUTPUT_GROUP_ERROR =
      "Output group %s is not permitted for " + "reference in filegroups.";

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException, ActionConflictException {
    String outputGroupName = ruleContext.attributes().get("output_group", Type.STRING);
    if (outputGroupName.endsWith(INTERNAL_SUFFIX)) {
      ruleContext.throwWithAttributeError(
          "output_group", String.format(ILLEGAL_OUTPUT_GROUP_ERROR, outputGroupName));
    }

    NestedSet<Artifact> filesToBuild =
        outputGroupName.isEmpty()
            ? PrerequisiteArtifacts.nestedSet(ruleContext.getRulePrerequisitesCollection(), "srcs")
            : getArtifactsForOutputGroup(outputGroupName, ruleContext.getPrerequisites("srcs"));

    InstrumentedFilesInfo instrumentedFilesProvider =
        InstrumentedFilesCollector.collect(
            ruleContext,
            // Seems strange to have "srcs" in "dependency attributes" instead of "source
            // attributes", but that's correct behavior here because this rule just forwards
            // files, it doesn't process them. It doesn't know if the dependencies of the stuff
            // in srcs is a runtime dependency of its consumers or not. Consumers decide which
            // of the following is the case about a filegroup it depends on based on whether the
            // attribute the dependency is via is in the consumer's source attributes or
            // dependency attributes:
            // * If the filegroup contains coverage-relevant source files, it should be depended
            //   on via something in source attributes. The dependencies for actions which generate
            //   source files are generally not runtime dependencies.
            // * If the dependencies of the filegroup might be coverage-relevant source files (e.g.
            //   a binary target is included in filegroup's srcs and the filegroup target is
            //   included in some other target's data), it should be depended on via something in
            //   dependency attributes.
            new InstrumentationSpec(FileTypeSet.ANY_FILE).withDependencyAttributes("srcs", "data"),
            /* reportedToActualSources= */ NestedSetBuilder.create(Order.STABLE_ORDER));

    // If you're visiting a filegroup as data, then we also visit its data as data.
    var dataRunfilesBuilder =
        new Runfiles.Builder(ruleContext.getWorkspaceName()).addTransitiveArtifacts(filesToBuild);
    if (ruleContext
        .getConfiguration()
        .getOptions()
        .get(CoreOptions.class)
        .filegroupRunfilesForData) {
      // If you're visiting a filegroup as data, then we also visit its data as data.
      dataRunfilesBuilder.addRunfiles(ruleContext, RunfilesProvider.DATA_RUNFILES);
    } else {
      dataRunfilesBuilder.addDataDeps(ruleContext);
    }
    RunfilesProvider runfilesProvider =
        RunfilesProvider.withData(
            new Runfiles.Builder(ruleContext.getWorkspaceName())
                .addRunfiles(ruleContext, RunfilesProvider.DEFAULT_RUNFILES)
                .build(),
            dataRunfilesBuilder.build());

    RuleConfiguredTargetBuilder builder =
        new RuleConfiguredTargetBuilder(ruleContext)
            .addProvider(RunfilesProvider.class, runfilesProvider)
            .setFilesToBuild(filesToBuild)
            .setRunfilesSupport(null, getExecutable(filesToBuild))
            .addNativeDeclaredProvider(instrumentedFilesProvider);

    return builder.build();
  }

  /**
   * Returns the single Artifact from filesToBuild or {@code null} if there are multiple elements.
   */
  @Nullable
  private Artifact getExecutable(NestedSet<Artifact> filesToBuild) {
    return filesToBuild.isSingleton() ? filesToBuild.getSingleton() : null;
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
