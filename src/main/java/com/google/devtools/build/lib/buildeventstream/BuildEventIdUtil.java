// Copyright 2016 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.buildeventstream;

import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId.ActionCompletedId;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId.ConfigurationId;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.List;
import javax.annotation.Nullable;
import javax.annotation.concurrent.Immutable;

/**
 * Utilities for working with {@link BuildEventId}.
 *
 * <p>Since event identifiers need to be created before the actual event, the event IDs are highly
 * structured so that equal identifiers can easily be generated. The main way of pregenerating event
 * identifiers that do not accidentally coincide is by providing a target or a target pattern;
 * therefore, those (if provided) are made specially visible.
 */
@Immutable
public final class BuildEventIdUtil {
  private BuildEventIdUtil() {}

  public static BuildEventId unknownBuildEventId(String details) {
    BuildEventId.UnknownBuildEventId id =
        BuildEventId.UnknownBuildEventId.newBuilder().setDetails(details).build();
    return BuildEventId.newBuilder().setUnknown(id).build();
  }

  public static BuildEventId progressId(int count) {
    BuildEventId.ProgressId id = BuildEventId.ProgressId.newBuilder().setOpaqueCount(count).build();
    return BuildEventId.newBuilder().setProgress(id).build();
  }

  public static BuildEventId buildStartedId() {
    BuildEventId.BuildStartedId startedId = BuildEventId.BuildStartedId.getDefaultInstance();
    return BuildEventId.newBuilder().setStarted(startedId).build();
  }

  public static BuildEventId unstructuredCommandlineId() {
    BuildEventId.UnstructuredCommandLineId commandLineId =
        BuildEventId.UnstructuredCommandLineId.getDefaultInstance();
    return BuildEventId.newBuilder().setUnstructuredCommandLine(commandLineId).build();
  }

  public static BuildEventId structuredCommandlineId(String commandLineLabel) {
    BuildEventId.StructuredCommandLineId commandLineId =
        BuildEventId.StructuredCommandLineId.newBuilder()
            .setCommandLineLabel(commandLineLabel)
            .build();
    return BuildEventId.newBuilder().setStructuredCommandLine(commandLineId).build();
  }

  public static BuildEventId optionsParsedId() {
    BuildEventId.OptionsParsedId optionsParsedId =
        BuildEventId.OptionsParsedId.getDefaultInstance();
    return BuildEventId.newBuilder().setOptionsParsed(optionsParsedId).build();
  }

  public static BuildEventId workspaceStatusId() {
    return BuildEventId.newBuilder()
        .setWorkspaceStatus(BuildEventId.WorkspaceStatusId.getDefaultInstance())
        .build();
  }

  public static BuildEventId buildMetadataId() {
    BuildEventId.BuildMetadataId buildMetadataId =
        BuildEventId.BuildMetadataId.getDefaultInstance();
    return BuildEventId.newBuilder().setBuildMetadata(buildMetadataId).build();
  }

  public static BuildEventId workspaceConfigId() {
    BuildEventId.WorkspaceConfigId workspaceConfigId =
        BuildEventId.WorkspaceConfigId.getDefaultInstance();
    return BuildEventId.newBuilder().setWorkspace(workspaceConfigId).build();
  }

  static BuildEventId fetchId(String url) {
    BuildEventId.FetchId fetchId = BuildEventId.FetchId.newBuilder().setUrl(url).build();
    return BuildEventId.newBuilder().setFetch(fetchId).build();
  }

  public static BuildEventId configurationId(String id) {
    BuildEventId.ConfigurationId configurationId =
        BuildEventId.ConfigurationId.newBuilder().setId(id).build();
    return BuildEventId.newBuilder().setConfiguration(configurationId).build();
  }

  public static BuildEventId nullConfigurationId() {
    return configurationId("none");
  }

  private static BuildEventId targetPatternExpanded(List<String> targetPattern, boolean skipped) {
    BuildEventId.PatternExpandedId patternId =
        BuildEventId.PatternExpandedId.newBuilder().addAllPattern(targetPattern).build();
    BuildEventId.Builder builder = BuildEventId.newBuilder();
    if (skipped) {
      builder.setPatternSkipped(patternId);
    } else {
      builder.setPattern(patternId);
    }
    return builder.build();
  }

  public static BuildEventId targetPatternExpanded(List<String> targetPattern) {
    return targetPatternExpanded(targetPattern, false);
  }

  public static BuildEventId targetPatternSkipped(List<String> targetPattern) {
    return targetPatternExpanded(targetPattern, true);
  }

  public static BuildEventId targetConfigured(Label label) {
    BuildEventId.TargetConfiguredId configuredId =
        BuildEventId.TargetConfiguredId.newBuilder().setLabel(label.toString()).build();
    return BuildEventId.newBuilder().setTargetConfigured(configuredId).build();
  }

  public static BuildEventId aspectConfigured(Label label, String aspect) {
    BuildEventId.TargetConfiguredId configuredId =
        BuildEventId.TargetConfiguredId.newBuilder()
            .setLabel(label.toString())
            .setAspect(aspect)
            .build();
    return BuildEventId.newBuilder().setTargetConfigured(configuredId).build();
  }

  public static BuildEventId targetCompleted(Label target, BuildEventId configuration) {
    BuildEventId.ConfigurationId configId = configuration.getConfiguration();
    BuildEventId.TargetCompletedId targetId =
        BuildEventId.TargetCompletedId.newBuilder()
            .setLabel(target.toString())
            .setConfiguration(configId)
            .build();
    return BuildEventId.newBuilder().setTargetCompleted(targetId).build();
  }

  public static BuildEventId configuredLabelId(Label label, BuildEventId configuration) {
    BuildEventId.ConfigurationId configId = configuration.getConfiguration();
    BuildEventId.ConfiguredLabelId labelId =
        BuildEventId.ConfiguredLabelId.newBuilder()
            .setLabel(label.toString())
            .setConfiguration(configId)
            .build();
    return BuildEventId.newBuilder().setConfiguredLabel(labelId).build();
  }

  public static BuildEventId unconfiguredLabelId(Label label) {
    BuildEventId.UnconfiguredLabelId labelId =
        BuildEventId.UnconfiguredLabelId.newBuilder().setLabel(label.toString()).build();
    return BuildEventId.newBuilder().setUnconfiguredLabel(labelId).build();
  }

  public static BuildEventId aspectCompleted(
      Label target, BuildEventId configuration, String aspect) {
    BuildEventId.ConfigurationId configId = configuration.getConfiguration();
    BuildEventId.TargetCompletedId targetId =
        BuildEventId.TargetCompletedId.newBuilder()
            .setLabel(target.toString())
            .setConfiguration(configId)
            .setAspect(aspect)
            .build();
    return BuildEventId.newBuilder().setTargetCompleted(targetId).build();
  }

  public static BuildEventId actionCompleted(PathFragment path) {
    return actionCompleted(path, null, null);
  }

  public static BuildEventId actionCompleted(
      PathFragment path, @Nullable Label label, @Nullable String configurationChecksum) {
    ActionCompletedId.Builder actionId =
        ActionCompletedId.newBuilder().setPrimaryOutput(path.toString());
    if (label != null) {
      actionId.setLabel(label.toString());
    }
    if (configurationChecksum != null) {
      actionId.setConfiguration(ConfigurationId.newBuilder().setId(configurationChecksum));
    }
    return BuildEventId.newBuilder().setActionCompleted(actionId).build();
  }

  public static BuildEventId fromArtifactGroupName(String name) {
    BuildEventId.NamedSetOfFilesId namedSetId =
        BuildEventId.NamedSetOfFilesId.newBuilder().setId(name).build();
    return BuildEventId.newBuilder().setNamedSet(namedSetId).build();
  }

  public static BuildEventId testResult(
      Label target, Integer run, Integer shard, Integer attempt, BuildEventId configuration) {
    BuildEventId.ConfigurationId configId = configuration.getConfiguration();
    BuildEventId.TestResultId resultId =
        BuildEventId.TestResultId.newBuilder()
            .setLabel(target.toString())
            .setConfiguration(configId)
            .setRun(run + 1)
            .setShard(shard + 1)
            .setAttempt(attempt)
            .build();
    return BuildEventId.newBuilder().setTestResult(resultId).build();
  }

  public static BuildEventId testResult(
      Label target, Integer run, Integer shard, BuildEventId configuration) {
    return testResult(target, run, shard, 1, configuration);
  }

  public static BuildEventId testSummary(Label target, BuildEventId configuration) {
    BuildEventId.ConfigurationId configId = configuration.getConfiguration();
    BuildEventId.TestSummaryId summaryId =
        BuildEventId.TestSummaryId.newBuilder()
            .setLabel(target.toString())
            .setConfiguration(configId)
            .build();
    return BuildEventId.newBuilder().setTestSummary(summaryId).build();
  }

  public static BuildEventId targetSummary(Label target, BuildEventId configuration) {
    BuildEventId.ConfigurationId configId = configuration.getConfiguration();
    BuildEventId.TargetSummaryId summaryId =
        BuildEventId.TargetSummaryId.newBuilder()
            .setLabel(target.toString())
            .setConfiguration(configId)
            .build();
    return BuildEventId.newBuilder().setTargetSummary(summaryId).build();
  }

  public static BuildEventId buildFinished() {
    BuildEventId.BuildFinishedId finishedId = BuildEventId.BuildFinishedId.getDefaultInstance();
    return BuildEventId.newBuilder().setBuildFinished(finishedId).build();
  }

  public static BuildEventId buildToolLogs() {
    return BuildEventId.newBuilder()
        .setBuildToolLogs(BuildEventId.BuildToolLogsId.getDefaultInstance())
        .build();
  }

  public static BuildEventId buildMetrics() {
    return BuildEventId.newBuilder()
        .setBuildMetrics(BuildEventId.BuildMetricsId.getDefaultInstance())
        .build();
  }

  public static BuildEventId convenienceSymlinksIdentifiedId() {
    return BuildEventId.newBuilder()
        .setConvenienceSymlinksIdentified(
            BuildEventId.ConvenienceSymlinksIdentifiedId.getDefaultInstance())
        .build();
  }
}
