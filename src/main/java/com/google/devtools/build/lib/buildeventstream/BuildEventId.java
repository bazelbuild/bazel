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

import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId.ActionCompletedId;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId.ConfigurationId;
import com.google.devtools.build.lib.causes.Cause;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.protobuf.TextFormat;
import java.io.Serializable;
import java.util.List;
import java.util.Objects;
import javax.annotation.Nullable;
import javax.annotation.concurrent.Immutable;

/**
 * Class of identifiers for publically posted events.
 *
 * <p>Since event identifiers need to be created before the actual event, the event IDs are highly
 * structured so that equal identifiers can easily be generated. The main way of pregenerating event
 * identifiers that do not accidentally coincide is by providing a target or a target pattern;
 * therefore, those (if provided) are made specially visible.
 */
@AutoCodec
@Immutable
public final class BuildEventId implements Serializable {
  private final BuildEventStreamProtos.BuildEventId protoid;

  @AutoCodec.VisibleForSerialization
  BuildEventId(BuildEventStreamProtos.BuildEventId protoid) {
    this.protoid = protoid;
  }

  @Override
  public int hashCode() {
    return Objects.hash(protoid);
  }

  @Override
  public boolean equals(Object other) {
    if (other == null || !other.getClass().equals(getClass())) {
      return false;
    }
    BuildEventId that = (BuildEventId) other;
    return Objects.equals(this.protoid, that.protoid);
  }

  @Override
  public String toString() {
    return "BuildEventId {" + TextFormat.printToString(protoid) + "}";
  }

  public BuildEventStreamProtos.BuildEventId asStreamProto() {
    return protoid;
  }

  public static BuildEventId unknownBuildEventId(String details) {
    BuildEventStreamProtos.BuildEventId.UnknownBuildEventId id =
        BuildEventStreamProtos.BuildEventId.UnknownBuildEventId.newBuilder()
            .setDetails(details)
            .build();
    return new BuildEventId(
        BuildEventStreamProtos.BuildEventId.newBuilder().setUnknown(id).build());
  }

  public static BuildEventId progressId(int count) {
    BuildEventStreamProtos.BuildEventId.ProgressId id =
        BuildEventStreamProtos.BuildEventId.ProgressId.newBuilder().setOpaqueCount(count).build();
    return new BuildEventId(
        BuildEventStreamProtos.BuildEventId.newBuilder().setProgress(id).build());
  }

  public static BuildEventId buildStartedId() {
    BuildEventStreamProtos.BuildEventId.BuildStartedId startedId =
        BuildEventStreamProtos.BuildEventId.BuildStartedId.getDefaultInstance();
    return new BuildEventId(
        BuildEventStreamProtos.BuildEventId.newBuilder().setStarted(startedId).build());
  }

  public static BuildEventId unstructuredCommandlineId() {
    BuildEventStreamProtos.BuildEventId.UnstructuredCommandLineId commandLineId =
        BuildEventStreamProtos.BuildEventId.UnstructuredCommandLineId.getDefaultInstance();
    return new BuildEventId(
        BuildEventStreamProtos.BuildEventId.newBuilder()
            .setUnstructuredCommandLine(commandLineId)
            .build());
  }

  public static BuildEventId structuredCommandlineId(String commandLineLabel) {
    BuildEventStreamProtos.BuildEventId.StructuredCommandLineId commandLineId =
        BuildEventStreamProtos.BuildEventId.StructuredCommandLineId.newBuilder()
            .setCommandLineLabel(commandLineLabel)
            .build();
    return new BuildEventId(
        BuildEventStreamProtos.BuildEventId.newBuilder()
            .setStructuredCommandLine(commandLineId)
            .build());
  }

  public static BuildEventId optionsParsedId() {
    BuildEventStreamProtos.BuildEventId.OptionsParsedId optionsParsedId =
        BuildEventStreamProtos.BuildEventId.OptionsParsedId.getDefaultInstance();
    return new BuildEventId(
        BuildEventStreamProtos.BuildEventId.newBuilder().setOptionsParsed(optionsParsedId).build());
  }

  public static BuildEventId workspaceStatusId() {
    return new BuildEventId(
        BuildEventStreamProtos.BuildEventId.newBuilder()
            .setWorkspaceStatus(
                BuildEventStreamProtos.BuildEventId.WorkspaceStatusId.getDefaultInstance())
            .build());
  }

  public static BuildEventId fetchId(String url) {
    BuildEventStreamProtos.BuildEventId.FetchId fetchId =
        BuildEventStreamProtos.BuildEventId.FetchId.newBuilder().setUrl(url).build();
    return new BuildEventId(
        BuildEventStreamProtos.BuildEventId.newBuilder().setFetch(fetchId).build());
  }

  public static BuildEventId configurationId(String id) {
    BuildEventStreamProtos.BuildEventId.ConfigurationId configurationId =
        BuildEventStreamProtos.BuildEventId.ConfigurationId.newBuilder().setId(id).build();
    return new BuildEventId(
        BuildEventStreamProtos.BuildEventId.newBuilder().setConfiguration(configurationId).build());
  }

  public static BuildEventId nullConfigurationId() {
    return configurationId("none");
  }

  private static BuildEventId targetPatternExpanded(List<String> targetPattern, boolean skipped) {
    BuildEventStreamProtos.BuildEventId.PatternExpandedId patternId =
        BuildEventStreamProtos.BuildEventId.PatternExpandedId.newBuilder()
            .addAllPattern(targetPattern)
            .build();
    BuildEventStreamProtos.BuildEventId.Builder builder =
        BuildEventStreamProtos.BuildEventId.newBuilder();
    if (skipped) {
      builder.setPatternSkipped(patternId);
    } else {
      builder.setPattern(patternId);
    }
    return new BuildEventId(builder.build());
  }

  public static BuildEventId targetPatternExpanded(List<String> targetPattern) {
    return targetPatternExpanded(targetPattern, false);
  }

  public static BuildEventId targetPatternSkipped(List<String> targetPattern) {
    return targetPatternExpanded(targetPattern, true);
  }

  public static BuildEventId targetConfigured(Label label) {
    BuildEventStreamProtos.BuildEventId.TargetConfiguredId configuredId =
        BuildEventStreamProtos.BuildEventId.TargetConfiguredId.newBuilder()
            .setLabel(label.toString())
            .build();
    return new BuildEventId(
        BuildEventStreamProtos.BuildEventId.newBuilder().setTargetConfigured(configuredId).build());
  }

  public static BuildEventId aspectConfigured(Label label, String aspect) {
    BuildEventStreamProtos.BuildEventId.TargetConfiguredId configuredId =
        BuildEventStreamProtos.BuildEventId.TargetConfiguredId.newBuilder()
            .setLabel(label.toString())
            .setAspect(aspect)
            .build();
    return new BuildEventId(
        BuildEventStreamProtos.BuildEventId.newBuilder().setTargetConfigured(configuredId).build());
  }

  public static BuildEventId targetCompleted(Label target, BuildEventId configuration) {
    BuildEventStreamProtos.BuildEventId.ConfigurationId configId =
        configuration.protoid.getConfiguration();
    BuildEventStreamProtos.BuildEventId.TargetCompletedId targetId =
        BuildEventStreamProtos.BuildEventId.TargetCompletedId.newBuilder()
            .setLabel(target.toString())
            .setConfiguration(configId)
            .build();
    return new BuildEventId(
        BuildEventStreamProtos.BuildEventId.newBuilder().setTargetCompleted(targetId).build());
  }

  public static BuildEventId configuredLabelId(Label label, BuildEventId configuration) {
    BuildEventStreamProtos.BuildEventId.ConfigurationId configId =
        configuration.protoid.getConfiguration();
    BuildEventStreamProtos.BuildEventId.ConfiguredLabelId labelId =
        BuildEventStreamProtos.BuildEventId.ConfiguredLabelId.newBuilder()
            .setLabel(label.toString())
            .setConfiguration(configId)
            .build();
    return new BuildEventId(
        BuildEventStreamProtos.BuildEventId.newBuilder().setConfiguredLabel(labelId).build());
  }

  public static BuildEventId unconfiguredLabelId(Label label) {
    BuildEventStreamProtos.BuildEventId.UnconfiguredLabelId labelId =
        BuildEventStreamProtos.BuildEventId.UnconfiguredLabelId.newBuilder()
            .setLabel(label.toString())
            .build();
    return new BuildEventId(
        BuildEventStreamProtos.BuildEventId.newBuilder().setUnconfiguredLabel(labelId).build());
  }

  public static BuildEventId aspectCompleted(Label target, String aspect) {
    BuildEventStreamProtos.BuildEventId.TargetCompletedId targetId =
        BuildEventStreamProtos.BuildEventId.TargetCompletedId.newBuilder()
            .setLabel(target.toString())
            .setAspect(aspect)
            .build();
    return new BuildEventId(
        BuildEventStreamProtos.BuildEventId.newBuilder().setTargetCompleted(targetId).build());
  }

  public static BuildEventId fromCause(Cause cause) {
    return new BuildEventId(cause.getIdProto());
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
    return new BuildEventId(
        BuildEventStreamProtos.BuildEventId.newBuilder().setActionCompleted(actionId).build());
  }

  public static BuildEventId fromArtifactGroupName(String name) {
    BuildEventStreamProtos.BuildEventId.NamedSetOfFilesId namedSetId =
        BuildEventStreamProtos.BuildEventId.NamedSetOfFilesId.newBuilder().setId(name).build();
    return new BuildEventId(
        BuildEventStreamProtos.BuildEventId.newBuilder().setNamedSet(namedSetId).build());
  }

  public static BuildEventId testResult(
      Label target, Integer run, Integer shard, Integer attempt, BuildEventId configuration) {
    BuildEventStreamProtos.BuildEventId.ConfigurationId configId =
        configuration.protoid.getConfiguration();
    BuildEventStreamProtos.BuildEventId.TestResultId resultId =
        BuildEventStreamProtos.BuildEventId.TestResultId.newBuilder()
            .setLabel(target.toString())
            .setConfiguration(configId)
            .setRun(run + 1)
            .setShard(shard + 1)
            .setAttempt(attempt)
            .build();
    return new BuildEventId(
        BuildEventStreamProtos.BuildEventId.newBuilder().setTestResult(resultId).build());
  }

  public static BuildEventId testResult(
      Label target, Integer run, Integer shard, BuildEventId configuration) {
    return testResult(target, run, shard, 1, configuration);
  }

  public static BuildEventId testSummary(Label target, BuildEventId configuration) {
    BuildEventStreamProtos.BuildEventId.ConfigurationId configId =
        configuration.protoid.getConfiguration();
    BuildEventStreamProtos.BuildEventId.TestSummaryId summaryId =
        BuildEventStreamProtos.BuildEventId.TestSummaryId.newBuilder()
            .setLabel(target.toString())
            .setConfiguration(configId)
            .build();
    return new BuildEventId(
        BuildEventStreamProtos.BuildEventId.newBuilder().setTestSummary(summaryId).build());
  }

  public static BuildEventId buildFinished() {
    BuildEventStreamProtos.BuildEventId.BuildFinishedId finishedId =
        BuildEventStreamProtos.BuildEventId.BuildFinishedId.getDefaultInstance();
    return new BuildEventId(
        BuildEventStreamProtos.BuildEventId.newBuilder().setBuildFinished(finishedId).build());
  }

  public static BuildEventId buildToolLogs() {
    return new BuildEventId(
        BuildEventStreamProtos.BuildEventId.newBuilder()
            .setBuildToolLogs(
                BuildEventStreamProtos.BuildEventId.BuildToolLogsId.getDefaultInstance())
            .build());
  }

  public static BuildEventId buildMetrics() {
    return new BuildEventId(
        BuildEventStreamProtos.BuildEventId.newBuilder()
            .setBuildMetrics(
                BuildEventStreamProtos.BuildEventId.BuildMetricsId.getDefaultInstance())
            .build());
  }

  public static BuildEventId queryOutput() {
    return new BuildEventId(
        BuildEventStreamProtos.BuildEventId.newBuilder()
            .setQueryOutput(
                BuildEventStreamProtos.BuildEventId.QueryOutputId.getDefaultInstance())
            .build());
  }
}
