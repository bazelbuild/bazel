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

package com.google.devtools.build.lib.analysis;

import static com.google.devtools.build.lib.buildeventstream.TestFileNameConstants.BASELINE_COVERAGE;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.EventReportingArtifacts;
import com.google.devtools.build.lib.analysis.TopLevelArtifactHelper.ArtifactsInOutputGroup;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesProvider;
import com.google.devtools.build.lib.analysis.test.TestProvider;
import com.google.devtools.build.lib.buildeventstream.ArtifactGroupNamer;
import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventConverters;
import com.google.devtools.build.lib.buildeventstream.BuildEventId;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.File;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.OutputGroup;
import com.google.devtools.build.lib.buildeventstream.BuildEventWithConfiguration;
import com.google.devtools.build.lib.buildeventstream.BuildEventWithOrderConstraint;
import com.google.devtools.build.lib.buildeventstream.GenericBuildEvent;
import com.google.devtools.build.lib.causes.Cause;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.NestedSetView;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.ConfiguredAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.TestSize;
import com.google.devtools.build.lib.packages.TestTimeout;
import com.google.devtools.build.lib.rules.AliasConfiguredTarget;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Collection;
import java.util.function.Function;

/** This event is fired as soon as a target is either built or fails. */
public final class TargetCompleteEvent
    implements SkyValue,
        BuildEventWithOrderConstraint,
        EventReportingArtifacts,
        BuildEventWithConfiguration {
  private final ConfiguredTargetAndData targetAndData;
  private final NestedSet<Cause> rootCauses;
  private final ImmutableList<BuildEventId> postedAfter;
  private final Iterable<ArtifactsInOutputGroup> outputs;
  private final NestedSet<Artifact> baselineCoverageArtifacts;
  private final boolean isTest;

  private TargetCompleteEvent(
      ConfiguredTargetAndData targetAndData,
      NestedSet<Cause> rootCauses,
      Iterable<ArtifactsInOutputGroup> outputs,
      boolean isTest) {
    this.targetAndData = targetAndData;
    this.rootCauses =
        (rootCauses == null) ? NestedSetBuilder.<Cause>emptySet(Order.STABLE_ORDER) : rootCauses;

    ImmutableList.Builder<BuildEventId> postedAfterBuilder = ImmutableList.builder();
    Label label = getTarget().getLabel();
    if (targetAndData.getConfiguredTarget() instanceof AliasConfiguredTarget) {
      label = ((AliasConfiguredTarget) targetAndData.getConfiguredTarget()).getOriginalLabel();
    }
    postedAfterBuilder.add(BuildEventId.targetConfigured(label));
    for (Cause cause : getRootCauses()) {
      postedAfterBuilder.add(BuildEventId.fromCause(cause));
    }
    this.postedAfter = postedAfterBuilder.build();
    this.outputs = outputs;
    this.isTest = isTest;
    InstrumentedFilesProvider instrumentedFilesProvider =
        this.targetAndData.getConfiguredTarget().getProvider(InstrumentedFilesProvider.class);
    if (instrumentedFilesProvider == null) {
      this.baselineCoverageArtifacts = null;
    } else {
      NestedSet<Artifact> baselineCoverageArtifacts =
          instrumentedFilesProvider.getBaselineCoverageArtifacts();
      if (!baselineCoverageArtifacts.isEmpty()) {
        this.baselineCoverageArtifacts = baselineCoverageArtifacts;
      } else {
        this.baselineCoverageArtifacts = null;
      }
    }
  }

  /** Construct a successful target completion event. */
  public static TargetCompleteEvent successfulBuild(
      ConfiguredTargetAndData ct, NestedSet<ArtifactsInOutputGroup> outputs) {
    return new TargetCompleteEvent(ct, null, outputs, false);
  }

  /** Construct a successful target completion event for a target that will be tested. */
  public static TargetCompleteEvent successfulBuildSchedulingTest(
      ConfiguredTargetAndData ct, NestedSet<ArtifactsInOutputGroup> outputs) {
    return new TargetCompleteEvent(ct, null, outputs, true);
  }

  /**
   * Construct a target completion event for a failed target, with the given non-empty root causes.
   */
  public static TargetCompleteEvent createFailed(
      ConfiguredTargetAndData ct, NestedSet<Cause> rootCauses) {
    Preconditions.checkArgument(!Iterables.isEmpty(rootCauses));
    return new TargetCompleteEvent(ct, rootCauses, ImmutableList.of(), false);
  }

  /** Returns the target associated with the event. */
  public ConfiguredTarget getTarget() {
    return targetAndData.getConfiguredTarget();
  }

  /** Determines whether the target has failed or succeeded. */
  public boolean failed() {
    return !rootCauses.isEmpty();
  }

  /** Get the root causes of the target. May be empty. */
  public Iterable<Cause> getRootCauses() {
    return rootCauses;
  }

  @Override
  public BuildEventId getEventId() {
    Label label = getTarget().getLabel();
    if (targetAndData.getConfiguredTarget() instanceof AliasConfiguredTarget) {
      label = ((AliasConfiguredTarget) targetAndData.getConfiguredTarget()).getOriginalLabel();
    }
    BuildConfiguration config = targetAndData.getConfiguration();
    BuildEventId configId =
        config == null ? BuildEventId.nullConfigurationId() : config.getEventId();
    return BuildEventId.targetCompleted(label, configId);
  }

  @Override
  public Collection<BuildEventId> getChildrenEvents() {
    ImmutableList.Builder<BuildEventId> childrenBuilder = ImmutableList.builder();
    for (Cause cause : getRootCauses()) {
      childrenBuilder.add(BuildEventId.fromCause(cause));
    }
    if (isTest) {
      // For tests, announce all the test actions that will minimally happen (except for
      // interruption). If after the result of a test action another attempt is necessary,
      // it will be announced with the action that made the new attempt necessary.
      Label label = targetAndData.getConfiguredTarget().getLabel();
      TestProvider.TestParams params =
          targetAndData.getConfiguredTarget().getProvider(TestProvider.class).getTestParams();
      for (int run = 0; run < Math.max(params.getRuns(), 1); run++) {
        for (int shard = 0; shard < Math.max(params.getShards(), 1); shard++) {
          childrenBuilder.add(
              BuildEventId.testResult(
                  label, run, shard, targetAndData.getConfiguration().getEventId()));
        }
      }
      childrenBuilder.add(
          BuildEventId.testSummary(label, targetAndData.getConfiguration().getEventId()));
    }
    return childrenBuilder.build();
  }

  // TODO(aehlig): remove as soon as we managed to get rid of the deprecated "important_output"
  // field.
  private static void addImportantOutputs(
      BuildEventStreamProtos.TargetComplete.Builder builder,
      BuildEventConverters converters,
      Iterable<Artifact> artifacts) {
    addImportantOutputs(builder, Artifact::getRootRelativePathString, converters, artifacts);
  }

  private static void addImportantOutputs(
      BuildEventStreamProtos.TargetComplete.Builder builder,
      Function<Artifact, String> artifactNameFunction,
      BuildEventConverters converters,
      Iterable<Artifact> artifacts) {
    for (Artifact artifact : artifacts) {
      String name = artifactNameFunction.apply(artifact);
      String uri = converters.pathConverter().apply(artifact.getPath());
      builder.addImportantOutput(File.newBuilder().setName(name).setUri(uri).build());
    }
  }

  @Override
  public BuildEventStreamProtos.BuildEvent asStreamProto(BuildEventConverters converters) {
    BuildEventStreamProtos.TargetComplete.Builder builder =
        BuildEventStreamProtos.TargetComplete.newBuilder();

    builder.setSuccess(!failed());
    builder.setTargetKind(targetAndData.getTarget().getTargetKind());
    builder.addAllTag(getTags());
    builder.addAllOutputGroup(getOutputFilesByGroup(converters.artifactGroupNamer()));

    if (isTest) {
      builder.setTestTimeoutSeconds(getTestTimeoutSeconds(targetAndData));
      builder.setTestSize(
          TargetConfiguredEvent.bepTestSize(
              TestSize.getTestSize(targetAndData.getTarget().getAssociatedRule())));
    }

    // TODO(aehlig): remove direct reporting of artifacts as soon as clients no longer
    // need it.
    for (ArtifactsInOutputGroup group : outputs) {
      if (group.areImportant()) {
        addImportantOutputs(builder, converters, group.getArtifacts());
      }
    }
    if (baselineCoverageArtifacts != null) {
      addImportantOutputs(
          builder, (artifact -> BASELINE_COVERAGE), converters, baselineCoverageArtifacts);
    }

    BuildEventStreamProtos.TargetComplete complete = builder.build();
    return GenericBuildEvent.protoChaining(this).setCompleted(complete).build();
  }

  @Override
  public Collection<BuildEventId> postedAfter() {
    return postedAfter;
  }

  @Override
  public Collection<NestedSet<Artifact>> reportedArtifacts() {
    ImmutableSet.Builder<NestedSet<Artifact>> builder =
        new ImmutableSet.Builder<NestedSet<Artifact>>();
    for (ArtifactsInOutputGroup artifactsInGroup : outputs) {
      builder.add(artifactsInGroup.getArtifacts());
    }
    if (baselineCoverageArtifacts != null) {
      builder.add(baselineCoverageArtifacts);
    }
    return builder.build();
  }

  @Override
  public Collection<BuildEvent> getConfigurations() {
    BuildConfiguration configuration = targetAndData.getConfiguration();
    if (configuration != null) {
      return ImmutableList.of(targetAndData.getConfiguration().toBuildEvent());
    } else {
      return ImmutableList.<BuildEvent>of();
    }
  }

  private Iterable<String> getTags() {
    // We are only interested in targets that are rules.
    if (!(targetAndData.getConfiguredTarget() instanceof RuleConfiguredTarget)) {
      return ImmutableList.<String>of();
    }
    AttributeMap attributes =
        ConfiguredAttributeMapper.of(
            (Rule) targetAndData.getTarget(),
            ((RuleConfiguredTarget) targetAndData.getConfiguredTarget()).getConfigConditions());
    // Every rule (implicitly) has a "tags" attribute.
    return attributes.get("tags", Type.STRING_LIST);
  }

  private Iterable<OutputGroup> getOutputFilesByGroup(ArtifactGroupNamer namer) {
    ImmutableList.Builder<OutputGroup> groups = ImmutableList.builder();
    for (ArtifactsInOutputGroup artifactsInOutputGroup : outputs) {
      OutputGroup.Builder groupBuilder = OutputGroup.newBuilder();
      groupBuilder.setName(artifactsInOutputGroup.getOutputGroup());
      groupBuilder.addFileSets(
          namer.apply(
              (new NestedSetView<Artifact>(artifactsInOutputGroup.getArtifacts())).identifier()));
      groups.add(groupBuilder.build());
    }
    if (baselineCoverageArtifacts != null) {
      groups.add(
          OutputGroup.newBuilder()
              .setName(BASELINE_COVERAGE)
              .addFileSets(
                  namer.apply(
                      (new NestedSetView<Artifact>(baselineCoverageArtifacts).identifier())))
              .build());
    }
    return groups.build();
  }

  /**
   * Returns timeout value in seconds that should be used for all test actions under this configured
   * target. We always use the "categorical timeouts" which are based on the --test_timeout flag. A
   * rule picks its timeout but ends up with the same effective value as all other rules in that
   * category and configuration.
   */
  private Long getTestTimeoutSeconds(ConfiguredTargetAndData targetAndData) {
    BuildConfiguration configuration = targetAndData.getConfiguration();
    Rule associatedRule = targetAndData.getTarget().getAssociatedRule();
    TestTimeout categoricalTimeout = TestTimeout.getTestTimeout(associatedRule);
    return configuration.getTestTimeout().get(categoricalTimeout).getSeconds();
  }
}
