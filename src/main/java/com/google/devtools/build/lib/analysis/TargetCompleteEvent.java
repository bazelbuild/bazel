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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.EventReportingArtifacts;
import com.google.devtools.build.lib.analysis.TopLevelArtifactHelper.ArtifactsInOutputGroup;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
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
import com.google.devtools.build.lib.packages.TestSize;
import com.google.devtools.build.lib.rules.test.TestProvider;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Collection;

/** This event is fired as soon as a target is either built or fails. */
public final class TargetCompleteEvent
    implements SkyValue,
        BuildEventWithOrderConstraint,
        EventReportingArtifacts,
        BuildEventWithConfiguration {
  private final ConfiguredTarget target;
  private final NestedSet<Cause> rootCauses;
  private final Collection<BuildEventId> postedAfter;
  private final Iterable<ArtifactsInOutputGroup> outputs;
  private final boolean isTest;

  private TargetCompleteEvent(
      ConfiguredTarget target,
      NestedSet<Cause> rootCauses,
      Iterable<ArtifactsInOutputGroup> outputs,
      boolean isTest) {
    this.target = target;
    this.rootCauses =
        (rootCauses == null) ? NestedSetBuilder.<Cause>emptySet(Order.STABLE_ORDER) : rootCauses;

    ImmutableList.Builder postedAfterBuilder = ImmutableList.builder();
    for (Cause cause : getRootCauses()) {
      postedAfterBuilder.add(BuildEventId.fromCause(cause));
    }
    this.postedAfter = postedAfterBuilder.build();
    this.outputs = outputs;
    this.isTest = isTest;
  }

  /** Construct a successful target completion event. */
  public static TargetCompleteEvent createSuccessfulTarget(
      ConfiguredTarget ct, NestedSet<ArtifactsInOutputGroup> outputs) {
    return new TargetCompleteEvent(ct, null, outputs, false);
  }

  /** Construct a successful target completion event for a target that will be tested. */
  public static TargetCompleteEvent createSuccessfulTestTarget(ConfiguredTarget ct) {
    return new TargetCompleteEvent(ct, null, ImmutableList.<ArtifactsInOutputGroup>of(), true);
  }


  /**
   * Construct a target completion event for a failed target, with the given non-empty root causes.
   */
  public static TargetCompleteEvent createFailed(ConfiguredTarget ct, NestedSet<Cause> rootCauses) {
    Preconditions.checkArgument(!Iterables.isEmpty(rootCauses));
    return new TargetCompleteEvent(
        ct, rootCauses, ImmutableList.<ArtifactsInOutputGroup>of(), false);
  }

  /**
   * Returns the target associated with the event.
   */
  public ConfiguredTarget getTarget() {
    return target;
  }

  /**
   * Determines whether the target has failed or succeeded.
   */
  public boolean failed() {
    return !rootCauses.isEmpty();
  }

  /** Get the root causes of the target. May be empty. */
  public Iterable<Cause> getRootCauses() {
    return rootCauses;
  }

  @Override
  public BuildEventId getEventId() {
    BuildConfiguration config = getTarget().getConfiguration();
    BuildEventId configId =
        config == null ? BuildEventId.nullConfigurationId() : config.getEventId();
    return BuildEventId.targetCompleted(getTarget().getLabel(), configId);
  }

  @Override
  public Collection<BuildEventId> getChildrenEvents() {
    ImmutableList.Builder childrenBuilder = ImmutableList.builder();
    for (Cause cause : getRootCauses()) {
      childrenBuilder.add(BuildEventId.fromCause(cause));
    }
    if (isTest) {
      // For tests, announce all the test actions that will minimally happen (except for
      // interruption). If after the result of a test action another attempt is necessary,
      // it will be announced with the action that made the new attempt necessary.
      Label label = target.getTarget().getLabel();
      TestProvider.TestParams params = target.getProvider(TestProvider.class).getTestParams();
      for (int run = 0; run < Math.max(params.getRuns(), 1); run++) {
        for (int shard = 0; shard < Math.max(params.getShards(), 1); shard++) {
          childrenBuilder.add(
              BuildEventId.testResult(label, run, shard, target.getConfiguration().getEventId()));
        }
      }
      childrenBuilder.add(BuildEventId.testSummary(label, target.getConfiguration().getEventId()));
    }
    return childrenBuilder.build();
  }

  private BuildEventStreamProtos.TestSize bepTestSize(TestSize size) {
    switch (size) {
      case SMALL:
        return BuildEventStreamProtos.TestSize.SMALL;
      case MEDIUM:
        return BuildEventStreamProtos.TestSize.MEDIUM;
      case LARGE:
        return BuildEventStreamProtos.TestSize.LARGE;
      case ENORMOUS:
        return BuildEventStreamProtos.TestSize.ENORMOUS;
      default:
        return BuildEventStreamProtos.TestSize.UNKNOWN;
    }
  }

  @Override
  public BuildEventStreamProtos.BuildEvent asStreamProto(BuildEventConverters converters) {
    BuildEventStreamProtos.TargetComplete.Builder builder =
        BuildEventStreamProtos.TargetComplete.newBuilder();

    builder.setSuccess(!failed());
    builder.setTargetKind(target.getTarget().getTargetKind());
    builder.addAllTag(getTags());
    builder.addAllOutputGroup(getOutputFilesByGroup(converters.artifactGroupNamer()));

    if (isTest) {
      builder.setTestSize(
          bepTestSize(TestSize.getTestSize(target.getTarget().getAssociatedRule())));
    }

    // TODO(aehlig): remove direct reporting of artifacts as soon as clients no longer
    // need it.
    for (ArtifactsInOutputGroup group : outputs) {
      if (group.areImportant()) {
        for (Artifact artifact : group.getArtifacts()) {
          String name = artifact.getPath().relativeTo(artifact.getRoot().getPath()).getPathString();
          String uri = converters.pathConverter().apply(artifact.getPath());
          builder.addImportantOutput(File.newBuilder().setName(name).setUri(uri).build());
        }
      }
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
    return builder.build();
  }

  @Override
  public Collection<BuildEvent> getConfigurations() {
    BuildConfiguration configuration = target.getConfiguration();
    if (configuration != null) {
      return ImmutableList.of(target.getConfiguration());
    } else {
      return ImmutableList.<BuildEvent>of();
    }
  }

  private Iterable<String> getTags() {
    // We are only interested in targets that are rules.
    if (!(target instanceof RuleConfiguredTarget)) {
      return ImmutableList.<String>of();
    }
    AttributeMap attributes = ConfiguredAttributeMapper.of((RuleConfiguredTarget) target);
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
    return groups.build();
  }
}
