// Copyright 2015 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactPathResolver;
import com.google.devtools.build.lib.actions.EventReportingArtifacts;
import com.google.devtools.build.lib.analysis.TopLevelArtifactHelper.ArtifactsInOutputGroup;
import com.google.devtools.build.lib.analysis.TopLevelArtifactHelper.ArtifactsToBuild;
import com.google.devtools.build.lib.buildeventstream.ArtifactGroupNamer;
import com.google.devtools.build.lib.buildeventstream.BuildEventContext;
import com.google.devtools.build.lib.buildeventstream.BuildEventId;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.OutputGroup;
import com.google.devtools.build.lib.buildeventstream.BuildEventWithOrderConstraint;
import com.google.devtools.build.lib.buildeventstream.GenericBuildEvent;
import com.google.devtools.build.lib.causes.Cause;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.NestedSetView;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.skyframe.AspectValue;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Collection;

/** This event is fired as soon as a top-level aspect is either built or fails. */
public class AspectCompleteEvent
    implements SkyValue, BuildEventWithOrderConstraint, EventReportingArtifacts {
  private final AspectValue aspectValue;
  private final NestedSet<Cause> rootCauses;
  private final Collection<BuildEventId> postedAfter;
  private final ArtifactPathResolver pathResolver;
  private final ArtifactsToBuild artifacts;
  private final BuildEventId configurationEventId;

  private AspectCompleteEvent(
      AspectValue aspectValue,
      NestedSet<Cause> rootCauses,
      ArtifactPathResolver pathResolver,
      ArtifactsToBuild artifacts,
      BuildEventId configurationEventId) {
    this.aspectValue = aspectValue;
    this.rootCauses =
        (rootCauses == null) ? NestedSetBuilder.<Cause>emptySet(Order.STABLE_ORDER) : rootCauses;
    ImmutableList.Builder postedAfterBuilder = ImmutableList.builder();
    for (Cause cause : getRootCauses()) {
      postedAfterBuilder.add(BuildEventId.fromCause(cause));
    }
    this.postedAfter = postedAfterBuilder.build();
    this.pathResolver = pathResolver;
    this.artifacts = artifacts;
    this.configurationEventId = configurationEventId;
  }

  /** Construct a successful target completion event. */
  public static AspectCompleteEvent createSuccessful(
      AspectValue value,
      ArtifactPathResolver pathResolver,
      ArtifactsToBuild artifacts,
      BuildEventId configurationEventId) {
    return new AspectCompleteEvent(value, null, pathResolver, artifacts, configurationEventId);
  }

  /**
   * Construct a target completion event for a failed target, with the given non-empty root causes.
   */
  public static AspectCompleteEvent createFailed(
      AspectValue value, NestedSet<Cause> rootCauses, BuildEventId configurationEventId) {
    Preconditions.checkArgument(!Iterables.isEmpty(rootCauses));
    return new AspectCompleteEvent(
        value, rootCauses, ArtifactPathResolver.IDENTITY, null, configurationEventId);
  }

  /**
   * Returns the target associated with the event.
   */
  public AspectValue getAspectValue() {
    return aspectValue;
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
    return BuildEventId.aspectCompleted(
        aspectValue.getLabel(),
        configurationEventId,
        aspectValue.getAspect().getDescriptor().getDescription());
  }

  @Override
  public Collection<BuildEventId> postedAfter() {
    return postedAfter;
  }

  @Override
  public Collection<BuildEventId> getChildrenEvents() {
    return ImmutableList.of();
  }

  @Override
  public ReportedArtifacts reportedArtifacts() {
    ImmutableSet.Builder<NestedSet<Artifact>> builder = ImmutableSet.builder();
    if (artifacts != null) {
      for (ArtifactsInOutputGroup artifactsInGroup : artifacts.getAllArtifactsByOutputGroup()) {
        builder.add(artifactsInGroup.getArtifacts());
      }
    }
    return new ReportedArtifacts(builder.build(), pathResolver);
  }

  @Override
  public BuildEventStreamProtos.BuildEvent asStreamProto(BuildEventContext converters) {
    ArtifactGroupNamer namer = converters.artifactGroupNamer();

    BuildEventStreamProtos.TargetComplete.Builder builder =
        BuildEventStreamProtos.TargetComplete.newBuilder();
    builder.setSuccess(!failed());
    if (artifacts != null) {
      for (ArtifactsInOutputGroup artifactsInGroup : artifacts.getAllArtifactsByOutputGroup()) {
        OutputGroup.Builder groupBuilder = OutputGroup.newBuilder();
        groupBuilder.setName(artifactsInGroup.getOutputGroup());
        groupBuilder.addFileSets(
            namer.apply(
                (new NestedSetView<Artifact>(artifactsInGroup.getArtifacts())).identifier()));
        builder.addOutputGroup(groupBuilder.build());
      }
    }
    return GenericBuildEvent.protoChaining(this).setCompleted(builder.build()).build();
  }
}
