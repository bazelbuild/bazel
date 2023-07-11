// Copyright 2017 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.MoreObjects.toStringHelper;
import static com.google.devtools.build.lib.analysis.config.BuildConfigurationValue.buildEvent;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventContext;
import com.google.devtools.build.lib.buildeventstream.BuildEventIdUtil;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId.ConfigurationId;
import com.google.devtools.build.lib.buildeventstream.BuildEventWithConfiguration;
import com.google.devtools.build.lib.buildeventstream.GenericBuildEvent;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.BuildConfigurationKey;
import java.util.Optional;
import javax.annotation.Nullable;

/**
 * Error message of an analysis root cause. This is separate from {@link AnalysisFailureEvent} to
 * avoid duplicating error messages in the stream if multiple targets fail due to the same root
 * cause. It also allows UIs to collate errors by root cause.
 */
public final class AnalysisRootCauseEvent implements BuildEventWithConfiguration {
  /**
   * A tri-state representation of the configuration to capture two different notions of nullness.
   *
   * <ul>
   *   <li>The contents of a non-empty value is a configuration value.
   *   <li>An {@link Optional#empty} represents the <i>null configuration</i>, used for
   *       unconfigurable targets, for example source files.
   *   <li>A null value means an <i>unavailable configuration</i>. Sometimes errors may occur for a
   *       transient {@link BuildConfigurationKey} for which a {@link BuildConfigurationValue} is
   *       never computed, for example, the intermediate configuration after the attribute
   *       transition occurs but before the rule transition.
   * </ul>
   */
  @Nullable private final Optional<BuildConfigurationValue> configuration;

  private final ConfigurationId configurationId;
  private final Label label;
  private final String errorMessage;

  public static AnalysisRootCauseEvent withConfigurationValue(
      @Nullable BuildConfigurationValue configuration, Label label, String errorMessage) {
    return new AnalysisRootCauseEvent(
        Optional.ofNullable(configuration),
        BuildConfigurationValue.configurationIdMessage(configuration),
        label,
        errorMessage);
  }

  public static AnalysisRootCauseEvent withUnavailableConfiguration(
      ConfigurationId configurationId, Label label, String errorMessage) {
    return new AnalysisRootCauseEvent(
        /* configuration= */ null, configurationId, label, errorMessage);
  }

  private AnalysisRootCauseEvent(
      @Nullable Optional<BuildConfigurationValue> configuration,
      ConfigurationId configurationId,
      Label label,
      String errorMessage) {
    this.configuration = configuration;
    this.configurationId = configurationId;
    this.label = label;
    this.errorMessage = errorMessage;
  }

  @VisibleForTesting
  public Label getLabel() {
    return label;
  }

  @Override
  public BuildEventId getEventId() {
    // This needs to match AnalysisFailedCause.getIdProto.
    return BuildEventIdUtil.configuredLabelId(label, configurationId);
  }

  @Override
  public ImmutableList<BuildEventId> getChildrenEvents() {
    return ImmutableList.of();
  }

  @Override
  public BuildEventStreamProtos.BuildEvent asStreamProto(BuildEventContext converters) {
    return GenericBuildEvent.protoChaining(this)
        .setAborted(
            BuildEventStreamProtos.Aborted.newBuilder()
                .setReason(BuildEventStreamProtos.Aborted.AbortReason.ANALYSIS_FAILURE)
                .setDescription(errorMessage)
                .build())
        .build();
  }

  @Override
  public ImmutableList<BuildEvent> getConfigurations() {
    if (configuration == null) {
      return ImmutableList.of();
    }
    return ImmutableList.of(buildEvent(configuration.orElse(null)));
  }

  @Override
  public boolean storeForReplay() {
    return true;
  }

  @Override
  public String toString() {
    return toStringHelper(this)
        .add("configuration", configuration)
        .add("configurationId", configurationId)
        .add("label", label)
        .add("errorMessage", errorMessage)
        .toString();
  }
}
