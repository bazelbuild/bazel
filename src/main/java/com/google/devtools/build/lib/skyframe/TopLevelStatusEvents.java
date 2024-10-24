// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;


import com.google.auto.value.AutoValue;
import com.google.devtools.build.lib.analysis.ConfiguredAspect;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.events.ExtendedEventHandler.Postable;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.skyframe.AspectKeyCreator.AspectKey;
import com.google.devtools.build.skyframe.SkyKey;

/**
 * A collection of events that mark the completion of the analysis/building of top level targets or
 * aspects.
 *
 * <p>These events are used to generate the final results summary.
 */
public final class TopLevelStatusEvents {
  private TopLevelStatusEvents() {}

  interface TopLevelStatusEventWithType extends Postable {
    Type getType();
  }

  /**
   * An event that marks the successful analysis of a top-level target, including tests. A skipped
   * target is still considered analyzed and a TopLevelTargetAnalyzedEvent is expected for it.
   */
  @AutoValue
  public abstract static class TopLevelTargetAnalyzedEvent implements TopLevelStatusEventWithType {
    public abstract ConfiguredTarget configuredTarget();

    public static TopLevelTargetAnalyzedEvent create(ConfiguredTarget configuredTarget) {
      return new AutoValue_TopLevelStatusEvents_TopLevelTargetAnalyzedEvent(configuredTarget);
    }

    @Override
    public Type getType() {
      return Type.TOP_LEVEL_TARGET_ANALYZED;
    }
  }

  /**
   * An event that signals that we can start planting the symlinks for the transitive packages under
   * a top level target.
   *
   * <p>Should always be sent out before {@link TopLevelEntityAnalysisConcludedEvent} to ensure
   * consistency.
   */
  @AutoValue
  public abstract static class TopLevelTargetReadyForSymlinkPlanting
      implements TopLevelStatusEventWithType {
    public abstract NestedSet<Package> transitivePackagesForSymlinkPlanting();

    public static TopLevelTargetReadyForSymlinkPlanting create(
        NestedSet<Package> transitivePackagesForSymlinkPlanting) {
      return new AutoValue_TopLevelStatusEvents_TopLevelTargetReadyForSymlinkPlanting(
          transitivePackagesForSymlinkPlanting);
    }

    @Override
    public Type getType() {
      return Type.TOP_LEVEL_TARGET_READY_FOR_SYMLINK_PLANTING;
    }
  }

  /** An event that marks the skipping of a top-level target, including skipped tests. */
  @AutoValue
  public abstract static class TopLevelTargetSkippedEvent implements TopLevelStatusEventWithType {
    public abstract ConfiguredTarget configuredTarget();

    public static TopLevelTargetSkippedEvent create(ConfiguredTarget configuredTarget) {
      return new AutoValue_TopLevelStatusEvents_TopLevelTargetSkippedEvent(configuredTarget);
    }

    @Override
    public Type getType() {
      return Type.TOP_LEVEL_TARGET_SKIPPED;
    }
  }

  /**
   * An event that marks the conclusion of the analysis of a top level target/aspect, successful or
   * otherwise.
   */
  @AutoValue
  public abstract static class TopLevelEntityAnalysisConcludedEvent
      implements TopLevelStatusEventWithType {
    public abstract SkyKey getAnalyzedTopLevelKey();

    public abstract boolean succeeded();

    public static TopLevelEntityAnalysisConcludedEvent create(
        SkyKey analyzedTopLevelKey, boolean succeeded) {
      return new AutoValue_TopLevelStatusEvents_TopLevelEntityAnalysisConcludedEvent(
          analyzedTopLevelKey, succeeded);
    }

    @Override
    public Type getType() {
      return Type.TOP_LEVEL_ENTITY_ANALYSIS_CONCLUDED;
    }
  }

  /**
   * An event that marks that a top-level target won't be skipped and is pending execution,
   * including test targets.
   */
  @AutoValue
  public abstract static class TopLevelTargetPendingExecutionEvent
      implements TopLevelStatusEventWithType {
    public abstract ConfiguredTarget configuredTarget();

    public abstract boolean isTest();

    public static TopLevelTargetPendingExecutionEvent create(
        ConfiguredTarget configuredTarget, boolean isTest) {
      return new AutoValue_TopLevelStatusEvents_TopLevelTargetPendingExecutionEvent(
          configuredTarget, isTest);
    }

    @Override
    public Type getType() {
      return Type.TOP_LEVEL_TARGET_PENDING_EXECUTION;
    }
  }

  /**
   * An event that denotes that some execution has started in this build.
   *
   * <p>Some special actions e.g. the WorkspaceStatusAction should be excluded from the execution
   * time.
   */
  @AutoValue
  public abstract static class SomeExecutionStartedEvent implements TopLevelStatusEventWithType {

    public static SomeExecutionStartedEvent create() {
      return new AutoValue_TopLevelStatusEvents_SomeExecutionStartedEvent(
          /* countedInExecutionTime= */ true);
    }

    public static SomeExecutionStartedEvent notCountedInExecutionTime() {
      return new AutoValue_TopLevelStatusEvents_SomeExecutionStartedEvent(
          /* countedInExecutionTime= */ false);
    }

    @Override
    public Type getType() {
      return Type.SOME_EXECUTION_STARTED;
    }

    public abstract boolean countedInExecutionTime();
  }
  /** An event that marks the successful build of a top-level target, including tests. */
  @AutoValue
  public abstract static class TopLevelTargetBuiltEvent implements TopLevelStatusEventWithType {
    abstract ConfiguredTargetKey configuredTargetKey();

    public static TopLevelTargetBuiltEvent create(ConfiguredTargetKey configuredTargetKey) {
      return new AutoValue_TopLevelStatusEvents_TopLevelTargetBuiltEvent(configuredTargetKey);
    }

    @Override
    public Type getType() {
      return Type.TOP_LEVEL_TARGET_BUILT;
    }
  }

  /** An event that marks the successful analysis of a test target. */
  @AutoValue
  public abstract static class TestAnalyzedEvent implements TopLevelStatusEventWithType {
    public abstract ConfiguredTarget configuredTarget();

    public abstract BuildConfigurationValue buildConfigurationValue();

    public abstract boolean isSkipped();

    public static TestAnalyzedEvent create(
        ConfiguredTarget configuredTarget,
        BuildConfigurationValue buildConfigurationValue,
        boolean isSkipped) {
      return new AutoValue_TopLevelStatusEvents_TestAnalyzedEvent(
          configuredTarget, buildConfigurationValue, isSkipped);
    }

    @Override
    public Type getType() {
      return Type.TEST_ANALYZED;
    }
  }

  /** An event that marks the successful analysis of an aspect. */
  @AutoValue
  public abstract static class AspectAnalyzedEvent implements TopLevelStatusEventWithType {
    abstract AspectKey aspectKey();

    public abstract ConfiguredAspect configuredAspect();

    public static AspectAnalyzedEvent create(
        AspectKey aspectKey, ConfiguredAspect configuredAspect) {
      return new AutoValue_TopLevelStatusEvents_AspectAnalyzedEvent(aspectKey, configuredAspect);
    }

    @Override
    public Type getType() {
      return Type.ASPECT_ANALYZED;
    }
  }

  /** An event that marks the successful building of an aspect. */
  @AutoValue
  public abstract static class AspectBuiltEvent implements TopLevelStatusEventWithType {
    abstract AspectKey aspectKey();

    public static AspectBuiltEvent create(AspectKey aspectKey) {
      return new AutoValue_TopLevelStatusEvents_AspectBuiltEvent(aspectKey);
    }

    @Override
    public Type getType() {
      return Type.ASPECT_BUILT;
    }
  }

  enum Type {
    TOP_LEVEL_TARGET_CONFIGURED,
    TOP_LEVEL_TARGET_ANALYZED,
    TOP_LEVEL_TARGET_READY_FOR_SYMLINK_PLANTING,
    TOP_LEVEL_TARGET_SKIPPED,
    TOP_LEVEL_ENTITY_ANALYSIS_CONCLUDED,
    TOP_LEVEL_TARGET_PENDING_EXECUTION,
    SOME_EXECUTION_STARTED,
    TOP_LEVEL_TARGET_BUILT,
    TEST_ANALYZED,
    ASPECT_ANALYZED,
    ASPECT_BUILT
  }
}
