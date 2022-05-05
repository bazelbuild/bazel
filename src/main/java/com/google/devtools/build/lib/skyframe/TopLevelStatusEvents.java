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
import com.google.devtools.build.lib.events.ExtendedEventHandler.ProgressLike;
import com.google.devtools.build.lib.skyframe.AspectKeyCreator.AspectKey;

/**
 * A collection of events that mark the completion of the analysis/building of top level targets or
 * aspects.
 *
 * <p>These events are used to generate the final results summary.
 *
 * <p>IMPORTANT: since these events can be fired from within a SkyFunction, there exists a risk of
 * duplication (e.g. a rerun of the SkyFunction due to missing values or error bubbling). Receivers
 * of these events should be robust enough to receive and de-duplicate events if necessary.
 */
public final class TopLevelStatusEvents {
  private TopLevelStatusEvents() {}

  @AutoValue
  abstract static class TopLevelTargetAnalyzedEvent implements ProgressLike {
    abstract ConfiguredTarget configuredTarget();

    static TopLevelTargetAnalyzedEvent create(ConfiguredTarget configuredTarget) {
      return new AutoValue_TopLevelStatusEvents_TopLevelTargetAnalyzedEvent(configuredTarget);
    }
  }

  @AutoValue
  abstract static class TopLevelTargetSkippedEvent implements ProgressLike {
    abstract ConfiguredTarget configuredTarget();

    static TopLevelTargetSkippedEvent create(ConfiguredTarget configuredTarget) {
      return new AutoValue_TopLevelStatusEvents_TopLevelTargetSkippedEvent(configuredTarget);
    }
  }

  @AutoValue
  abstract static class TopLevelTargetBuiltEvent implements ProgressLike {
    abstract ConfiguredTargetKey configuredTargetKey();

    static TopLevelTargetBuiltEvent create(ConfiguredTargetKey configuredTargetKey) {
      return new AutoValue_TopLevelStatusEvents_TopLevelTargetBuiltEvent(configuredTargetKey);
    }
  }

  /** An event that marks the successful analysis of a test target. */
  @AutoValue
  public abstract static class TestAnalyzedEvent implements ProgressLike {
    public abstract ConfiguredTarget configuredTarget();

    public abstract BuildConfigurationValue buildConfigurationValue();

    public abstract boolean isSkipped();

    static TestAnalyzedEvent create(
        ConfiguredTarget configuredTarget, BuildConfigurationValue buildConfigurationValue) {
      return new AutoValue_TopLevelStatusEvents_TestAnalyzedEvent(
          configuredTarget, buildConfigurationValue, /*isSkipped=*/ false);
    }

    static TestAnalyzedEvent createSkipped(
        ConfiguredTarget configuredTarget, BuildConfigurationValue buildConfigurationValue) {
      return new AutoValue_TopLevelStatusEvents_TestAnalyzedEvent(
          configuredTarget, buildConfigurationValue, /*isSkipped=*/ true);
    }
  }

  @AutoValue
  abstract static class AspectAnalyzedEvent implements ProgressLike {
    abstract AspectKey aspectKey();

    abstract ConfiguredAspect configuredAspect();

    static AspectAnalyzedEvent create(AspectKey aspectKey, ConfiguredAspect configuredAspect) {
      return new AutoValue_TopLevelStatusEvents_AspectAnalyzedEvent(aspectKey, configuredAspect);
    }
  }

  @AutoValue
  abstract static class AspectBuiltEvent implements ProgressLike {
    abstract AspectKey aspectKey();

    static AspectBuiltEvent create(AspectKey aspectKey) {
      return new AutoValue_TopLevelStatusEvents_AspectBuiltEvent(aspectKey);
    }
  }
}
