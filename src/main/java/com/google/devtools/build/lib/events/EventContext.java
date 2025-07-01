// Copyright 2024 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.events;

import com.google.auto.value.AutoValue;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId.ConfigurationId;
import javax.annotation.Nullable;

/**
 * A generic container for information about an event's origin, used for decoupled data binding.
 *
 * <p>This object is attached to an {@link Event} using {@link Event#withProperty} and consumed by
 * specialized handlers like the {@code OutputSuppressionFilter}.
 */
@AutoValue
public abstract class EventContext {
  @Nullable
  public abstract String getTargetLabel();

  @Nullable
  public abstract String getPackage();

  @Nullable
  public abstract String getRuleClass();

  @Nullable
  public abstract ConfigurationId getPlatform();

  public static Builder builder() {
    return new AutoValue_EventContext.Builder();
  }

  /** Builder for {@link EventContext}. */
  @AutoValue.Builder
  public abstract static class Builder {
    public abstract Builder setTargetLabel(String value);

    public abstract Builder setPackage(String value);

    public abstract Builder setRuleClass(String value);

    public abstract Builder setPlatform(ConfigurationId value);

    public abstract EventContext build();
  }
}
