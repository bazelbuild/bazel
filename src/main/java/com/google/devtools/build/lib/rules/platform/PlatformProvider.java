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

package com.google.devtools.build.lib.rules.platform;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import java.util.Map;

/** Provider for a platform, which is a group of constraints and values. */
@AutoValue
@Immutable
public abstract class PlatformProvider implements TransitiveInfoProvider {
  public abstract ImmutableMap<ConstraintSettingProvider, ConstraintValueProvider> constraints();

  public abstract ImmutableMap<String, String> remoteExecutionProperties();

  public static Builder builder() {
    return new AutoValue_PlatformProvider.Builder();
  }

  /** A Builder instance to configure a new {@link PlatformProvider}. */
  @AutoValue.Builder
  public abstract static class Builder {
    public abstract Builder constraints(
        Map<ConstraintSettingProvider, ConstraintValueProvider> constraints);

    public abstract Builder remoteExecutionProperties(Map<String, String> properties);

    public abstract PlatformProvider build();
  }
}
