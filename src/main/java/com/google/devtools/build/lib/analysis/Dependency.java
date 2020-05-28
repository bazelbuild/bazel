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

import com.google.auto.value.AutoValue;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransition;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.AspectDescriptor;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import javax.annotation.Nullable;

/**
 * A dependency of a configured target through a label.
 *
 * <p>All instances have an explicit configuration, which includes the target and the configuration
 * of the dependency configured target and any aspects that may be required, as well as the
 * configurations for these aspects and transition keys. {@link Dependency#getTransitionKeys}
 * provides some more context on transition keys.
 *
 * <p>Note that the presence of an aspect here does not necessarily mean that it will be created.
 * They will be filtered based on the {@link TransitiveInfoProvider} instances their associated
 * configured targets have (we cannot do that here because the configured targets are not available
 * yet). No error or warning is reported in this case, because it is expected that rules sometimes
 * over-approximate the providers they supply in their definitions.
 */
@AutoValue
public abstract class Dependency {
  /** Builder to assist in creating dependency instances. */
  @AutoValue.Builder
  public abstract static class Builder {
    /** Sets the label of the target this dependency points to. */
    public abstract Builder setLabel(Label label);

    /** Sets the configuration intended for this dependency. */
    public abstract Builder setConfiguration(BuildConfiguration configuration);

    /** Explicitly set the configuration for this dependency to null. */
    public Builder withNullConfiguration() {
      return setConfiguration(null);
    }

    /** Add aspects to this Dependency. The same configuration is applied to all aspects. */
    public abstract Builder setAspects(AspectCollection aspects);

    /** Sets the keys of a configuration transition. */
    public Builder setTransitionKey(String key) {
      return setTransitionKeys(ImmutableList.of(key));
    }

    /** Sets the keys of a configuration transition. */
    public abstract Builder setTransitionKeys(ImmutableList<String> keys);

    // Not public.
    abstract Dependency autoBuild();

    /** Returns the full Dependency instance. */
    public Dependency build() {
      Dependency dependency = autoBuild();
      if (dependency.getConfiguration() == null) {
        Preconditions.checkState(
            dependency.getAspects().equals(AspectCollection.EMPTY),
            "Dependency with null Configuration cannot have aspects");
      }
      return dependency;
    }
  }

  /** Returns a new {@link Builder} to create {@link Dependency} instances. */
  public static Builder builder() {
    return new AutoValue_Dependency.Builder()
        .setAspects(AspectCollection.EMPTY)
        .setTransitionKeys(ImmutableList.of());
  }

  /** Returns the label of the target this dependency points to. */
  public abstract Label getLabel();

  /** Returns the explicit configuration intended for this dependency. */
  @Nullable
  public abstract BuildConfiguration getConfiguration();

  /**
   * Returns the set of aspects which should be evaluated and combined with the configured target
   * pointed to by this dependency.
   *
   * @see #getAspectConfiguration(AspectDescriptor)
   */
  public abstract AspectCollection getAspects();

  /** Returns the configuration an aspect should be evaluated with. */
  @Nullable
  public BuildConfiguration getAspectConfiguration(AspectDescriptor aspect) {
    return getConfiguration();
  }

  /**
   * Returns the keys of a configuration transition, if any exist, associated with this dependency.
   * See {@link ConfigurationTransition#apply} for more details. Normally, this returns an empty
   * list, when there was no configuration transition in effect, or one with a single entry, when
   * there was a specific configuration transition result that led to this. It may also return a
   * list with multiple entries if the dependency has a null configuration, yet the outgoing edge
   * has a split transition. In such cases all transition keys returned by the transition are tagged
   * to the dependency.
   */
  public abstract ImmutableList<String> getTransitionKeys();

  /** Returns the ConfiguredTargetKey needed to fetch this dependency. */
  public ConfiguredTargetKey getConfiguredTargetKey() {
    return ConfiguredTargetKey.builder()
        .setLabel(getLabel())
        .setConfiguration(getConfiguration())
        .build();
  }
}
