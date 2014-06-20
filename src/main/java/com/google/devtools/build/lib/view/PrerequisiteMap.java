// Copyright 2014 Google Inc. All rights reserved.
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

package com.google.devtools.build.lib.view;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableTable;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.view.config.BuildConfiguration;

import java.util.Objects;

import javax.annotation.concurrent.Immutable;

/**
 * A class that holds all the prerequisites for the analysis of a single target.
 */
@Immutable
public final class PrerequisiteMap {
  // This class is not meant to be outside of the analysis phase machinery and is only public
  // in order to be accessible from the .view.skyframe package.

  /**
   * A holder class for {@link BuildConfiguration} instances that allows {@code null} values,
   * because none of the Table implementations allow them.
   */
  private static final class ConfigurationHolder {
    private final BuildConfiguration configuration;

    public ConfigurationHolder(BuildConfiguration configuration) {
      this.configuration = configuration;
    }

    @Override
    public int hashCode() {
      return configuration == null ? 0 : configuration.hashCode();
    }

    @Override
    public boolean equals(Object o) {
      if (o == this) {
        return true;
      }
      if (!(o instanceof ConfigurationHolder)) {
        return false;
      }
      return Objects.equals(configuration, ((ConfigurationHolder) o).configuration);
    }
  }

  /**
   * A prerequisite used for analyzing a single configured target. It entails the respective
   * {@link TransitiveInfoCollection} and the dependent {@link Target} object.
   *
   * <p>Note that the configured target itself (ie. its
   * {@link RuleConfiguredTarget#initializationHook} should not be passed the dependent
   * {@link Target} object. It is purely there to be used in {@link RuleContext} to do the
   * dependency filtering.
   *
   * <p>It would be better to do said filtering in {@link ConfiguredTargetGraph}, but it is quite
   * hard to make it both correct and fast enough when incremental analysis is in effect.
   * Therefore, we pass the dependent {@link Target} objects for the time being and rely on
   * {@link RuleContext} not to forward that to the actual configured target classes.
   */
  public static final class Prerequisite {
    private final TransitiveInfoCollection transitiveInfoCollection;
    private final Target target;

    public Prerequisite(TransitiveInfoCollection transitiveInfoCollection, Target target) {
      this.transitiveInfoCollection = Preconditions.checkNotNull(transitiveInfoCollection);
      this.target = Preconditions.checkNotNull(target);
    }

    public TransitiveInfoCollection getTransitiveInfoCollection() {
      return transitiveInfoCollection;
    }

    public Target getTarget() {
      return target;
    }
  }

  private final ImmutableTable<Label, ConfigurationHolder, Prerequisite> prerequisites;

  private PrerequisiteMap(
      ImmutableTable<Label, ConfigurationHolder, Prerequisite> prerequisites) {
    this.prerequisites = prerequisites;
  }

  public Prerequisite get(Label label, BuildConfiguration configuration) {
    return prerequisites.get(label, new ConfigurationHolder(configuration));
  }

  /**
   * Builder class for {@link PrerequisiteMap}.
   */
  public static class Builder {
    private final boolean useProviderProxies;
    private ImmutableTable.Builder<Label, ConfigurationHolder, Prerequisite> prerequisites;

    public Builder(boolean useProviderProxies) {
      this.useProviderProxies = useProviderProxies;
      prerequisites = ImmutableTable.builder();
    }

    public void add(ConfiguredTarget configuredTarget) {
      TransitiveInfoCollection transitiveInfoCollection = useProviderProxies
          ? TransitiveInfoProxy.createCollectionProxy(configuredTarget)
          : configuredTarget;
      prerequisites.put(
          configuredTarget.getLabel(),
          new ConfigurationHolder(configuredTarget.getConfiguration()),
          new Prerequisite(transitiveInfoCollection, configuredTarget.getTarget()));
    }

    public PrerequisiteMap build() {
      return new PrerequisiteMap(prerequisites.build());
    }
  }
}
