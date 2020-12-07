// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis.config;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.devtools.build.lib.analysis.ToolchainCollection.DEFAULT_EXEC_GROUP_NAME;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.PlatformOptions;
import com.google.devtools.build.lib.analysis.config.transitions.PatchTransition;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionFactory;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.AttributeTransitionData;
import com.google.devtools.build.lib.rules.config.FeatureFlagValue;
import com.google.devtools.build.lib.starlarkbuildapi.StarlarkConfigApi.ExecTransitionFactoryApi;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * {@link TransitionFactory} implementation which creates a {@link PatchTransition} which will
 * transition to a configuration suitable for building dependencies for the execution platform of
 * the depending target.
 */
public class ExecutionTransitionFactory
    implements TransitionFactory<AttributeTransitionData>, ExecTransitionFactoryApi {

  private final String execGroup;

  private ExecutionTransitionFactory(String execGroup) {
    this.execGroup = execGroup;
  }

  /**
   * Returns a new {@link ExecutionTransitionFactory} for the default {@link
   * com.google.devtools.build.lib.packages.ExecGroup}.
   */
  public static ExecutionTransitionFactory create() {
    return new ExecutionTransitionFactory(DEFAULT_EXEC_GROUP_NAME);
  }

  /**
   * Returns a new {@link ExecutionTransitionFactory} for the given {@link
   * com.google.devtools.build.lib.packages.ExecGroup}.
   */
  public static ExecutionTransitionFactory create(String execGroup) {
    return new ExecutionTransitionFactory(execGroup);
  }

  @Override
  public PatchTransition create(AttributeTransitionData data) {
    return new ExecutionTransition(data.executionPlatform());
  }

  public String getExecGroup() {
    return execGroup;
  }

  @Override
  public boolean isHost() {
    return false;
  }

  @Override
  public boolean isTool() {
    return true;
  }

  private static class ExecutionTransition implements PatchTransition {
    @Nullable private final Label executionPlatform;

    public ExecutionTransition(@Nullable Label executionPlatform) {
      this.executionPlatform = executionPlatform;
    }

    @Override
    public String getName() {
      return "exec";
    }

    @Override
    public boolean isHostTransition() {
      return false;
    }

    // We added this cache after observing an O(100,000)-node build graph that applied multiple exec
    // transitions on every node via an aspect. Before this cache, this produced O(500,000)
    // BuildOptions instances that consumed over 3 gigabytes of memory.
    private static final BuildOptionsCache<Label> cache = new BuildOptionsCache<>();

    @Override
    public ImmutableSet<Class<? extends FragmentOptions>> requiresOptionFragments() {
      return ImmutableSet.of(CoreOptions.class, PlatformOptions.class);
    }

    @Override
    public BuildOptions patch(BuildOptionsView options, EventHandler eventHandler) {
      if (executionPlatform == null) {
        // No execution platform is known, so don't change anything.
        return options.underlying();
      }
      return cache.applyTransition(
          options,
          // The execution platform impacts the output's --platform_suffix and --platforms flags.
          executionPlatform,
          () -> {
            // Start by converting to host options.
            BuildOptionsView execOptions =
                new BuildOptionsView(
                    options.underlying().createHostOptions(), requiresOptionFragments());

            // Then unset isHost, if CoreOptions is available.
            CoreOptions coreOptions =
                Preconditions.checkNotNull(execOptions.get(CoreOptions.class));
            coreOptions.isHost = false;
            coreOptions.isExec = true;
            coreOptions.outputDirectoryName = null;
            coreOptions.platformSuffix =
                String.format("-exec-%X", executionPlatform.getCanonicalForm().hashCode());

            // Then set the target to the saved execution platform if there is one.
            if (execOptions.get(PlatformOptions.class) != null) {
              execOptions.get(PlatformOptions.class).platforms =
                  ImmutableList.of(executionPlatform);
            }

            BuildOptions result = execOptions.underlying();
            // Remove any FeatureFlags that were set.
            ImmutableList<Label> featureFlags =
                execOptions.underlying().getStarlarkOptions().entrySet().stream()
                    .filter(entry -> entry.getValue() instanceof FeatureFlagValue)
                    .map(Map.Entry::getKey)
                    .collect(toImmutableList());
            if (!featureFlags.isEmpty()) {
              BuildOptions.Builder resultBuilder = result.toBuilder();
              featureFlags.stream().forEach(flag -> resultBuilder.removeStarlarkOption(flag));
              result = resultBuilder.build();
            }

            return result;
          });
    }
  }
}
