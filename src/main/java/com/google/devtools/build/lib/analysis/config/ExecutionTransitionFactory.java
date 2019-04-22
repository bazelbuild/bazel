package com.google.devtools.build.lib.analysis.config;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.PlatformOptions;
import com.google.devtools.build.lib.analysis.config.transitions.PatchTransition;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionFactory;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.AttributeTransitionData;
import javax.annotation.Nullable;

public class ExecutionTransitionFactory implements TransitionFactory<AttributeTransitionData> {

  @Override
  public PatchTransition create(AttributeTransitionData data) {
    return new ExecutionTransition(data.executionPlatform());
  }

  @Override
  public boolean isHost() {
    return false;
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

    @Override
    public BuildOptions patch(BuildOptions options) {
      if (executionPlatform == null) {
        // No execution platform is known, so don't change anything.
        return options;
      }

      // Start by converting to host options.
      BuildOptions execConfiguration = options.createHostOptions();

      // Then unset isHost, if CoreOptions is available.
      if (execConfiguration.get(CoreOptions.class) != null) {
        execConfiguration.get(CoreOptions.class).isHost = false;
      }

      // Then set the target to the saved execution platform if there is one.
      if (execConfiguration.get(PlatformOptions.class) != null) {
        execConfiguration.get(PlatformOptions.class).platforms =
            ImmutableList.of(executionPlatform);
      }

      return execConfiguration;
    }
  }
}
