package com.google.devtools.build.lib.analysis;

import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransition;
import com.google.devtools.build.lib.analysis.config.transitions.PatchTransition;
import com.google.devtools.build.lib.cmdline.Label;

import javax.annotation.Nullable;

public class ToolchainTransition implements PatchTransition {

  public static ConfigurationTransition create(@Nullable ToolchainContext toolchainContext) {
    return new ToolchainTransition(toolchainContext);
  }

  private final Label execPlatform;

  private ToolchainTransition(ToolchainContext toolchainContext) {
    this.execPlatform = toolchainContext.executionPlatform().label();
  }

  @Override
  public BuildOptions patch(BuildOptions options) {
    BuildOptions result = options.clone();

    // Store the exec platform label for later use.
    PlatformOptions platformOptions = result.get(PlatformOptions.class);
    platformOptions.toolchainExecPlatformPassthrough = this.execPlatform;

    return result;
  }
}
