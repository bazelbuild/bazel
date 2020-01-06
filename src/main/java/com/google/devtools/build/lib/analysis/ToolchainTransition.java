package com.google.devtools.build.lib.analysis;

import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransition;
import com.google.devtools.build.lib.analysis.config.transitions.PatchTransition;
import javax.annotation.Nullable;

public class ToolchainTransition implements PatchTransition {

  public static ConfigurationTransition create(@Nullable ToolchainContext toolchainContext) {
    return new ToolchainTransition();
  }

  @Override
  public BuildOptions patch(BuildOptions options) {
    // TODO(katre): Implement me.
    return options;
  }
}
