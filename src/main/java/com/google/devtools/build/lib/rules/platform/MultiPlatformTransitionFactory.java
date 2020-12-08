package com.google.devtools.build.lib.rules.platform;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.PlatformOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptionsView;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransition;
import com.google.devtools.build.lib.analysis.config.transitions.SplitTransition;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionFactory;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.AttributeTransitionData;
import com.google.devtools.build.lib.starlarkbuildapi.platform.MultiPlatformTransitionApi;

public class MultiPlatformTransitionFactory implements TransitionFactory<AttributeTransitionData>, MultiPlatformTransitionApi {

  @Override
  public ConfigurationTransition create(AttributeTransitionData data) {
    return new MultiPlatformTransition();
  }

  @Override
  public boolean isSplit() {
    return true;
  }

  private static final class MultiPlatformTransition implements SplitTransition {
    @Override
    public ImmutableSet<Class<? extends FragmentOptions>> requiresOptionFragments() {
      return ImmutableSet.of(
          CoreOptions.class,
          PlatformOptions.class);
    }

    @Override
    public ImmutableMap<String, BuildOptions> split(
        BuildOptionsView buildOptions, EventHandler eventHandler) {
      // TODO: implement me
      return ImmutableMap.of(
          buildOptions.get(CoreOptions.class).cpu, buildOptions.underlying());
    }
  }
}
