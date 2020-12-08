package com.google.devtools.build.lib.rules.platform;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.PlatformOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptionsView;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.transitions.SplitTransition;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.starlarkbuildapi.platform.MultiPlatformTransitionApi;

public class MultiPlatformTransition implements SplitTransition, MultiPlatformTransitionApi {

  public static final MultiPlatformTransition MULTI_PLATFORM_TRANSITION = new MultiPlatformTransition();

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
