package com.google.devtools.build.lib.rules.platform;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
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
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.AttributeTransitionData;
import com.google.devtools.build.lib.starlarkbuildapi.platform.MultiPlatformTransitionApi;
import java.util.List;
import javax.annotation.Nullable;

public class MultiPlatformTransitionFactory implements TransitionFactory<AttributeTransitionData>, MultiPlatformTransitionApi {

  @Override
  public ConfigurationTransition create(AttributeTransitionData data) {
    return MultiPlatformTransition.create(data.fatTargetPlatforms());
  }

  @Override
  public boolean isSplit() {
    return true;
  }

  @AutoValue
  static abstract class MultiPlatformTransition implements SplitTransition {

    public static MultiPlatformTransition create(List<Label> fatTargetLabels) {
      return new AutoValue_MultiPlatformTransitionFactory_MultiPlatformTransition(ImmutableList.copyOf(fatTargetLabels));
    }

    protected abstract ImmutableList<Label> fatTargetLabels();

    @Override
    public ImmutableSet<Class<? extends FragmentOptions>> requiresOptionFragments() {
      return ImmutableSet.of(
          CoreOptions.class,
          PlatformOptions.class);
    }

    @Override
    public ImmutableMap<String, BuildOptions> split(
        BuildOptionsView buildOptions, EventHandler eventHandler) {
      if (fatTargetLabels().isEmpty()) {
        return ImmutableMap.of(
            buildOptions.get(CoreOptions.class).cpu, buildOptions.underlying());
      }

      ImmutableMap.Builder<String, BuildOptions> transitions = ImmutableMap.builder();
      for (Label targetPlatform : fatTargetLabels()) {
        BuildOptionsView splitOptions = buildOptions.clone();
        splitOptions.get(PlatformOptions.class).platforms = ImmutableList.of(targetPlatform);
        splitOptions.get(CoreOptions.class).platformSuffix =
            String.format("-target-%X", targetPlatform.getCanonicalForm().hashCode());
        
        transitions.put(targetPlatform.toString(), splitOptions.underlying());
      }

      return transitions.build();
    }
  }
}
