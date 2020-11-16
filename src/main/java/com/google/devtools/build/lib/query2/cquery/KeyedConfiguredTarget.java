package com.google.devtools.build.lib.query2.cquery;

import com.google.auto.value.AutoValue;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.BuildConfigurationValue;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import javax.annotation.Nullable;

@AutoValue
public abstract class KeyedConfiguredTarget {

  public static KeyedConfiguredTarget create(
      ConfiguredTargetKey key, ConfiguredTarget configuredTarget) {
    return new AutoValue_KeyedConfiguredTarget(key, configuredTarget);
  }

  @Nullable
  public abstract ConfiguredTargetKey key();

  public abstract ConfiguredTarget configuredTarget();

  public Label label() {
    return configuredTarget().getOriginalLabel();
  }

  public BuildConfigurationValue.Key configurationKey() {
    return configuredTarget().getConfigurationKey();
  }

  public String configurationChecksum() {
    return configuredTarget().getConfigurationChecksum();
  }
}
