package com.google.devtools.build.lib.skyframe;

import com.google.auto.value.AutoValue;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.events.ExtendedEventHandler;

@AutoValue
public abstract class HostPlatformInfoEvent implements ExtendedEventHandler.Postable {

  public abstract PlatformInfo hostPlatformInfo();

  public static HostPlatformInfoEvent create(PlatformInfo hostPlatformInfo) {
    return new AutoValue_HostPlatformInfoEvent(hostPlatformInfo);
  }

  @Override
  public boolean storeForReplay() {
    return true;
  }
}
