package com.google.devtools.build.lib.skyframe;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Interner;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Map;
import java.util.Objects;

/** A value which allows lookup of {@link PlatformInfo} data. */
@AutoCodec
@AutoValue
public abstract class PlatformLookupValue implements SkyValue {

  /** Returns the {@link SkyKey} for {@link RegisteredToolchainsValue}s. */
  public static Key key(Iterable<ConfiguredTargetKey> platformKeys) {
    return Key.of(platformKeys);
  }

  /** A {@link SkyKey} for {@code PlatformLookupValue}. */
  @AutoCodec
  static class Key implements SkyKey {
    private static final Interner<Key> interners = BlazeInterners.newWeakInterner();

    private final ImmutableList<ConfiguredTargetKey> platformKeys;

    private Key(Iterable<ConfiguredTargetKey> platformKeys) {
      this.platformKeys = ImmutableList.copyOf(platformKeys);
    }

    @AutoCodec.Instantiator
    @AutoCodec.VisibleForSerialization
    static Key of(Iterable<ConfiguredTargetKey> platformKeys) {
      return interners.intern(new Key(platformKeys));
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.PLATFORM_LOOKUP;
    }

    ImmutableList<ConfiguredTargetKey> platformKeys() {
      return platformKeys;
    }

    @Override
    public boolean equals(Object obj) {
      if (!(obj instanceof Key)) {
        return false;
      }
      Key that = (Key) obj;
      return Objects.equals(this.platformKeys, that.platformKeys);
    }

    @Override
    public int hashCode() {
      return platformKeys.hashCode();
    }
  }

  @AutoCodec.Instantiator
  public static PlatformLookupValue create(Map<ConfiguredTargetKey, PlatformInfo> platforms) {
    return new AutoValue_PlatformLookupValue(platforms);
  }

  public abstract Map<ConfiguredTargetKey, PlatformInfo> platforms();
}
