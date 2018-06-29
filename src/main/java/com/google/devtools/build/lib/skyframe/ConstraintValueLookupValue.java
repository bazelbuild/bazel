package com.google.devtools.build.lib.skyframe;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Interner;
import com.google.devtools.build.lib.analysis.platform.ConstraintValueInfo;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/** A value which allows lookup of {@link ConstraintValueInfo} data. */
@AutoCodec
@AutoValue
public abstract class ConstraintValueLookupValue implements SkyValue {

  /** Returns the {@link SkyKey} for {@link ConstraintValueLookupValue}s. */
  public static Key key(Iterable<ConfiguredTargetKey> constraintValueKeys) {
    return Key.of(constraintValueKeys);
  }

  /** A {@link SkyKey} for {@code ConstraintLookupValue}. */
  @AutoCodec
  static class Key implements SkyKey {
    private static final Interner<Key> interners = BlazeInterners.newWeakInterner();

    private final ImmutableList<ConfiguredTargetKey> constraintValueKeys;

    private Key(Iterable<ConfiguredTargetKey> constraintValueKeys) {
      this.constraintValueKeys = ImmutableList.copyOf(constraintValueKeys);
    }

    @AutoCodec.Instantiator
    @AutoCodec.VisibleForSerialization
    static Key of(Iterable<ConfiguredTargetKey> constraintValueKeys) {
      return interners.intern(new Key(constraintValueKeys));
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.CONSTRAINT_VALUE_LOOKUP;
    }

    ImmutableList<ConfiguredTargetKey> constraintValueKeys() {
      return constraintValueKeys;
    }

    @Override
    public boolean equals(Object obj) {
      if (!(obj instanceof Key)) {
        return false;
      }
      Key that = (Key) obj;
      return Objects.equals(this.constraintValueKeys, that.constraintValueKeys);
    }

    @Override
    public int hashCode() {
      return constraintValueKeys.hashCode();
    }
  }

  @AutoCodec.Instantiator
  public static ConstraintValueLookupValue create(List<ConstraintValueInfo> constraintValues) {
    return new AutoValue_ConstraintValueLookupValue(constraintValues);
  }

  public abstract List<ConstraintValueInfo> constraintValues();
}
