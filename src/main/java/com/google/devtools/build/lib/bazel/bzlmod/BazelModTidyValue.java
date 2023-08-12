package com.google.devtools.build.lib.bazel.bzlmod;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions.LockfileMode;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationConstant;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Map;
import net.starlark.java.eval.StarlarkSemantics;

/** All Skyframe information required for the {@code bazel mod tidy} command. */
@AutoValue
public abstract class BazelModTidyValue implements SkyValue {

  @SerializationConstant public static final SkyKey KEY = () -> SkyFunctions.BAZEL_MOD_TIDY;

  /** The path of the buildozer binary provided by the "buildozer" module. */
  public abstract Path buildozer();

  /** The value of {@link ModuleFileFunction#MODULE_OVERRIDES}. */
  public abstract ImmutableMap<String, ModuleOverride> moduleOverrides();

  /** The value of {@link ModuleFileFunction#IGNORE_DEV_DEPS}. */
  public abstract boolean ignoreDevDeps();

  /** The value of {@link BazelLockFileFunction#LOCKFILE_MODE}. */
  public abstract LockfileMode lockfileMode();

  /**
   * The value of {@link
   * com.google.devtools.build.lib.skyframe.PrecomputedValue#STARLARK_SEMANTICS}.
   */
  public abstract StarlarkSemantics starlarkSemantics();

  static BazelModTidyValue create(
      Path buildozer,
      Map<String, ModuleOverride> moduleOverrides,
      boolean ignoreDevDeps,
      LockfileMode lockfileMode,
      StarlarkSemantics starlarkSemantics) {
    return new AutoValue_BazelModTidyValue(
        buildozer,
        ImmutableMap.copyOf(moduleOverrides),
        ignoreDevDeps,
        lockfileMode,
        starlarkSemantics);
  }
}
