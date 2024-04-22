// Copyright 2022 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

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

  public abstract ImmutableMap<String, CompiledModuleFile> includeLabelToCompiledModuleFile();

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
      ImmutableMap<String, CompiledModuleFile> includeLabelToCompiledModuleFile,
      Map<String, ModuleOverride> moduleOverrides,
      boolean ignoreDevDeps,
      LockfileMode lockfileMode,
      StarlarkSemantics starlarkSemantics) {
    return new AutoValue_BazelModTidyValue(
        buildozer,
        includeLabelToCompiledModuleFile,
        ImmutableMap.copyOf(moduleOverrides),
        ignoreDevDeps,
        lockfileMode,
        starlarkSemantics);
  }
}
