// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.config;

import static com.google.common.base.Strings.nullToEmpty;

import com.google.common.base.Verify;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Objects;
import javax.annotation.Nullable;

/** A return value of {@link FlagSetFunction} */
public class FlagSetValue implements SkyValue {

  private final BuildOptions topLevelBuildOptions;

  /** Key for {@link FlagSetValue} based on the raw flags. */
  @ThreadSafety.Immutable
  @AutoCodec
  public static final class Key implements SkyKey {
    private static final SkyKeyInterner<Key> interner = SkyKey.newInterner();
    private final Label projectFile;
    private final String sclConfig;
    private final BuildOptions targetOptions;
    private final ImmutableSet<String> userOptions;

    private final boolean enforceCanonical;

    public Key(
        Label projectFile,
        @Nullable String sclConfig,
        BuildOptions targetOptions,
        ImmutableSet<String> userOptions,
        boolean enforceCanonical) {
      this.projectFile = Verify.verifyNotNull(projectFile);
      this.sclConfig = nullToEmpty(sclConfig);
      this.targetOptions = Verify.verifyNotNull(targetOptions);
      this.userOptions = Verify.verifyNotNull(userOptions);
      this.enforceCanonical = enforceCanonical;
    }

    public static Key create(
        Label projectFile,
        String sclConfig,
        BuildOptions targetOptions,
        ImmutableSet<String> userOptions,
        boolean enforceCanonical) {
      return interner.intern(
          new Key(projectFile, sclConfig, targetOptions, userOptions, enforceCanonical));
    }

    public Label getProjectFile() {
      return projectFile;
    }

    public String getSclConfig() {
      return sclConfig;
    }

    public BuildOptions getTargetOptions() {
      return targetOptions;
    }

    public ImmutableSet<String> getUserOptions() {
      return userOptions;
    }

    /**
     * Whether {@code --scl_config} must match an officially supported project configuration. See
     * {@link com.google.devtools.build.lib.buildtool.BuildRequestOptions#enforceProjectConfigs}.
     */
    public boolean enforceCanonical() {
      return enforceCanonical;
    }

    @Override
    public SkyKeyInterner<?> getSkyKeyInterner() {
      return interner;
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.FLAG_SET;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (o == null || getClass() != o.getClass()) {
        return false;
      }
      Key key = (Key) o;
      return Objects.equals(projectFile, key.projectFile)
          && Objects.equals(sclConfig, key.sclConfig)
          && Objects.equals(targetOptions, key.targetOptions)
          && Objects.equals(userOptions, key.userOptions)
          && (enforceCanonical == key.enforceCanonical);
    }

    @Override
    public int hashCode() {
      return Objects.hash(projectFile, sclConfig, targetOptions, userOptions, enforceCanonical);
    }
  }

  public static FlagSetValue create(BuildOptions buildOptions) {
    return new FlagSetValue(buildOptions);
  }

  public FlagSetValue(BuildOptions buildOptions) {
    this.topLevelBuildOptions = buildOptions;
  }

  public BuildOptions getTopLevelBuildOptions() {
    return topLevelBuildOptions;
  }
}
