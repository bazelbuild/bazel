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

import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Objects;

/** A return value of {@link FlagSetFunction} */
public class FlagSetValue implements SkyValue {

  private final BuildOptions topLevelBuildOptions;

  /** Key for {@link FlagSetValue} based on the raw flags. */
  @ThreadSafety.Immutable
  @AutoCodec
  public static final class Key implements SkyKey {
    private static final SkyKeyInterner<Key> interner = SkyKey.newInterner();
    // private final String sclFile;
    private final PathFragment projectFile;
    private final String sclConfig;
    private final BuildOptions targetOptions;

    public Key(PathFragment projectFile, String sclConfig, BuildOptions targetOptions) {
      this.projectFile = projectFile;
      this.sclConfig = sclConfig;
      this.targetOptions = targetOptions;
    }

    public static Key create(
        PathFragment projectFile, String sclConfig, BuildOptions targetOptions) {
      return interner.intern(new Key(projectFile, sclConfig, targetOptions));
    }

    public PathFragment getProjectFile() {
      return projectFile;
    }

    public String getSclConfig() {
      return sclConfig;
    }

    public BuildOptions getTargetOptions() {
      return targetOptions;
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
          && Objects.equals(targetOptions, key.targetOptions);
    }

    @Override
    public int hashCode() {
      return Objects.hash(projectFile, sclConfig, targetOptions);
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
