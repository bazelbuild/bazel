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
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationConstant;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.List;

/** All Skyframe information required for the {@code bazel mod tidy} command. */
@AutoValue
public abstract class BazelModTidyValue implements SkyValue {

  @SerializationConstant public static final SkyKey KEY = () -> SkyFunctions.BAZEL_MOD_TIDY;

  /** Buildozer fixups for incorrect use_repo declarations by the root module. */
  public abstract ImmutableList<RootModuleFileFixup> fixups();

  /** The path of the buildozer binary provided by the "buildozer" module. */
  public abstract Path buildozer();

  /** The set of paths to the root MODULE.bazel file and all its includes. */
  public abstract ImmutableSet<PathFragment> moduleFilePaths();

  static BazelModTidyValue create(
      List<RootModuleFileFixup> fixups,
      Path buildozer,
      ImmutableSet<PathFragment> moduleFilePaths) {
    return new AutoValue_BazelModTidyValue(
        ImmutableList.copyOf(fixups), buildozer, moduleFilePaths);
  }
}
