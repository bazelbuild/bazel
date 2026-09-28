// Copyright 2026 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.bzlmod;

import static java.util.Objects.requireNonNull;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationConstant;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

/**
 * Resolves the buildozer binary path and the set of root module file paths, without evaluating
 * module extensions.
 *
 * <p>This is a lightweight alternative to {@link BazelModTidyValue} for commands that only need to
 * run buildozer (e.g. {@code bazel mod upgrade}) without computing module extension fixups.
 *
 * @param buildozer The path of the buildozer binary provided by the "buildozer" module.
 * @param moduleFilePaths The set of paths to the root MODULE.bazel file and all its includes.
 */
public record BuildozerBinaryValue(Path buildozer, ImmutableSet<PathFragment> moduleFilePaths)
    implements SkyValue {
  public BuildozerBinaryValue {
    requireNonNull(buildozer, "buildozer");
    requireNonNull(moduleFilePaths, "moduleFilePaths");
  }

  @SerializationConstant
  public static final SkyKey KEY = () -> SkyFunctions.BUILDOZER_BINARY;
}
