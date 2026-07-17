// Copyright 2025 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skyframe;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.devtools.build.lib.packages.MacroInstance;
import com.google.devtools.build.lib.packages.PackagePieceIdentifier;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

/**
 * A Skyframe value representing the declaration of a symbolic macro instance.
 *
 * <p>The corresponding {@link com.google.devtools.build.skyframe.SkyKey} is {@link
 * MacroInstanceValue.Key}.
 *
 * <p>The purpose of this class is to store potentially large data (macro attribute values and the
 * Starlark stack) in a skyvalue rather than directly in a {@link PackagePieceValue.ForMacro}'s
 * skykey.
 */
@AutoCodec
public record MacroInstanceValue(MacroInstance macroInstance) implements SkyValue {
  public MacroInstanceValue {
    checkNotNull(macroInstance);
  }

  /** A SkyKey for a {@link MacroInstanceValue}. */
  @AutoCodec
  public static record Key(PackagePieceIdentifier packagePieceId, String macroInstanceName)
      implements SkyKey {
    public Key {
      checkNotNull(packagePieceId);
      checkNotNull(macroInstanceName);
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.MACRO_INSTANCE;
    }
  }
}
