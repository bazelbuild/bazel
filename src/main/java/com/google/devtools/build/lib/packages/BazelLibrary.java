// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.packages;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.syntax.Environment.GlobalFrame;
import com.google.devtools.build.lib.syntax.MethodLibrary;
import com.google.devtools.build.lib.syntax.Runtime;

/**
 * A helper class containing built in skylark functions for Bazel (BUILD files and .bzl files).
 */
public class BazelLibrary {

  /** A global frame containing pure Skylark builtins and some Bazel builtins. */
  @AutoCodec public static final GlobalFrame GLOBALS = createGlobals();

  private static GlobalFrame createGlobals() {
    ImmutableMap.Builder<String, Object> builder = ImmutableMap.builder();

    Runtime.addConstantsToBuilder(builder);
    MethodLibrary.addBindingsToBuilder(builder);

    return GlobalFrame.createForBuiltins(builder.build());
  }
}
