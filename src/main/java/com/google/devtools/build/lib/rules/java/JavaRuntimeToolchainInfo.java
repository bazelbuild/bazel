// Copyright 2018 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.rules.java;

import static java.util.Objects.requireNonNull;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.platform.ToolchainInfo;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;

/**
 * A wrapper class for {@link JavaRuntimeInfo} that can be used to expose it to the toolchain
 * resolution system.
 */
@ThreadSafety.Immutable
@AutoCodec
public final class JavaRuntimeToolchainInfo extends ToolchainInfo {
  private final JavaRuntimeInfo javaRuntime;

  @AutoCodec.Instantiator
  public JavaRuntimeToolchainInfo(JavaRuntimeInfo javaRuntime) {
    super(ImmutableMap.of(), Location.BUILTIN);
    this.javaRuntime = requireNonNull(javaRuntime);
  }

  public JavaRuntimeInfo javaRuntime() {
    return javaRuntime;
  }
}
