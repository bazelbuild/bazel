// Copyright 2021 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.proto;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.platform.ToolchainInfo;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;

/** Information about the tools used by the {@code proto_*} and {@code LANG_proto_*} rules. */
@Immutable
@AutoValue
public abstract class ProtoToolchainInfo {
  private static final String PROTOC_ATTR_NAME = "protoc";
  private static final String PROTOC_OPTIONS_ATTR_NAME = "protoc_options";

  /** The {@code protoc} binary to use for proto actions. */
  public abstract FilesToRunProvider getProtoc();

  /** Additional options to pass to {@code protoc}. */
  public abstract ImmutableList<String> getProtocOptions();

  /**
   * Constructs a {@link ProtoToolchainInfo} from {@link ToolchainInfo} (i.e. as returned by {@code
   * proto_toolchain} from {@code @rules_proto}).
   */
  public static ProtoToolchainInfo fromToolchainInfo(ToolchainInfo toolchain) throws EvalException {
    FilesToRunProvider protoc = getValue(toolchain, PROTOC_ATTR_NAME, FilesToRunProvider.class);
    ImmutableList<String> protocOptions =
        Sequence.cast(
                getValue(toolchain, PROTOC_OPTIONS_ATTR_NAME, Object.class),
                String.class,
                PROTOC_OPTIONS_ATTR_NAME)
            .getImmutableList();
    return new AutoValue_ProtoToolchainInfo(protoc, protocOptions);
  }

  private static <T> T getValue(ToolchainInfo toolchain, String name, Class<T> clazz)
      throws EvalException {
    T value = toolchain.getValue(name, clazz);
    if (value == null) {
      throw Starlark.errorf("Proto toolchain does not have mandatory field '%s'", name);
    }
    return value;
  }
}
