// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis.platform;

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import com.google.devtools.build.lib.skylarkbuildapi.platform.ToolchainTypeInfoApi;
import com.google.devtools.build.lib.syntax.Printer;
import java.util.Objects;

/** A provider that supplies information about a specific toolchain type. */
@Immutable
@AutoCodec
public class ToolchainTypeInfo extends NativeInfo implements ToolchainTypeInfoApi {
  /** Name used in Skylark for accessing this provider. */
  public static final String SKYLARK_NAME = "ToolchainTypeInfo";

  /** Provider singleton constant. */
  public static final BuiltinProvider<ToolchainTypeInfo> PROVIDER =
      new BuiltinProvider<ToolchainTypeInfo>(SKYLARK_NAME, ToolchainTypeInfo.class) {};

  private final Label typeLabel;

  public static ToolchainTypeInfo create(Label typeLabel, Location location) {
    return new ToolchainTypeInfo(typeLabel, location);
  }

  public static ToolchainTypeInfo create(Label typeLabel) {
    return create(typeLabel, Location.BUILTIN);
  }

  @VisibleForSerialization
  ToolchainTypeInfo(Label typeLabel, Location location) {
    super(PROVIDER, location);
    this.typeLabel = typeLabel;
  }

  @Override
  public Label typeLabel() {
    return typeLabel;
  }

  @Override
  public void repr(Printer printer) {
    printer.format("ToolchainTypeInfo(%s)", typeLabel);
  }

  @Override
  public int hashCode() {
    return Objects.hashCode(typeLabel);
  }

  @Override
  public boolean equals(Object other) {
    if (!(other instanceof ToolchainTypeInfo)) {
      return false;
    }

    ToolchainTypeInfo otherToolchainTypeInfo = (ToolchainTypeInfo) other;
    return Objects.equals(typeLabel, otherToolchainTypeInfo.typeLabel);
  }
}
