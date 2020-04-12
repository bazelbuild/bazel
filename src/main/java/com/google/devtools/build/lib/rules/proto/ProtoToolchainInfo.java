// Copyright 2020 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.platform.ToolchainInfo;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skylarkbuildapi.proto.ProtoToolchainInfoApi;
import com.google.devtools.build.lib.syntax.Location;

/**
 * Information about the tools used by the <code>proto_*</code> and <code>LANG_proto_*</code> rules.
 */
@Immutable
@AutoCodec
public class ProtoToolchainInfo extends ToolchainInfo implements ProtoToolchainInfoApi {
  @AutoCodec.Instantiator
  public ProtoToolchainInfo() {
    super(ImmutableMap.of(), Location.BUILTIN);
  }
}
