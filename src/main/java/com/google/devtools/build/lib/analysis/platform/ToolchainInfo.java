// Copyright 2017 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.packages.NativeProvider;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skylarkbuildapi.platform.ToolchainInfoApi;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.FunctionSignature;
import com.google.devtools.build.lib.syntax.SkylarkDict;
import com.google.devtools.build.lib.syntax.SkylarkType;
import java.util.Map;

/**
 * A provider that supplies information about a specific language toolchain, including what platform
 * constraints are required for execution and for the target platform.
 */
@AutoCodec
@Immutable
public class ToolchainInfo extends NativeInfo implements ToolchainInfoApi {

  /** Name used in Skylark for accessing this provider. */
  public static final String SKYLARK_NAME = "ToolchainInfo";

  private static final FunctionSignature.WithValues<Object, SkylarkType> SIGNATURE =
      FunctionSignature.WithValues.create(
          FunctionSignature.of(
              /*numMandatoryPositionals=*/ 0,
              /*numOptionalPositionals=*/ 0,
              /*numMandatoryNamedOnly*/ 0,
              /*starArg=*/ false,
              /*kwArg=*/ true,
              /*names=*/ "data"),
          /*defaultValues=*/ null,
          /*types=*/ ImmutableList.<SkylarkType>of(SkylarkType.DICT));

  /** Skylark constructor and identifier for this provider. */
  @AutoCodec
  public static final NativeProvider<ToolchainInfo> PROVIDER =
      new NativeProvider<ToolchainInfo>(ToolchainInfo.class, SKYLARK_NAME, SIGNATURE) {
        @Override
        protected ToolchainInfo createInstanceFromSkylark(
            Object[] args, Environment env, Location loc) throws EvalException {
          Map<String, Object> data =
              SkylarkDict.castSkylarkDictOrNoneToDict(args[0], String.class, Object.class, "data");
          return ToolchainInfo.create(data, loc);
        }
      };

  @AutoCodec.Instantiator
  public ToolchainInfo(Map<String, Object> values, Location location) {
    super(PROVIDER, ImmutableMap.copyOf(values), location);
  }

  public static ToolchainInfo create(Map<String, Object> toolchainData) {
    return create(toolchainData, Location.BUILTIN);
  }

  public static ToolchainInfo create(Map<String, Object> toolchainData, Location loc) {
    return new ToolchainInfo(toolchainData, loc);
  }

  /** Add make variables to be exported to dependers. */
  public void addGlobalMakeVariables(ImmutableMap.Builder<String, String> globalMakeEnvBuilder) {}
}
