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
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.ClassObjectConstructor;
import com.google.devtools.build.lib.packages.NativeClassObjectConstructor;
import com.google.devtools.build.lib.packages.SkylarkClassObject;
import com.google.devtools.build.lib.packages.SkylarkProviderIdentifier;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.FunctionSignature;
import com.google.devtools.build.lib.syntax.SkylarkDict;
import com.google.devtools.build.lib.syntax.SkylarkType;
import java.util.Map;

/**
 * A provider that supplied information about a specific language toolchain, including what platform
 * constraints are required for execution and for the target platform.
 */
@SkylarkModule(
  name = "ToolchainInfo",
  doc = "Provides access to data about a specific toolchain.",
  category = SkylarkModuleCategory.PROVIDER
)
@Immutable
public class ToolchainInfo extends SkylarkClassObject {

  /** Name used in Skylark for accessing this provider. */
  public static final String SKYLARK_NAME = "ToolchainInfo";

  private static final FunctionSignature.WithValues<Object, SkylarkType> SIGNATURE =
      FunctionSignature.WithValues.create(
          FunctionSignature.of(
              /*numMandatoryPositionals=*/ 0,
              /*numOptionalPositionals=*/ 0,
              /*numMandatoryNamedOnly*/ 1,
              /*starArg=*/ false,
              /*kwArg=*/ true,
              /*names=*/ "type",
              "data"),
          /*defaultValues=*/ null,
          /*types=*/ ImmutableList.<SkylarkType>of(SkylarkType.of(Label.class), SkylarkType.DICT));

  /** Skylark constructor and identifier for this provider. */
  public static final ClassObjectConstructor SKYLARK_CONSTRUCTOR =
      new NativeClassObjectConstructor(SKYLARK_NAME, SIGNATURE) {
        @Override
        protected ToolchainInfo createInstanceFromSkylark(Object[] args, Location loc)
            throws EvalException {
          // Based on SIGNATURE above, the args are label, map.
          Label type = (Label) args[0];
          Map<String, Object> data =
              SkylarkDict.castSkylarkDictOrNoneToDict(args[1], String.class, Object.class, "data");
          return ToolchainInfo.create(type, data, loc);
        }
      };

  /** Identifier used to retrieve this provider from rules which export it. */
  public static final SkylarkProviderIdentifier SKYLARK_IDENTIFIER =
      SkylarkProviderIdentifier.forKey(SKYLARK_CONSTRUCTOR.getKey());

  private final Label type;

  private ToolchainInfo(Label type, Map<String, Object> toolchainData, Location loc) {
    super(
        SKYLARK_CONSTRUCTOR,
        ImmutableMap.<String, Object>builder().put("type", type).putAll(toolchainData).build(),
        loc);

    this.type = type;
  }

  public Label type() {
    return type;
  }

  public static ToolchainInfo create(Label type, Map<String, Object> toolchainData) {
    return create(type, toolchainData, Location.BUILTIN);
  }

  public static ToolchainInfo create(Label type, Map<String, Object> toolchainData, Location loc) {
    return new ToolchainInfo(type, toolchainData, loc);
  }
}
