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

package com.google.devtools.build.lib.rules.platform;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.platform.ConstraintSettingInfo;
import com.google.devtools.build.lib.analysis.platform.ConstraintValueInfo;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.analysis.platform.ToolchainInfo;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.ClassObjectConstructor;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkSignature;
import com.google.devtools.build.lib.syntax.BuiltinFunction;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.SkylarkDict;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkSignatureProcessor;
import com.google.devtools.build.lib.syntax.Type.ConversionException;

/** Skylark namespace used to interact with Blaze's platform APIs. */
@SkylarkModule(
  name = "platform_common",
  doc = "Functions for Skylark to interact with Blaze's platform APIs."
)
public class PlatformCommon {

  @SkylarkCallable(
    name = PlatformInfo.SKYLARK_NAME,
    doc = "The key used to retrieve the provider containing platform_info's value.",
    structField = true
  )
  public ClassObjectConstructor getPlatformInfoConstructor() {
    return PlatformInfo.SKYLARK_CONSTRUCTOR;
  }

  @SkylarkCallable(
    name = ConstraintSettingInfo.SKYLARK_NAME,
    doc = "The key used to retrieve the provider containing constraint_setting_info's value.",
    structField = true
  )
  public ClassObjectConstructor getConstraintSettingInfoConstructor() {
    return ConstraintSettingInfo.SKYLARK_CONSTRUCTOR;
  }

  @SkylarkCallable(
    name = ConstraintValueInfo.SKYLARK_NAME,
    doc = "The key used to retrieve the provider containing constraint_value_info's value.",
    structField = true
  )
  public ClassObjectConstructor getConstraintValueInfoConstructor() {
    return ConstraintValueInfo.SKYLARK_CONSTRUCTOR;
  }

  @SkylarkCallable(
    name = ToolchainInfo.SKYLARK_NAME,
    doc = "The key used to retrieve the provider containing toolchain data.",
    structField = true
  )
  public ClassObjectConstructor getToolchainInfoConstructor() {
    return ToolchainInfo.SKYLARK_CONSTRUCTOR;
  }

  @SkylarkSignature(
    name = "toolchain",
    doc =
        "<i>(Experimental)</i> "
            + "Returns a toolchain provider that can be configured to provide rule implementations "
            + "access to needed configuration.",
    objectType = PlatformCommon.class,
    returnType = ToolchainInfo.class,
    parameters = {
      @Param(name = "self", type = PlatformCommon.class, doc = "the platform_rules instance"),
      @Param(
        name = "exec_compatible_with",
        type = SkylarkList.class,
        generic1 = TransitiveInfoCollection.class,
        defaultValue = "[]",
        named = true,
        positional = false,
        doc = "Constraints the platform must fulfill to execute this toolchain."
      ),
      @Param(
        name = "target_compatible_with",
        type = SkylarkList.class,
        generic1 = TransitiveInfoCollection.class,
        defaultValue = "[]",
        named = true,
        positional = false,
        doc = "Constraints fulfilled by the target platform for this toolchain."
      ),
    },
    extraKeywords =
        @Param(
          name = "toolchainData",
          doc = "Extra information stored for the consumer of the toolchain."
        ),
    useLocation = true
  )
  private static final BuiltinFunction createToolchain =
      new BuiltinFunction("toolchain") {
        @SuppressWarnings("unchecked")
        public ToolchainInfo invoke(
            PlatformCommon self,
            SkylarkList<TransitiveInfoCollection> execCompatibleWith,
            SkylarkList<TransitiveInfoCollection> targetCompatibleWith,
            SkylarkDict<String, Object> skylarkToolchainData,
            Location loc)
            throws ConversionException, EvalException {

          Iterable<ConstraintValueInfo> execConstraints =
              ConstraintValue.constraintValues(execCompatibleWith);
          Iterable<ConstraintValueInfo> targetConstraints =
              ConstraintValue.constraintValues(targetCompatibleWith);
          ImmutableMap<String, Object> toolchainData =
              ImmutableMap.copyOf(
                  SkylarkDict.castSkylarkDictOrNoneToDict(
                      skylarkToolchainData, String.class, Object.class, "toolchainData"));

          return new ToolchainInfo(execConstraints, targetConstraints, toolchainData, loc);
        }
      };

  static {
    SkylarkSignatureProcessor.configureSkylarkFunctions(PlatformCommon.class);
  }
}
