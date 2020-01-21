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

package com.google.devtools.build.lib.skylarkbuildapi.platform;

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.skylarkbuildapi.core.StructApi;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkConstructor;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.syntax.Dict;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Sequence;
import com.google.devtools.build.lib.syntax.StarlarkSemantics.FlagIdentifier;
import java.util.Map;

/** Info object representing data about a specific platform. */
@SkylarkModule(
    name = "PlatformInfo",
    doc =
        "Provides access to data about a specific platform. "
            + PlatformInfoApi.EXPERIMENTAL_WARNING,
    category = SkylarkModuleCategory.PROVIDER)
public interface PlatformInfoApi<
        ConstraintSettingInfoT extends ConstraintSettingInfoApi,
        ConstraintValueInfoT extends ConstraintValueInfoApi>
    extends StructApi {

  String EXPERIMENTAL_WARNING =
      "<i>Note: This API is experimental and may change at any time. It is disabled by default, "
          + "but may be enabled with <code>--experimental_platforms_api</code></i>";

  @SkylarkCallable(
      name = "label",
      doc = "The label of the target that created this platform.",
      structField = true,
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_PLATFORM_API)
  Label label();

  @SkylarkCallable(
      name = "constraints",
      doc =
          "The <a href=\"ConstraintValueInfo.html\">ConstraintValueInfo</a> instances that define "
              + "this platform.",
      structField = true,
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_PLATFORM_API)
  ConstraintCollectionApi<ConstraintSettingInfoT, ConstraintValueInfoT> constraints();

  @SkylarkCallable(
      name = "remoteExecutionProperties",
      doc = "Properties that are available for the use of remote execution.",
      structField = true,
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_PLATFORM_API)
  String remoteExecutionProperties();

  @SkylarkCallable(
      name = "exec_properties",
      doc = "Properties to configure a remote execution platform.",
      structField = true,
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_PLATFORM_API)
  Map<String, String> execProperties();

  /** Provider for {@link PlatformInfoApi} objects. */
  @SkylarkModule(name = "Provider", documented = false, doc = "")
  interface Provider<
          ConstraintSettingInfoT extends ConstraintSettingInfoApi,
          ConstraintValueInfoT extends ConstraintValueInfoApi,
          PlatformInfoT extends PlatformInfoApi<ConstraintSettingInfoT, ConstraintValueInfoT>>
      extends ProviderApi {

    @SkylarkCallable(
        name = "PlatformInfo",
        doc = "The <code>PlatformInfo</code> constructor.",
        documented = false,
        parameters = {
          @Param(
              name = "label",
              type = Label.class,
              named = true,
              positional = false,
              doc = "The label for this platform."),
          @Param(
              name = "parent",
              type = PlatformInfoApi.class,
              defaultValue = "None",
              named = true,
              positional = false,
              noneable = true,
              doc = "The parent of this platform."),
          @Param(
              name = "constraint_values",
              type = Sequence.class,
              defaultValue = "[]",
              generic1 = ConstraintValueInfoApi.class,
              named = true,
              positional = false,
              doc = "The constraint values for the platform"),
          @Param(
              name = "exec_properties",
              type = Dict.class,
              defaultValue = "None",
              named = true,
              positional = false,
              noneable = true,
              doc = "The exec properties for the platform.")
        },
        selfCall = true,
        useLocation = true,
        enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_PLATFORM_API)
    @SkylarkConstructor(objectType = PlatformInfoApi.class, receiverNameForDoc = "PlatformInfo")
    PlatformInfoT platformInfo(
        Label label,
        Object parent,
        Sequence<?> constraintValues,
        Object execProperties,
        Location location)
        throws EvalException;
  }
}
