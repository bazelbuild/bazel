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

package com.google.devtools.build.lib.skylarkbuildapi.apple;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.syntax.StarlarkValue;
import javax.annotation.Nullable;

/** A configuration fragment for Objective C. */
@SkylarkModule(
    name = "objc",
    category = SkylarkModuleCategory.CONFIGURATION_FRAGMENT,
    doc = "A configuration fragment for Objective-C.")
public interface ObjcConfigurationApi<ApplePlatformTypeApiT extends ApplePlatformTypeApi>
    extends StarlarkValue {

  @SkylarkCallable(
      name = "ios_simulator_device",
      structField = true,
      allowReturnNones = true,
      doc = "The type of device (e.g. 'iPhone 6') to use when running on the simulator.")
  String getIosSimulatorDevice();

  @SkylarkCallable(
      name = "ios_simulator_version",
      structField = true,
      allowReturnNones = true,
      doc = "The SDK version of the iOS simulator to use when running on the simulator.")
  DottedVersionApi<?> getIosSimulatorVersion();

  @SkylarkCallable(
      name = "simulator_device_for_platform_type",
      allowReturnNones = true,
      doc = "The type of device (e.g., 'iPhone 6' to simulate when running on the simulator.",
      parameters = {
        @Param(
            name = "platform_type",
            positional = true,
            named = false,
            type = ApplePlatformTypeApi.class,
            doc = "The apple platform type."),
      })
  String getSimulatorDeviceForPlatformType(ApplePlatformTypeApiT platformType);

  @SkylarkCallable(
      name = "simulator_version_for_platform_type",
      allowReturnNones = true,
      doc = "The SDK version of the simulator to use when running on the simulator.",
      parameters = {
        @Param(
            name = "platform_type",
            positional = true,
            named = false,
            type = ApplePlatformTypeApi.class,
            doc = "The apple platform type."),
      })
  DottedVersionApi<?> getSimulatorVersionForPlatformType(ApplePlatformTypeApiT platformType);

  @SkylarkCallable(
      name = "generate_dsym",
      doc = "Whether to generate debug symbol(.dSYM) artifacts.",
      structField = true)
  boolean generateDsym();

  @SkylarkCallable(
      name = "generate_linkmap",
      doc = "Whether to generate linkmap artifacts.",
      structField = true)
  boolean generateLinkmap();

  @SkylarkCallable(
      name = "run_memleaks",
      structField = true,
      doc = "Returns a boolean indicating whether memleaks should be run during tests or not.")
  boolean runMemleaks();

  @SkylarkCallable(
      name = "copts_for_current_compilation_mode",
      structField = true,
      doc =
          "Returns a list of default options to use for compiling Objective-C in the current "
              + "mode.")
  ImmutableList<String> getCoptsForCompilationMode();

  @SkylarkCallable(
      name = "copts",
      structField = true,
      doc =
          "Returns a list of options to use for compiling Objective-C.These options are applied"
              + " after any default options but before options specified in the attributes of the"
              + " rule.")
  ImmutableList<String> getCopts();

  @SkylarkCallable(
      name = "signing_certificate_name",
      structField = true,
      allowReturnNones = true,
      doc =
          "Returns the flag-supplied certificate name to be used in signing, or None if no such "
              + "certificate was specified.")
  @Nullable
  String getSigningCertName();

  @SkylarkCallable(
      name = "uses_device_debug_entitlements",
      structField = true,
      doc =
          "Returns whether device debug entitlements should be included when signing an "
              + "application.")
  boolean useDeviceDebugEntitlements();

  @SkylarkCallable(
      name = "enable_apple_binary_native_protos",
      structField = true,
      doc = "Returns whether apple_binary should generate and link protos natively.")
  boolean enableAppleBinaryNativeProtos();
}
