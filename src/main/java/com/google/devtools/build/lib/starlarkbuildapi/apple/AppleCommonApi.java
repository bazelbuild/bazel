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

package com.google.devtools.build.lib.starlarkbuildapi.apple;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import com.google.devtools.build.lib.starlarkbuildapi.SplitTransitionProviderApi;
import com.google.devtools.build.lib.starlarkbuildapi.StarlarkAspectApi;
import com.google.devtools.build.lib.starlarkbuildapi.StarlarkRuleContextApi;
import com.google.devtools.build.lib.starlarkbuildapi.apple.AppleStaticLibraryInfoApi.AppleStaticLibraryInfoProvider;
import com.google.devtools.build.lib.starlarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.StructApi;
import com.google.devtools.build.lib.starlarkbuildapi.platform.ConstraintValueInfoApi;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.StarlarkValue;

/** Interface for a module with useful functions for creating apple-related rule implementations. */
@StarlarkBuiltin(
    name = "apple_common",
    doc = "Functions for Starlark to access internals of the apple rule implementations.")
public interface AppleCommonApi<
        FileApiT extends FileApi,
        ConstraintValueT extends ConstraintValueInfoApi,
        StarlarkRuleContextT extends StarlarkRuleContextApi<ConstraintValueT>,
        ObjcProviderApiT extends ObjcProviderApi<?>,
        XcodeConfigInfoApiT extends XcodeConfigInfoApi<?, ?>,
        ApplePlatformApiT extends ApplePlatformApi>
    extends StarlarkValue {

  @StarlarkMethod(
      name = "apple_toolchain",
      doc = "Utilities for resolving items from the apple toolchain.")
  AppleToolchainApi<?> getAppleToolchain();

  @StarlarkMethod(
      name = "platform_type",
      doc =
          "An enum-like struct that contains the following fields corresponding to Apple platform "
              + "types:<br><ul><li><code>ios</code></li><li><code>macos</code></li>"
              + "<li><code>tvos</code></li><li><code>watchos</code></li></ul><p>These values can"
              + " be passed to methods that expect a platform type, like the 'apple' configuration"
              + " fragment's <a href='apple.html#multi_arch_platform'>multi_arch_platform</a>"
              + " method.<p>Example:<p><pre class='language-python'>\n"
              + "ctx.fragments.apple.multi_arch_platform(apple_common.platform_type.ios)\n"
              + "</pre>",
      structField = true)
  StructApi getPlatformTypeStruct();

  @StarlarkMethod(
      name = "platform",
      doc =
          "An enum-like struct that contains the following fields corresponding to Apple "
              + "platforms:<br><ul>"
              + "<li><code>ios_device</code></li>"
              + "<li><code>ios_simulator</code></li>"
              + "<li><code>macos</code></li>"
              + "<li><code>tvos_device</code></li>"
              + "<li><code>tvos_simulator</code></li>"
              + "<li><code>watchos_device</code></li>"
              + "<li><code>watchos_device</code></li>"
              + "</ul><p>"
              + "These values can be passed to methods that expect a platform, like "
              + "<a href='XcodeVersionConfig.html#sdk_version_for_platform'>"
              + "XcodeVersionConfig.sdk_version_for_platform</a>.",
      structField = true)
  StructApi getPlatformStruct();

  @StarlarkMethod(
      name = "XcodeProperties",
      doc =
          "The constructor/key for the <code>XcodeVersionProperties</code> provider.<p>"
              + "If a target propagates the <code>XcodeVersionProperties</code> provider,"
              + " use this as the key with which to retrieve it. Example:<br>"
              + "<pre class='language-python'>\n"
              + "dep = ctx.attr.deps[0]\n"
              + "p = dep[apple_common.XcodeVersionProperties]\n"
              + "</pre>",
      structField = true)
  ProviderApi getXcodeVersionPropertiesConstructor();

  @StarlarkMethod(
      name = "XcodeVersionConfig",
      doc = "The constructor/key for the <code>XcodeVersionConfig</code> provider.",
      structField = true)
  ProviderApi getXcodeVersionConfigConstructor();

  @StarlarkMethod(
      // TODO(b/63899207): This currently does not match ObjcProvider.STARLARK_NAME as it requires
      // a migration of existing Starlark rules.
      name = "Objc",
      doc =
          "The constructor/key for the <code>Objc</code> provider.<p>"
              + "If a target propagates the <code>Objc</code> provider, use this as the "
              + "key with which to retrieve it. Example:<br>"
              + "<pre class='language-python'>\n"
              + "dep = ctx.attr.deps[0]\n"
              + "p = dep[apple_common.Objc]\n"
              + "</pre>",
      structField = true)
  ProviderApi getObjcProviderConstructor();

  @StarlarkMethod(
      name = "AppleDynamicFramework",
      doc =
          "The constructor/key for the <code>AppleDynamicFramework</code> provider.<p>"
              + "If a target propagates the <code>AppleDynamicFramework</code> provider, use this "
              + "as the key with which to retrieve it. Example:<br>"
              + "<pre class='language-python'>\n"
              + "dep = ctx.attr.deps[0]\n"
              + "p = dep[apple_common.AppleDynamicFramework]\n"
              + "</pre>",
      structField = true)
  ProviderApi getAppleDynamicFrameworkConstructor();

  @StarlarkMethod(
      name = "AppleDylibBinary",
      doc =
          "The constructor/key for the <code>AppleDylibBinary</code> provider.<p>If a target"
              + " propagates the <code>AppleDylibBinary</code> provider, use this as the key with"
              + " which to retrieve it. Example:<br><pre class='language-python'>\n"
              + "dep = ctx.attr.deps[0]\n"
              + "p = dep[apple_common.AppleDylibBinary]\n"
              + "</pre>",
      structField = true)
  ProviderApi getAppleDylibBinaryConstructor();

  @StarlarkMethod(
      name = "AppleExecutableBinary",
      doc =
          "The constructor/key for the <code>AppleExecutableBinary</code> provider.<p>"
              + "If a target propagates the <code>AppleExecutableBinary</code> provider,"
              + " use this as the key with which to retrieve it. Example:<br>"
              + "<pre class='language-python'>\n"
              + "dep = ctx.attr.deps[0]\n"
              + "p = dep[apple_common.AppleExecutableBinary]\n"
              + "</pre>",
      structField = true)
  ProviderApi getAppleExecutableBinaryConstructor();

  @StarlarkMethod(
      name = "AppleStaticLibrary",
      doc =
          "The constructor/key for the <code>AppleStaticLibrary</code> provider.<p>"
              + "If a target propagates the <code>AppleStaticLibrary</code> provider, use "
              + "this as the key with which to retrieve it. Example:<br>"
              + "<pre class='language-python'>\n"
              + "dep = ctx.attr.deps[0]\n"
              + "p = dep[apple_common.AppleStaticLibrary]\n"
              + "</pre>",
      structField = true)
  AppleStaticLibraryInfoProvider<?, ?> getAppleStaticLibraryProvider();

  @StarlarkMethod(
      name = "AppleDebugOutputs",
      doc =
          "The constructor/key for the <code>AppleDebugOutputs</code> provider.<p>If a target"
              + " propagates the <code>AppleDebugOutputs</code> provider, use this as the key with"
              + " which to retrieve it. Example:<br><pre class='language-python'>\n"
              + "dep = ctx.attr.deps[0]\n"
              + "p = dep[apple_common.AppleDebugOutputs]\n"
              + "</pre>",
      structField = true)
  ProviderApi getAppleDebugOutputsConstructor();

  @StarlarkMethod(
      name = "AppleLoadableBundleBinary",
      doc =
          "The constructor/key for the <code>AppleLoadableBundleBinary</code> provider.<p>"
              + "If a target propagates the <code>AppleLoadableBundleBinary</code> provider, "
              + "use this as the key with which to retrieve it. Example:<br>"
              + "<pre class='language-python'>\n"
              + "dep = ctx.attr.deps[0]\n"
              + "p = dep[apple_common.AppleLoadableBundleBinary]\n"
              + "</pre>",
      structField = true)
  ProviderApi getAppleLoadableBundleBinaryConstructor();

  @StarlarkMethod(
      name = "apple_host_system_env",
      doc =
          "Returns a <a href='dict.html'>dict</a> of environment variables that should be set "
              + "for actions that need to run build tools on an Apple host system, such as the "
              + " version of Xcode that should be used. The keys are variable names and the values "
              + " are their corresponding values.",
      parameters = {
        @Param(
            name = "xcode_config",
            positional = true,
            named = false,
            type = XcodeConfigInfoApi.class,
            doc = "A provider containing information about the xcode configuration."),
      })
  ImmutableMap<String, String> getAppleHostSystemEnv(XcodeConfigInfoApiT xcodeConfig);

  @StarlarkMethod(
      name = "target_apple_env",
      doc =
          "Returns a <code>dict</code> of environment variables that should be set for actions "
              + "that build targets of the given Apple platform type. For example, this dictionary "
              + "contains variables that denote the platform name and SDK version with which to "
              + "build. The keys are variable names and the values are their corresponding values.",
      parameters = {
        @Param(
            name = "xcode_config",
            positional = true,
            named = false,
            type = XcodeConfigInfoApi.class,
            doc = "A provider containing information about the xcode configuration."),
        @Param(
            name = "platform",
            positional = true,
            named = false,
            type = ApplePlatformApi.class,
            doc = "The apple platform."),
      })
  ImmutableMap<String, String> getTargetAppleEnvironment(
      XcodeConfigInfoApiT xcodeConfig, ApplePlatformApiT platform);

  @StarlarkMethod(
      name = "multi_arch_split",
      doc =
          "A configuration transition for rule attributes to build dependencies in one or more"
              + " Apple platforms. <p>Use of this transition requires that the 'platform_type' and"
              + " 'minimum_os_version' string attributes are defined and mandatory on the"
              + " rule.</p><p>The value of the platform_type attribute will dictate the target"
              + " architectures  for which dependencies along this configuration transition will"
              + " be built.</p><p>Options are:</p><ul><li><code>ios</code>: architectures gathered"
              + " from <code>--ios_multi_cpus</code>.</li><li><code>macos</code>: architectures"
              + " gathered from <code>--macos_cpus</code>.</li><li><code>tvos</code>:"
              + " architectures gathered from"
              + " <code>--tvos_cpus</code>.</li><li><code>watchos</code>: architectures gathered"
              + " from <code>--watchos_cpus</code>.</li></ul><p>minimum_os_version should be a"
              + " dotted version string such as '7.3', and is used to set the minimum operating"
              + " system on the configuration similarly based on platform type. For example,"
              + " specifying platform_type 'ios' and minimum_os_version '8.0' will ensure that"
              + " dependencies are built with minimum iOS version '8.0'.",
      structField = true)
  SplitTransitionProviderApi getMultiArchSplitProvider();

  @StarlarkMethod(
      name = "new_objc_provider",
      doc = "Creates a new ObjcProvider instance.",
      parameters = {
        @Param(
            name = "uses_swift",
            type = Boolean.class,
            defaultValue = "False",
            named = true,
            positional = false,
            doc = "Whether this provider should enable Swift support.")
      },
      extraKeywords =
          @Param(
              name = "kwargs",
              type = Dict.class,
              defaultValue = "{}",
              doc = "Dictionary of arguments."),
      useStarlarkThread = true)
  // This method is registered statically for Starlark, and never called directly.
  ObjcProviderApi<?> newObjcProvider(
      Boolean usesSwift, Dict<String, Object> kwargs, StarlarkThread thread) throws EvalException;

  @StarlarkMethod(
      name = "new_dynamic_framework_provider",
      doc = "Creates a new AppleDynamicFramework provider instance.",
      parameters = {
        @Param(
            name = "binary",
            type = FileApi.class,
            named = true,
            noneable = true,
            positional = false,
            defaultValue = "None",
            doc = "The dylib binary artifact of the dynamic framework."),
        @Param(
            name = "objc",
            type = ObjcProviderApi.class,
            named = true,
            positional = false,
            doc =
                "An ObjcProvider which contains information about the transitive "
                    + "dependencies linked into the binary."),
        @Param(
            name = "framework_dirs",
            type = Depset.class,
            generic1 = String.class,
            named = true,
            noneable = true,
            positional = false,
            defaultValue = "None",
            doc =
                "The framework path names used as link inputs in order to link against the dynamic "
                    + "framework."),
        @Param(
            name = "framework_files",
            type = Depset.class,
            generic1 = FileApi.class,
            named = true,
            noneable = true,
            positional = false,
            defaultValue = "None",
            doc =
                "The full set of artifacts that should be included as inputs to link against the "
                    + "dynamic framework")
      })
  AppleDynamicFrameworkInfoApi<?> newDynamicFrameworkProvider(
      Object dylibBinary,
      ObjcProviderApiT depsObjcProvider,
      Object dynamicFrameworkDirs,
      Object dynamicFrameworkFiles)
      throws EvalException;

  @StarlarkMethod(
      name = "link_multi_arch_binary",
      doc =
          "Links a (potentially multi-architecture) binary targeting Apple platforms. This "
              + "method comprises a bulk of the logic of the <code>apple_binary</code> rule, and "
              + "is exposed as an API to iterate on migration of <code>apple_binary</code> to "
              + "Starlark.\n"
              + "<p>This API is <b>highly experimental</b> and subject to change at any time. Do "
              + "not depend on the stability of this function at this time.",
      parameters = {
        @Param(
            name = "ctx",
            type = StarlarkRuleContextApi.class,
            named = true,
            positional = false,
            doc = "The Starlark rule context."),
        @Param(
            name = "extra_linkopts",
            type = Sequence.class,
            generic1 = String.class,
            named = true,
            positional = false,
            defaultValue = "[]",
            doc = "Extra linkopts to be passed to the linker action."),
        @Param(
            name = "extra_link_inputs",
            type = Sequence.class,
            generic1 = FileApi.class,
            named = true,
            positional = false,
            defaultValue = "[]",
            doc = "Extra files to pass to the linker action."),
      },
      useStarlarkThread = true)
  // TODO(b/70937317): Iterate on, improve, and solidify this API.
  StructApi linkMultiArchBinary(
      StarlarkRuleContextT starlarkRuleContext,
      Sequence<?> extraLinkopts, // <String> expected.
      Sequence<?> extraLinkInputs, // <? extends FileApi> expected.
      StarlarkThread thread)
      throws EvalException, InterruptedException;

  @StarlarkMethod(
      name = "dotted_version",
      doc = "Creates a new <a href=\"DottedVersion.html\">DottedVersion</a> instance.",
      parameters = {
        @Param(
            name = "version",
            type = String.class,
            doc = "The string representation of the DottedVersion.")
      })
  DottedVersionApi<?> dottedVersion(String version) throws EvalException;

  @StarlarkMethod(
      name = "objc_proto_aspect",
      doc =
          "objc_proto_aspect gathers the proto dependencies of the attached rule target, and"
              + " propagates the proto values of its dependencies through the ObjcProto provider.",
      structField = true)
  StarlarkAspectApi getObjcProtoAspect();
}
