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

package com.google.devtools.build.lib.starlarkbuildapi.objc;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import com.google.devtools.build.lib.starlarkbuildapi.StarlarkRuleContextApi;
import com.google.devtools.build.lib.starlarkbuildapi.apple.ApplePlatformApi;
import com.google.devtools.build.lib.starlarkbuildapi.apple.AppleToolchainApi;
import com.google.devtools.build.lib.starlarkbuildapi.apple.DottedVersionApi;
import com.google.devtools.build.lib.starlarkbuildapi.apple.XcodeConfigInfoApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.StructApi;
import com.google.devtools.build.lib.starlarkbuildapi.cpp.CcInfoApi;
import com.google.devtools.build.lib.starlarkbuildapi.platform.ConstraintValueInfoApi;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.NoneType;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.StarlarkInt;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.StarlarkValue;

/** Interface for a module with useful functions for creating apple-related rule implementations. */
@StarlarkBuiltin(
    name = "apple_common",
    category = DocCategory.TOP_LEVEL_MODULE,
    doc = "Functions for Starlark to access internals of the apple rule implementations.")
public interface AppleCommonApi<
        FileApiT extends FileApi,
        ConstraintValueT extends ConstraintValueInfoApi,
        StarlarkRuleContextT extends StarlarkRuleContextApi<ConstraintValueT>,
        CcInfoApiT extends CcInfoApi<?>,
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
              + "types:<br><ul>" //
              + "<li><code>ios</code></li>" //
              + "<li><code>macos</code></li>" //
              + "<li><code>tvos</code></li>" //
              + "<li><code>visionos</code></li>" //
              + "<li><code>watchos</code></li>" //
              + "</ul><p>" //
              + "These values can be passed to methods that expect a platform type, like the"
              + " 'apple' configuration fragment's <a"
              + " href='../fragments/apple.html#multi_arch_platform'>multi_arch_platform</a>"
              + " method.<p>Example:<p><pre class='language-python'>\n"
              + "ctx.fragments.apple.multi_arch_platform(apple_common.platform_type.ios)\n"
              + "</pre>",
      structField = true)
  StructApi getPlatformTypeStruct();

  @StarlarkMethod(
      name = "platform",
      doc =
          "An enum-like struct that contains the following fields corresponding to Apple "
              + "platforms:<br><ul>" //
              + "<li><code>ios_device</code></li>" //
              + "<li><code>ios_simulator</code></li>" //
              + "<li><code>macos</code></li>" //
              + "<li><code>tvos_device</code></li>" //
              + "<li><code>tvos_simulator</code></li>" //
              + "<li><code>visionos_device</code></li>" //
              + "<li><code>visionos_simulator</code></li>" //
              + "<li><code>watchos_device</code></li>" //
              + "<li><code>watchos_simulator</code></li>" //
              + "</ul><p>" //
              + "These values can be passed to methods that expect a platform, like "
              + "<a href='../providers/XcodeVersionConfig.html#sdk_version_for_platform'>"
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
      name = "apple_host_system_env",
      doc =
          "Returns a <a href='../core/dict.html'>dict</a> of environment variables that should be"
              + " set for actions that need to run build tools on an Apple host system, such as the"
              + "  version of Xcode that should be used. The keys are variable names and the values"
              + "  are their corresponding values.",
      parameters = {
        @Param(
            name = "xcode_config",
            positional = true,
            named = false,
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
            doc = "A provider containing information about the xcode configuration."),
        @Param(
            name = "platform",
            positional = true,
            named = false,
            doc = "The apple platform."),
      })
  ImmutableMap<String, String> getTargetAppleEnvironment(
      XcodeConfigInfoApiT xcodeConfig, ApplePlatformApiT platform);

  @StarlarkMethod(
      name = "new_objc_provider",
      doc = "Creates a new ObjcProvider instance.",
      parameters = {},
      extraKeywords =
          @Param(name = "kwargs", defaultValue = "{}", doc = "Dictionary of arguments."),
      useStarlarkThread = true)
  // This method is registered statically for Starlark, and never called directly.
  ObjcProviderApi<?> newObjcProvider(Dict<String, Object> kwargs, StarlarkThread thread)
      throws EvalException;

  @StarlarkMethod(
      name = "new_dynamic_framework_provider",
      doc = "Creates a new AppleDynamicFramework provider instance.",
      parameters = {
        @Param(
            name = "binary",
            allowedTypes = {
              @ParamType(type = FileApi.class),
              @ParamType(type = NoneType.class),
            },
            named = true,
            positional = false,
            defaultValue = "None",
            doc = "The dylib binary artifact of the dynamic framework."),
        @Param(
            name = "cc_info",
            named = true,
            positional = false,
            defaultValue = "None",
            doc =
                "A CcInfo which contains information about the transitive dependencies "
                    + "linked into the binary."),
        @Param(
            name = "objc",
            named = true,
            positional = false,
            defaultValue = "None",
            doc =
                "An ObjcProvider which contains information about the transitive "
                    + "dependencies linked into the binary."),
        @Param(
            name = "framework_dirs",
            allowedTypes = {
              @ParamType(type = Depset.class, generic1 = String.class),
              @ParamType(type = NoneType.class),
            },
            named = true,
            positional = false,
            defaultValue = "None",
            doc =
                "The framework path names used as link inputs in order to link against the dynamic "
                    + "framework."),
        @Param(
            name = "framework_files",
            allowedTypes = {
              @ParamType(type = Depset.class, generic1 = FileApi.class),
              @ParamType(type = NoneType.class),
            },
            named = true,
            positional = false,
            defaultValue = "None",
            doc =
                "The full set of artifacts that should be included as inputs to link against the "
                    + "dynamic framework")
      },
      useStarlarkThread = true)
  AppleDynamicFrameworkInfoApi<?> newDynamicFrameworkProvider(
      Object dylibBinary,
      Object depsCcInfo,
      Object depsObjcProvider,
      Object dynamicFrameworkDirs,
      Object dynamicFrameworkFiles,
      StarlarkThread thread)
      throws EvalException;

  @StarlarkMethod(
      name = "new_executable_binary_provider",
      doc = "Creates a new AppleExecutableBinaryInfo provider instance.",
      parameters = {
        @Param(
            name = "binary",
            allowedTypes = {
              @ParamType(type = FileApi.class),
              @ParamType(type = NoneType.class),
            },
            named = true,
            positional = false,
            defaultValue = "None",
            doc = "The binary artifact of the executable."),
        @Param(
            name = "cc_info",
            named = true,
            positional = false,
            defaultValue = "None",
            doc =
                "A CcInfo which contains information about the transitive dependencies "
                    + "linked into the binary."),
        @Param(
            name = "objc",
            named = true,
            positional = false,
            defaultValue = "None",
            doc =
                "An ObjcProvider which contains information about the transitive "
                    + "dependencies linked into the binary.")
      },
      useStarlarkThread = true)
  AppleExecutableBinaryApi newExecutableBinaryProvider(
      Object executableBinary, Object depsCcInfo, Object depsObjcProvider, StarlarkThread thread)
      throws EvalException;

  @StarlarkMethod(
      name = "link_multi_arch_binary",
      doc =
          "Links a (potentially multi-architecture) binary targeting Apple platforms. This "
              + "method comprises a bulk of the logic of the Starlark <code>apple_binary</code> "
              + "rule in the rules_apple domain and exists to aid in the migration of its "
              + "linking logic to Starlark in rules_apple.\n"
              + "<p>This API is <b>highly experimental</b> and subject to change at any time. Do "
              + "not depend on the stability of this function at this time.",
      parameters = {
        @Param(name = "ctx", named = true, positional = false, doc = "The Starlark rule context."),
        @Param(
            name = "avoid_deps",
            allowedTypes = {
              @ParamType(type = Sequence.class, generic1 = TransitiveInfoCollection.class),
              @ParamType(type = NoneType.class),
            },
            named = true,
            positional = false,
            defaultValue = "None",
            doc =
                "A list of <code>Target</code>s that are in the dependency graph of the binary but"
                    + " whose libraries should not be linked into the binary. This is the case for"
                    + " dependencies that will be found at runtime in another image, such as the"
                    + " bundle loader or any dynamic libraries/frameworks that will be loaded by"
                    + " this binary."),
        @Param(
            name = "extra_linkopts",
            allowedTypes = {@ParamType(type = Sequence.class, generic1 = String.class)},
            named = true,
            positional = false,
            defaultValue = "[]",
            doc = "Extra linkopts to be passed to the linker action."),
        @Param(
            name = "extra_link_inputs",
            allowedTypes = {@ParamType(type = Sequence.class, generic1 = FileApi.class)},
            named = true,
            positional = false,
            defaultValue = "[]",
            doc = "Extra files to pass to the linker action."),
        @Param(
            name = "extra_requested_features",
            allowedTypes = {@ParamType(type = Sequence.class, generic1 = String.class)},
            named = true,
            positional = false,
            defaultValue = "[]",
            doc = "Extra requested features to be passed to the linker action."),
        @Param(
            name = "extra_disabled_features",
            allowedTypes = {@ParamType(type = Sequence.class, generic1 = String.class)},
            named = true,
            positional = false,
            defaultValue = "[]",
            doc = "Extra disabled features to be passed to the linker action."),
        @Param(
            name = "stamp",
            named = true,
            positional = false,
            defaultValue = "-1",
            doc =
                "Whether to include build information in the linked binary. If 1, build "
                    + "information is always included. If 0, build information is always excluded. "
                    + "If -1 (the default), then the behavior is determined by the --[no]stamp "
                    + "flag. This should be set to 0 when generating the executable output for "
                    + "test rules."),
        @Param(
            name = "variables_extension",
            positional = false,
            named = true,
            documented = false,
            allowedTypes = {
              @ParamType(type = Dict.class),
              @ParamType(type = NoneType.class),
            },
            defaultValue = "None"),
      },
      useStarlarkThread = true)
  // TODO(b/70937317): Iterate on, improve, and solidify this API.
  StructApi linkMultiArchBinary(
      StarlarkRuleContextT starlarkRuleContext,
      Object avoidDeps, // Sequence<TransitiveInfoCollection> expected.
      Sequence<?> extraLinkopts, // <String> expected.
      Sequence<?> extraLinkInputs, // <? extends FileApi> expected.
      Sequence<?> extraRequestedFeatures, // <String> expected.
      Sequence<?> extraDisabledFeatures, // <String> expected.
      StarlarkInt stamp,
      Object variablesExtension,
      StarlarkThread thread)
      throws EvalException, InterruptedException;

  @StarlarkMethod(
      name = "link_multi_arch_static_library",
      doc =
          "Links a (potentially multi-architecture) static library targeting Apple platforms."
              + " This method comprises a part of the Starlark <code>apple_static_library</code>"
              + " rule logic, in the rules_apple domain and exists to aid in the migration of its"
              + " linking logic to Starlark in rules_apple.\n"
              + "<p>This API is <b>highly experimental</b> and subject to change at any time."
              + " Do not depend on the stability of this function at this time.",
      parameters = {
        @Param(name = "ctx", named = true, positional = false, doc = "The Starlark rule context."),
      },
      useStarlarkThread = true)
  StructApi linkMultiArchStaticLibrary(
      StarlarkRuleContextT starlarkRuleContext, StarlarkThread thread)
      throws EvalException, InterruptedException;

  @StarlarkMethod(
      name = "dotted_version",
      doc = "Creates a new <a href=\"../builtins/DottedVersion.html\">DottedVersion</a> instance.",
      parameters = {
        @Param(name = "version", doc = "The string representation of the DottedVersion.")
      })
  DottedVersionApi<?> dottedVersion(String version) throws EvalException;
}
