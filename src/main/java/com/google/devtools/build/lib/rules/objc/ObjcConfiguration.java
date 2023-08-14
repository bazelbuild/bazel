// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.objc;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.CompilationMode;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.config.Fragment;
import com.google.devtools.build.lib.analysis.config.RequiresOptions;
import com.google.devtools.build.lib.analysis.starlark.StarlarkRuleContext;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.BuiltinRestriction;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.rules.apple.ApplePlatform.PlatformType;
import com.google.devtools.build.lib.rules.apple.DottedVersion;
import com.google.devtools.build.lib.rules.cpp.CppOptions;
import com.google.devtools.build.lib.starlarkbuildapi.apple.ObjcConfigurationApi;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.StarlarkThread;

/** A compiler configuration containing flags required for Objective-C compilation. */
@Immutable
@RequiresOptions(options = {CppOptions.class, ObjcCommandLineOptions.class})
public class ObjcConfiguration extends Fragment implements ObjcConfigurationApi<PlatformType> {
  @VisibleForTesting
  static final ImmutableList<String> DBG_COPTS =
      ImmutableList.of("-O0", "-DDEBUG=1", "-fstack-protector", "-fstack-protector-all", "-g");

  @VisibleForTesting
  static final ImmutableList<String> GLIBCXX_DBG_COPTS =
      ImmutableList.of(
          "-D_GLIBCXX_DEBUG", "-D_GLIBCXX_DEBUG_PEDANTIC", "-D_GLIBCPP_CONCEPT_CHECKS");

  @VisibleForTesting
  static final ImmutableList<String> OPT_COPTS =
      ImmutableList.of(
          "-Os", "-DNDEBUG=1", "-Wno-unused-variable", "-Winit-self", "-Wno-extra");

  private final DottedVersion iosSimulatorVersion;
  private final String iosSimulatorDevice;
  // TODO(b/236152224): Delete after Starlark uses are migrated to CppConfiguration.
  private final boolean generateLinkmap;
  private final boolean runMemleaks;
  // TODO(b/236152224): Delete after Starlark uses are migrated to CppConfiguration.
  private final ImmutableList<String> copts;
  private final CompilationMode compilationMode;
  private final ImmutableList<String> fastbuildOptions;
  // TODO(b/236152224): Delete after Starlark uses are migrated to CppConfiguration.
  private final boolean enableBinaryStripping;
  @Nullable private final String signingCertName;
  private final boolean debugWithGlibcxx;
  private final boolean deviceDebugEntitlements;
  private final boolean avoidHardcodedCompilationFlags;
  private final boolean disallowSdkFrameworksAttributes;
  private final boolean alwayslinkByDefault;

  public ObjcConfiguration(BuildOptions buildOptions) {
    CoreOptions options = buildOptions.get(CoreOptions.class);
    ObjcCommandLineOptions objcOptions = buildOptions.get(ObjcCommandLineOptions.class);
    CppOptions cppOptions = buildOptions.get(CppOptions.class);

    this.iosSimulatorDevice = objcOptions.iosSimulatorDevice;
    this.iosSimulatorVersion = DottedVersion.maybeUnwrap(objcOptions.iosSimulatorVersion);
    this.generateLinkmap = cppOptions.objcGenerateLinkmap;
    this.runMemleaks = objcOptions.runMemleaks;
    this.copts = ImmutableList.copyOf(cppOptions.objcoptList);
    this.compilationMode = Preconditions.checkNotNull(options.compilationMode, "compilationMode");
    this.fastbuildOptions = ImmutableList.copyOf(objcOptions.fastbuildOptions);
    this.enableBinaryStripping = cppOptions.objcEnableBinaryStripping;
    this.signingCertName = objcOptions.iosSigningCertName;
    this.debugWithGlibcxx = objcOptions.debugWithGlibcxx;
    this.deviceDebugEntitlements = objcOptions.deviceDebugEntitlements;
    this.avoidHardcodedCompilationFlags =
        objcOptions.incompatibleAvoidHardcodedObjcCompilationFlags;
    this.disallowSdkFrameworksAttributes = objcOptions.incompatibleDisallowSdkFrameworksAttributes;
    this.alwayslinkByDefault = objcOptions.incompatibleObjcAlwayslinkByDefault;
  }

  /**
   * Returns the type of device (e.g. 'iPhone 6') to simulate when running on the simulator.
   */
  @Override
  public String getIosSimulatorDevice() {
    // TODO(bazel-team): Deprecate in favor of getSimulatorDeviceForPlatformType(IOS).
    return iosSimulatorDevice;
  }

  @Override
  public DottedVersion getIosSimulatorVersion() {
    // TODO(bazel-team): Deprecate in favor of getSimulatorVersionForPlatformType(IOS).
    return iosSimulatorVersion;
  }

  /**
   * Returns whether linkmap generation is enabled.
   */
  @Override
  public boolean generateLinkmap() {
    return generateLinkmap;
  }

  @Override
  public boolean runMemleaks() {
    return runMemleaks;
  }

  /**
   * Returns the current compilation mode.
   */
  public CompilationMode getCompilationMode() {
    return compilationMode;
  }

  /**
   * Returns the default set of clang options for the current compilation mode.
   */
  @Override
  public ImmutableList<String> getCoptsForCompilationMode() {
    switch (compilationMode) {
      case DBG:
        ImmutableList.Builder<String> opts = ImmutableList.builder();
        if (!this.avoidHardcodedCompilationFlags) {
          opts.addAll(DBG_COPTS);
        }
        if (this.debugWithGlibcxx) {
          opts.addAll(GLIBCXX_DBG_COPTS);
        }
        return opts.build();
      case FASTBUILD:
        return fastbuildOptions;
      case OPT:
        return this.avoidHardcodedCompilationFlags ? ImmutableList.of() : OPT_COPTS;
      default:
        throw new AssertionError();
    }
  }

  /**
   * Returns options passed to (Apple) clang when compiling Objective C. These options should be
   * applied after any default options but before options specified in the attributes of the rule.
   */
  @Override
  public ImmutableList<String> getCopts() {
    return copts;
  }

  /**
   * Returns whether to perform symbol and dead-code strippings on linked binaries. The strippings
   * are performed iff --compilation_mode=opt and --objc_enable_binary_stripping are specified.
   */
  @Override
  public boolean shouldStripBinary() {
    return this.enableBinaryStripping && getCompilationMode() == CompilationMode.OPT;
  }

  /**
   * Returns the flag-supplied certificate name to be used in signing or {@code null} if no such
   * certificate was specified.
   */
  @Override
  public String getSigningCertName() {
    return this.signingCertName;
  }

  /**
   * Returns whether device debug entitlements should be included when signing an application.
   *
   * <p>Note that debug entitlements will be included only if the --device_debug_entitlements flag
   * is set <b>and</b> the compilation mode is not {@code opt}.
   */
  @Override
  public boolean useDeviceDebugEntitlements() {
    return deviceDebugEntitlements && compilationMode != CompilationMode.OPT;
  }

  /** Returns whether sdk_frameworks and weak_sdk_frameworks attributes are disallowed. */
  @Override
  public boolean disallowSdkFrameworksAttributes() {
    return disallowSdkFrameworksAttributes;
  }

  /** Returns whether objc_library and objc_import should default to alwayslink=True. */
  @Override
  public boolean alwayslinkByDefault() {
    return alwayslinkByDefault;
  }

  /**
   * Looks at any explicit value for alwayslink on ctx and then falls back to the value of
   * alwayslink_by_default.
   */
  @Override
  public boolean targetShouldAlwayslink(StarlarkRuleContext ruleContext, StarlarkThread thread)
      throws EvalException {
    BuiltinRestriction.failIfCalledOutsideBuiltins(thread);

    AttributeMap attributes = ruleContext.getRuleContext().attributes();
    if (attributes.isAttributeValueExplicitlySpecified("alwayslink")) {
      return attributes.get("alwayslink", Type.BOOLEAN);
    }

    return alwayslinkByDefault;
  }
}
