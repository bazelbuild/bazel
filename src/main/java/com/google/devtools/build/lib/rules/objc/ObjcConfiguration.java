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
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.CompilationMode;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.rules.apple.ApplePlatform.PlatformType;
import com.google.devtools.build.lib.rules.apple.DottedVersion;
import com.google.devtools.build.lib.rules.cpp.HeaderDiscovery;
import com.google.devtools.build.lib.skylarkbuildapi.apple.ObjcConfigurationApi;
import javax.annotation.Nullable;

/** A compiler configuration containing flags required for Objective-C compilation. */
@Immutable
public class ObjcConfiguration extends BuildConfiguration.Fragment
    implements ObjcConfigurationApi<PlatformType> {
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
  private final DottedVersion watchosSimulatorVersion;
  private final String watchosSimulatorDevice;
  private final DottedVersion tvosSimulatorVersion;
  private final String tvosSimulatorDevice;
  private final boolean generateDsym;
  private final boolean generateLinkmap;
  private final boolean runMemleaks;
  private final ImmutableList<String> copts;
  private final CompilationMode compilationMode;
  private final ImmutableList<String> fastbuildOptions;
  private final boolean enableBinaryStripping;
  private final boolean moduleMapsEnabled;
  @Nullable private final String signingCertName;
  private final boolean debugWithGlibcxx;
  @Nullable private final Label extraEntitlements;
  private final boolean deviceDebugEntitlements;
  private final boolean enableAppleBinaryNativeProtos;
  private final HeaderDiscovery.DotdPruningMode dotdPruningPlan;
  private final boolean experimentalHeaderThinning;
  private final int objcHeaderThinningPartitionSize;
  private final Label objcHeaderScannerTool;
  private final Label appleSdk;
  private final boolean strictObjcModuleMaps;
  private final boolean disableObjcLibraryResources;

  ObjcConfiguration(ObjcCommandLineOptions objcOptions, BuildConfiguration.Options options) {
    this.iosSimulatorDevice =
        Preconditions.checkNotNull(objcOptions.iosSimulatorDevice, "iosSimulatorDevice");
    this.iosSimulatorVersion =
        Preconditions.checkNotNull(DottedVersion.maybeUnwrap(objcOptions.iosSimulatorVersion),
            "iosSimulatorVersion");
    this.watchosSimulatorDevice =
        Preconditions.checkNotNull(objcOptions.watchosSimulatorDevice, "watchosSimulatorDevice");
    this.watchosSimulatorVersion =
        Preconditions.checkNotNull(DottedVersion.maybeUnwrap(objcOptions.watchosSimulatorVersion),
            "watchosSimulatorVersion");
    this.tvosSimulatorDevice =
        Preconditions.checkNotNull(objcOptions.tvosSimulatorDevice, "tvosSimulatorDevice");
    this.tvosSimulatorVersion =
        Preconditions.checkNotNull(DottedVersion.maybeUnwrap(objcOptions.tvosSimulatorVersion),
            "tvosSimulatorVersion");
    this.generateLinkmap = objcOptions.generateLinkmap;
    this.runMemleaks = objcOptions.runMemleaks;
    this.copts = ImmutableList.copyOf(objcOptions.copts);
    this.compilationMode = Preconditions.checkNotNull(options.compilationMode, "compilationMode");
    this.generateDsym =
        objcOptions.appleGenerateDsym
            || (objcOptions.appleEnableAutoDsymDbg && this.compilationMode == CompilationMode.DBG);
    this.fastbuildOptions = ImmutableList.copyOf(objcOptions.fastbuildOptions);
    this.enableBinaryStripping = objcOptions.enableBinaryStripping;
    this.moduleMapsEnabled = objcOptions.enableModuleMaps;
    this.signingCertName = objcOptions.iosSigningCertName;
    this.debugWithGlibcxx = objcOptions.debugWithGlibcxx;
    this.extraEntitlements = objcOptions.extraEntitlements;
    this.deviceDebugEntitlements = objcOptions.deviceDebugEntitlements;
    this.enableAppleBinaryNativeProtos = objcOptions.enableAppleBinaryNativeProtos;
    this.dotdPruningPlan =
        objcOptions.useDotdPruning
            ? HeaderDiscovery.DotdPruningMode.USE
            : HeaderDiscovery.DotdPruningMode.DO_NOT_USE;
    this.experimentalHeaderThinning = objcOptions.experimentalObjcHeaderThinning;
    this.objcHeaderThinningPartitionSize = objcOptions.objcHeaderThinningPartitionSize;
    this.objcHeaderScannerTool = objcOptions.objcHeaderScannerTool;
    this.appleSdk = objcOptions.appleSdk;
    this.strictObjcModuleMaps = objcOptions.strictObjcModuleMaps;
    this.disableObjcLibraryResources = objcOptions.disableObjcLibraryResources;
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

  @Override
  public String getSimulatorDeviceForPlatformType(PlatformType platformType) {
    switch (platformType) {
      case IOS:
        return iosSimulatorDevice;
      case TVOS:
        return tvosSimulatorDevice;
      case WATCHOS:
        return watchosSimulatorDevice;
      default:
        throw new IllegalArgumentException(
            "ApplePlatform type " + platformType + " does not support " + "simulators.");
    }
  }

  @Override
  public DottedVersion getSimulatorVersionForPlatformType(PlatformType platformType) {
    switch (platformType) {
      case IOS:
        return iosSimulatorVersion;
      case TVOS:
        return tvosSimulatorVersion;
      case WATCHOS:
        return watchosSimulatorVersion;
      default:
        throw new IllegalArgumentException(
            "ApplePlatform type " + platformType + " does not support " + "simulators.");
    }
  }

  /**
   * Returns whether dSYM generation is enabled.
   */
  @Override
  public boolean generateDsym() {
    return generateDsym;
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
        if (this.debugWithGlibcxx) {
          return ImmutableList.<String>builder()
              .addAll(DBG_COPTS)
              .addAll(GLIBCXX_DBG_COPTS)
              .build();
        } else {
          return DBG_COPTS;
        }
      case FASTBUILD:
        return fastbuildOptions;
      case OPT:
        return OPT_COPTS;
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
   * Whether module map generation and interpretation is enabled.
   */
  public boolean moduleMapsEnabled() {
    return moduleMapsEnabled;
  }

  /**
   * Returns whether to perform symbol and dead-code strippings on linked binaries. The strippings
   * are performed iff --compilation_mode=opt and --objc_enable_binary_stripping are specified.
   */
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
   * Returns the extra entitlements plist specified as a flag or {@code null} if none was given.
   */
  @Nullable
  public Label getExtraEntitlements() {
    return extraEntitlements;
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

  /** Returns true if apple_binary targets should generate and link Objc protos. */
  @Override
  public boolean enableAppleBinaryNativeProtos() {
    return enableAppleBinaryNativeProtos;
  }

  /** Returns the DotdPruningPlan for compiles in this build. */
  public HeaderDiscovery.DotdPruningMode getDotdPruningPlan() {
    return dotdPruningPlan;
  }

  /** Returns true if header thinning of ObjcCompile actions is enabled to reduce action inputs. */
  public boolean useExperimentalHeaderThinning() {
    return experimentalHeaderThinning;
  }

  /** Returns the max number of source files to add to each header scanning action. */
  public int objcHeaderThinningPartitionSize() {
    return objcHeaderThinningPartitionSize;
  }

  /** Returns the label for the ObjC header scanner tool. */
  public Label getObjcHeaderScannerTool() {
    return objcHeaderScannerTool;
  }

  /** Returns the label for the Apple SDK for current build configuration. */
  public Label getAppleSdk() {
    return appleSdk;
  }

  /** Returns true if Objective-C module maps should only be propagated to direct dependencies. */
  public boolean useStrictObjcModuleMaps() {
    return strictObjcModuleMaps;
  }

  /** Returns true if Objective-C library resource attributes are disallowed. */
  public boolean disableObjcLibraryResources() {
    return disableObjcLibraryResources;
  }
}
