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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.LabelConverter;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Attribute.SplitTransition;
import com.google.devtools.build.lib.rules.apple.DottedVersion;
import com.google.devtools.build.lib.rules.apple.DottedVersionConverter;
import com.google.devtools.common.options.Converters.CommaSeparatedOptionListConverter;
import com.google.devtools.common.options.EnumConverter;
import com.google.devtools.common.options.Option;
import java.util.List;

/**
 * Command-line options for building Objective-C targets.
 */
public class ObjcCommandLineOptions extends FragmentOptions {
  @Option(
    name = "ios_simulator_version",
    defaultValue = "9.3",
    category = "run",
    converter = DottedVersionConverter.class,
    help =
        "The version of iOS to run on the simulator when running or testing. This is ignored "
            + "for ios_test rules if a target device is specified in the rule."
  )
  public DottedVersion iosSimulatorVersion;

  @Option(name = "ios_simulator_device",
      defaultValue = "iPhone 5s",
      category = "run",
      help = "The device to simulate when running an iOS application in the simulator, e.g. "
          + "'iPhone 6'. You can get a list of devices by running 'xcrun simctl list devicetypes' "
          + "on the machine the simulator will be run on.")
  public String iosSimulatorDevice;

  @Option(
      name = "watchos_simulator_version",
      defaultValue = "2.0",
      category = "run",
      converter = DottedVersionConverter.class,
      help = "The version of watchOS to run on the simulator when running or testing."
  )
  public DottedVersion watchosSimulatorVersion;

  @Option(name = "watchos_simulator_device",
      defaultValue = "Apple Watch - 38mm",
      category = "run",
      help = "The device to simulate when running an watchOS application in the simulator, e.g. "
          + "'Apple Watch - 38mm'. You can get a list of devices by running 'xcrun simctl list "
          + "devicetypes' on the machine the simulator will be run on.")
  public String watchosSimulatorDevice;

  @Option(
      name = "tvos_simulator_version",
      defaultValue = "9.0",
      category = "run",
      converter = DottedVersionConverter.class,
      help = "The version of tvOS to run on the simulator when running or testing."
  )
  public DottedVersion tvosSimulatorVersion;

  @Option(name = "tvos_simulator_device",
      defaultValue = "Apple TV 1080p",
      category = "run",
      help = "The device to simulate when running an tvOS application in the simulator, e.g. "
          + "'Apple TV 1080p'. You can get a list of devices by running 'xcrun simctl list "
          + "devicetypes' on the machine the simulator will be run on.")
  public String tvosSimulatorDevice;

  @Option(name = "objc_generate_linkmap",
      defaultValue = "false",
      category = "flags",
      help = "Specifies whether to generate a linkmap file.")
  public boolean generateLinkmap;

  @Option(name = "objccopt",
      allowMultiple = true,
      defaultValue = "",
      category = "flags",
      help = "Additional options to pass to Objective C compilation.")
  public List<String> copts;

  @Option(name = "ios_memleaks",
      defaultValue =  "false",
      category = "misc",
      help = "Enable checking for memory leaks in ios_test targets.")
  public boolean runMemleaks;

  @Option(name = "experimental_enable_objc_cc_deps",
      defaultValue = "false",
      category = "undocumented",
      help = "Allows objc_* rules to depend on cc_library and causes any objc dependencies to be "
          + "built with --cpu set to \"ios_<--ios_cpu>\" for any values in --ios_multi_cpu.")
  public boolean enableCcDeps;

  @Option(name = "experimental_objc_fastbuild_options",
      defaultValue = "-O0,-DDEBUG=1",
      category = "undocumented",
      converter = CommaSeparatedOptionListConverter.class,
      help = "Uses these strings as objc fastbuild compiler options.")
  public List<String> fastbuildOptions;

  @Option(
    name = "experimental_objc_enable_module_maps",
    defaultValue = "false",
    category = "undocumented",
    help = "Enables module map generation and interpretation."
  )
  public boolean enableModuleMaps;

  @Option(
    name = "objc_enable_binary_stripping",
    defaultValue = "false",
    category = "flags",
    help =
        "Whether to perform symbol and dead-code strippings on linked binaries. Binary "
            + "strippings will be performed if both this flag and --compilationMode=opt are "
            + "specified."
  )
  public boolean enableBinaryStripping;

  @Option(
    name = "apple_generate_dsym",
    defaultValue = "false",
    category = "flags",
    help = "Whether to generate debug symbol(.dSYM) file(s)."
  )
  public boolean appleGenerateDsym;

  @Option(
    name = "ios_signing_cert_name",
    defaultValue = "null",
    category = "flags",
    help =
        "Certificate name to use for iOS signing. If not set will fall back to provisioning "
            + "profile. May be the certificate's keychain identity preference or (substring) of "
            + "the certificate's common name, as per codesign's man page (SIGNING IDENTITIES)."
  )
  public String iosSigningCertName;

  @Option(
    name = "experimental_use_absolute_paths_for_actions",
    defaultValue = "false",
    category = "undocumented",
    help = "If set, then all actions objc actions will be executed with absolute paths."
  )
  public boolean useAbsolutePathsForActions;

  @Option(
    name = "xcode_override_workspace_root",
    defaultValue = "",
    category = "xcode",
    help =
        "If set, then this path will be used as workspace_root and mainGroup path when "
            + "generating an .xcodeproj/project.pbxproj file."
  )
  public String xcodeOverrideWorkspaceRoot;

  @Option(
    name = "objc_includes_prioritize_static_libs",
    defaultValue = "false",
    category = "flags",
    help =
        "If set, the linker invocation will contain static library includes before frameworks"
            + " and system libraries."
  )
  public boolean prioritizeStaticLibs;

  @Option(
    name = "objc_debug_with_GLIBCXX",
    defaultValue = "true",
    category = "undocumented",
    help =
      "If set, and compilation mode is set to 'dbg', define GLIBCXX_DEBUG, "
        + " GLIBCXX_DEBUG_PEDANTIC and GLIBCPP_CONCEPT_CHECKS."
  )
  public boolean debugWithGlibcxx;

  @Option(
    name = "extra_entitlements",
    defaultValue = "null",
    category = "flags",
    converter = LabelConverter.class,
    help =
        "Location of a .entitlements file that is merged into any iOS signing action in this "
            + "build."
  )
  public Label extraEntitlements;

  @Option(
    name = "device_debug_entitlements",
    defaultValue = "true",
    category = "flags",
    help =
        "If set, and compilation mode is not 'opt', objc apps will include debug entitlements "
            + "when signing."
  )
  public boolean deviceDebugEntitlements;

  @Option(
      name = "deprecated_generate_xcode_project",
      defaultValue = "true",
      category = "flags",
      help =
          "If set, will generate xcode project for targets that support this. Will be removed soon."
  )
  public boolean generateXcodeProject;

  /**
   * Specifies the circumstances under which a CROSSTOOL is used for objc in this configuration.
   */
  public enum ObjcCrosstoolMode {
    /** The CROSSTOOL is used for all objc compile, archive, and link actions. */
    ALL,

    /**
     * The CROSSTOOL is used for all objc compile and archive actions originating from an
     * objc_library target.
     */
    LIBRARY,

    /** The CROSSTOOL is not used for any objc action. */
    OFF
  }

  /**
   * Converter for {@link ObjcCrosstoolMode}.
   */
  public static class ObjcCrosstoolUsageConverter extends EnumConverter<ObjcCrosstoolMode> {
    public ObjcCrosstoolUsageConverter() {
      super(ObjcCrosstoolMode.class, "objc crosstool mode");
    }
  }

  @Option(
      name = "experimental_objc_crosstool",
      defaultValue = "off",
      category = "undocumented",
      converter = ObjcCrosstoolUsageConverter.class
  )
  public ObjcCrosstoolMode objcCrosstoolMode;

  // TODO(b/34260565): Remove in favor of --experimental_objc_crosstool
  @Option(
      name = "experimental_objc_library",
      defaultValue = "false",
      category = "undocumented"
  )
  public boolean experimentalObjcLibrary;

  @Option(
    name = "objc_use_dotd_pruning",
    defaultValue = "true",
    category = "flags",
    help =
        "If set, .d files emited by clang will be used to prune the set of inputs passed into objc "
            + "compiles."
  )
  public boolean useDotdPruning;

  @Option(
    name = "enable_apple_binary_native_protos",
    defaultValue = "true",
    category = "flags",
    help =
        "If set, apple_binary will generate and link objc protos into the output binary."
  )
  public boolean enableAppleBinaryNativeProtos;

  @Option(
    name = "experimental_objc_header_thinning",
    defaultValue = "false",
    category = "flags",
    help =
        "If set then ObjcCompile actions will have their action inputs reduced by running a tool "
            + "to detect which headers are actually required for compilation."
  )
  public boolean experimentalObjcHeaderThinning;

  @Option(
    name = "objc_header_scanner_tool",
    defaultValue = "@bazel_tools//tools/objc:header_scanner",
    category = "undocumented",
    converter = LabelConverter.class,
    help =
        "Location of tool to scan Objective-C code for inclusions and output a .headers_list "
            + "file."
  )
  public Label objcHeaderScannerTool;

  @Override
  public FragmentOptions getHost(boolean fallback) {
    ObjcCommandLineOptions host = (ObjcCommandLineOptions) super.getHost(fallback);
    // This should have the same value in both target and host configurations
    host.objcHeaderScannerTool = this.objcHeaderScannerTool;
    return host;
  }

  @SuppressWarnings("unchecked")
  @Override
  public List<SplitTransition<BuildOptions>> getPotentialSplitTransitions() {
    return ImmutableList.<SplitTransition<BuildOptions>>builder().add(
            IosApplication.SPLIT_ARCH_TRANSITION, IosExtension.MINIMUM_OS_AND_SPLIT_ARCH_TRANSITION,
            AppleWatch1Extension.MINIMUM_OS_AND_SPLIT_ARCH_TRANSITION,
            AppleCrosstoolSplitTransition.APPLE_CROSSTOOL_SPLIT_TRANSITION)
        .addAll(MultiArchSplitTransitionProvider.getPotentialSplitTransitions())
        .build();
  }
}
