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
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.packages.Attribute.SplitTransition;
import com.google.devtools.build.lib.rules.apple.DottedVersion;
import com.google.devtools.build.lib.rules.apple.DottedVersionConverter;
import com.google.devtools.build.lib.rules.objc.ReleaseBundlingSupport.SplitArchTransition.ConfigurationDistinguisher;
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
    defaultValue = "8.4",
    category = "run",
    converter = DottedVersionConverter.class,
    deprecationWarning = "Use target_device instead to drive the simulator to use.",
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

  @Option(name = "objc_generate_debug_symbols",
      defaultValue = "false",
      category = "flags",
      help = "Specifies whether to generate debug symbol(.dSYM) file.")
  public boolean generateDebugSymbols;

  @Option(name = "objccopt",
      allowMultiple = true,
      defaultValue = "",
      category = "flags",
      help = "Additional options to pass to Objective C compilation.")
  public List<String> copts;

  @Option(
    name = "ios_minimum_os",
    defaultValue = DEFAULT_MINIMUM_IOS,
    category = "flags",
    converter = DottedVersionConverter.class,
    help = "Minimum compatible iOS version for target simulators and devices."
  )
  public DottedVersion iosMinimumOs;

  @Option(name = "ios_memleaks",
      defaultValue =  "false",
      category = "misc",
      help = "Enable checking for memory leaks in ios_test targets.")
  public boolean runMemleaks;

  @Option(name = "ios_split_cpu",
      defaultValue = "",
      category = "undocumented",
      help =
          "Don't set this value from the command line - it is derived from  ios_multi_cpus only.")
  public String iosSplitCpu;

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

  // This option exists because two configurations are not allowed to have the same cache key
  // (partially derived from options). Since we have multiple transitions (see
  // getPotentialSplitTransitions below) that may result in the same configuration values at runtime
  // we need an artificial way to distinguish between them. This option must only be set by those
  // transitions for this purpose.
  // TODO(bazel-team): Remove this once we have dynamic configurations but make sure that different
  // configurations (e.g. by min os version) always use different output paths.
  @Option(name = "iOS configuration distinguisher",
      defaultValue = "UNKNOWN",
      converter = ConfigurationDistinguisherConverter.class,
      category = "undocumented")
  public ConfigurationDistinguisher configurationDistinguisher;

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
    name = "objc_includes_prioritize_system_libs",
    defaultValue = "false",
    category = "flags",
    help =
        "If set, the linker invocation will contain system library includes before framework"
        + " includes."
  )
  public boolean prioritizeSystemLibsOverFrameworks;
  
  @VisibleForTesting static final String DEFAULT_MINIMUM_IOS = "7.0";

  @Override
  public List<SplitTransition<BuildOptions>> getPotentialSplitTransitions() {
    return ImmutableList.of(
        IosApplication.SPLIT_ARCH_TRANSITION, IosExtension.MINIMUM_OS_AND_SPLIT_ARCH_TRANSITION);
  }

  /** Converter for the iOS configuration distinguisher. */
  public static final class ConfigurationDistinguisherConverter
      extends EnumConverter<ConfigurationDistinguisher> {
    public ConfigurationDistinguisherConverter() {
      super(ConfigurationDistinguisher.class, "Objective C configuration distinguisher");
    }
  }
}
