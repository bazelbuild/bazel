// Copyright 2014 Google Inc. All rights reserved.
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
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.LabelConverter;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.packages.Attribute.SplitTransition;
import com.google.devtools.build.lib.rules.objc.ReleaseBundlingSupport.SplitArchTransition.ConfigurationDistinguisher;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.common.options.Converters.CommaSeparatedOptionListConverter;
import com.google.devtools.common.options.EnumConverter;
import com.google.devtools.common.options.Option;

import java.util.List;

/**
 * Command-line options for building Objective-C targets.
 */
public class ObjcCommandLineOptions extends FragmentOptions {
  // TODO(cparsons): Validate version flag value.
  @Option(name = "xcode_version",
      defaultValue = "",
      category = "undocumented",
      help = "If specified, uses xcode of the given version for relevant build actions. "
          + "If unspecified, uses the executor default version of xcode."
      )
  public String xcodeVersion;

  @Option(name = "ios_sdk_version",
      defaultValue = DEFAULT_SDK_VERSION,
      category = "build",
      help = "Specifies the version of the iOS SDK to use to build iOS applications."
      )
  public String iosSdkVersion;

  @VisibleForTesting static final String DEFAULT_SDK_VERSION = "8.4";

  @Option(name = "ios_simulator_version",
      defaultValue = "8.4",
      category = "run",
      deprecationWarning = "Use target_device instead to drive the simulator to use.",
      help = "The version of iOS to run on the simulator when running or testing. This is ignored "
          + "for ios_test rules if a target device is specified in the rule.")
  public String iosSimulatorVersion;

  @Option(name = "ios_simulator_device",
      defaultValue = "iPhone 5s",
      category = "run",
      help = "The device to simulate when running an iOS application in the simulator, e.g. "
          + "'iPhone 6'. You can get a list of devices by running 'xcrun simctl list devicetypes' "
          + "on the machine the simulator will be run on.")
  public String iosSimulatorDevice;

  @Option(name = "ios_cpu",
      defaultValue = DEFAULT_IOS_CPU,
      category = "build",
      help = "Specifies to target CPU of iOS compilation.")
  public String iosCpu;

  @Option(name = "xcode_options",
      defaultValue = "Debug",
      category = "undocumented",
      deprecationWarning = "Use --compilation_mode instead.",
      help = "Specifies the name of the build settings to use.")
  public String xcodeOptions;

  @Option(name = "objc_generate_debug_symbols",
      defaultValue = "false",
      category = "undocumented",
      help = "Specifies whether to generate debug symbol(.dSYM) file.")
  public boolean generateDebugSymbols;

  @Option(name = "objccopt",
      allowMultiple = true,
      defaultValue = "",
      category = "flags",
      help = "Additional options to pass to Objective C compilation.")
  public List<String> copts;

  @Option(name = "ios_minimum_os",
      defaultValue = DEFAULT_MINIMUM_IOS,
      category = "flags",
      help = "Minimum compatible iOS version for target simulators and devices.")
  public String iosMinimumOs;

  @Option(name = "ios_memleaks",
      defaultValue =  "false",
      category = "misc",
      help = "Enable checking for memory leaks in ios_test targets.")
  public boolean runMemleaks;

  @Option(name = "ios_multi_cpus",
      converter = CommaSeparatedOptionListConverter.class,
      defaultValue = "",
      category = "flags",
      help = "Comma-separated list of architectures to build an ios_application with. The result "
          + "is a universal binary containing all specified architectures.")
  public List<String> iosMultiCpus;

  @Option(name = "ios_split_cpu",
      defaultValue = "",
      category = "undocumented",
      help =
          "Don't set this value from the command line - it is derived from  ios_multi_cpus only.")
  public String iosSplitCpu;

  @Option(name = "objc_dump_syms_binary",
      defaultValue = "//tools/objc:dump_syms",
      category = "undocumented",
      converter = LabelConverter.class)
  public Label dumpSyms;

  @Option(name = "default_ios_provisiong_profile",
      defaultValue = "//tools/objc:default_provisioning_profile",
      category = "undocumented",
      converter = LabelConverter.class)
  public Label defaultProvisioningProfile;

  @Option(name = "objc_per_proto_includes",
      defaultValue = "false",
      category = "undocumented",
      help = "Whether to add include path entries for every individual proto file.")
  public boolean perProtoIncludes;

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

  @Option(name = "objc_enable_binary_stripping",
      defaultValue = "false",
      category = "flags",
      help = "Whether to perform symbol and dead-code strippings on linked binaries. Binary "
          + "strippings will be performed if both this flag and --compilationMode=opt are "
          + "specified.")
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

  @VisibleForTesting static final String DEFAULT_MINIMUM_IOS = "7.0";
  @VisibleForTesting static final String DEFAULT_IOS_CPU = "x86_64";

  @Override
  public void addAllLabels(Multimap<String, Label> labelMap) {
    if (generateDebugSymbols) {
      labelMap.put("dump_syms", dumpSyms);
    }

    if (getPlatform() == Platform.DEVICE) {
      labelMap.put("default_provisioning_profile", defaultProvisioningProfile);
    }
  }

  private Platform getPlatform() {
    for (String architecture : iosMultiCpus) {
      if (Platform.forArch(architecture) == Platform.DEVICE) {
        return Platform.DEVICE;
      }
    }
    return Platform.forArch(iosCpu);
  }

  @Override
  public List<SplitTransition<BuildOptions>> getPotentialSplitTransitions() {
    return ImmutableList.of(
        IosApplication.SPLIT_ARCH_TRANSITION, IosExtension.MINIMUM_OS_AND_SPLIT_ARCH_TRANSITION);
  }

  public static final class ConfigurationDistinguisherConverter
      extends EnumConverter<ConfigurationDistinguisher> {
    public ConfigurationDistinguisherConverter() {
      super(ConfigurationDistinguisher.class, "Objective C configuration distinguisher");
    }
  }
}
