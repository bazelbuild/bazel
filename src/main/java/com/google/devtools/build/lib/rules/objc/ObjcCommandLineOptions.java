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

import static com.google.devtools.build.xcode.common.BuildOptionsUtil.DEFAULT_OPTIONS_NAME;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.common.options.Option;

import java.util.List;

/**
 * Command-line options for building Objective-C targets.
 */
public class
    ObjcCommandLineOptions extends FragmentOptions {
  @Option(name = "ios_sdk_version",
      defaultValue = DEFAULT_SDK_VERSION,
      category = "build",
      help = "Specifies the version of the iOS SDK to use to build iOS applications."
      )
  public String iosSdkVersion;

  @VisibleForTesting static final String DEFAULT_SDK_VERSION = "8.1";

  @Option(name = "ios_simulator_version",
      defaultValue = "7.1",
      category = "run",
      help = "The version of iOS to run on the simulator when running or testing. This is ignored "
          + "for ios_test rules if a target device is specified in the rule.")
  public String iosSimulatorVersion;

  @Option(name = "ios_simulator_device",
      defaultValue = "iPhone 6",
      category = "run",
      help = "The device to simulate when running an iOS application in the simulator, e.g. "
          + "'iPhone 6'. You can get a list of devices by running 'xcrun simctl list devicetypes' "
          + "on the machine the simulator will be run on.")
  public String iosSimulatorDevice;

  @Option(name = "ios_cpu",
      defaultValue = "i386",
      category = "build",
      help = "Specifies to target CPU of iOS compilation.")
  public String iosCpu;

  @Option(name = "xcode_options",
      defaultValue = DEFAULT_OPTIONS_NAME,
      category = "undocumented",
      help = "Specifies the name of the build settings to use.")
  // TODO(danielwh): Do literally anything with this flag. Ideally, pass it to xcodegen via a
  // control proto.
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

  @VisibleForTesting static final String DEFAULT_MINIMUM_IOS = "7.0";

  @Override
  public void addAllLabels(Multimap<String, Label> labelMap) {}
}
