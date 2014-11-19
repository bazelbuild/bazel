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

import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.view.config.FragmentOptions;
import com.google.devtools.common.options.Option;

/**
 * Command-line options for building Objective-C targets.
 */
public class ObjcCommandLineOptions extends FragmentOptions {
  @Option(name = "ios_sdk_version",
      defaultValue = "7.1",
      category = "undocumented",
      help = "Specifies the version of the iOS SDK to use to build iOS applications."
      )
  public String iosSdkVersion;

  @Option(name = "ios_simulator_version",
      defaultValue = "7.1",
      category = "undocumented",
      help = "The version of iOS to run on the simulator when running tests.")
  public String iosSimulatorVersion;

  @Option(name = "ios_cpu",
      defaultValue = "i386",
      category = "undocumented",
      help = "Specifies to target CPU of iOS compilation.")
  public String iosCpu;

  @Option(name = "xcode_options",
      defaultValue = DEFAULT_OPTIONS_NAME,
      category = "undocumented",
      help = "Specifies the name of the build settings to use.")
  public String xcodeOptions;

  @Option(name = "objc_generate_debug_symbols",
      defaultValue = "false",
      category = "undocumented",
      help = "Specifies whether to generate debug symbol(.dSYM) file.")
  public boolean generateDebugSymbols;

  @Override
  public void addAllLabels(Multimap<String, Label> labelMap) {}
}
