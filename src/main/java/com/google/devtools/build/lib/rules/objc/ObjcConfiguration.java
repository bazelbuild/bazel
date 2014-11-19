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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.view.config.BuildConfiguration.Fragment;
import com.google.devtools.build.xcode.common.Platform;

/**
 * A compiler configuration containing flags required for Objective-C compilation.
 */
public class ObjcConfiguration extends Fragment {
  private final String iosSdkVersion;
  private final String iosSimulatorVersion;
  private final String iosCpu;
  private final String xcodeOptions;
  private final boolean generateDebugSymbols;

  ObjcConfiguration(ObjcCommandLineOptions objcOptions) {
    this.iosSdkVersion = Preconditions.checkNotNull(objcOptions.iosSdkVersion, "iosSdkVersion");
    this.iosSimulatorVersion =
        Preconditions.checkNotNull(objcOptions.iosSimulatorVersion, "iosSimulatorVersion");
    this.iosCpu = Preconditions.checkNotNull(objcOptions.iosCpu, "iosCpu");
    this.xcodeOptions = Preconditions.checkNotNull(objcOptions.xcodeOptions, "xcodeOptions");
    this.generateDebugSymbols = objcOptions.generateDebugSymbols;
  }

  public String getIosSdkVersion() {
    return iosSdkVersion;
  }

  public String getIosSimulatorVersion() {
    return iosSimulatorVersion;
  }

  public String getIosCpu() {
    return iosCpu;
  }

  public Platform getPlatform() {
    return Platform.forArch(getIosCpu());
  }

  public String getXcodeOptions() {
    return xcodeOptions;
  }

  public boolean generateDebugSymbols() {
    return generateDebugSymbols;
  }

  @Override
  public String getName() {
    return "Objective-C";
  }

  @Override
  public String cacheKey() {
    return iosSdkVersion;
  }
}
