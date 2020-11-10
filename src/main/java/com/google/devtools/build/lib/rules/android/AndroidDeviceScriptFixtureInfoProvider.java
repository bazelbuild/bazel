// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.android;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;

/**
 * Information about an {@code android_device_script_fixture} to run as part of an {@code
 * android_instrumentation_test}.
 */
@Immutable
public class AndroidDeviceScriptFixtureInfoProvider extends NativeInfo {

  private static final String STARLARK_NAME = "DeviceScriptFixtureInfo";
  public static final BuiltinProvider<AndroidDeviceScriptFixtureInfoProvider> STARLARK_CONSTRUCTOR =
      new BuiltinProvider<AndroidDeviceScriptFixtureInfoProvider>(
          STARLARK_NAME, AndroidDeviceScriptFixtureInfoProvider.class) {};

  private final Artifact fixtureScript;
  private final NestedSet<Artifact> supportApks;
  private final boolean daemon;
  private final boolean strictExit;

  public AndroidDeviceScriptFixtureInfoProvider(
      Artifact fixtureScript, NestedSet<Artifact> supportApks, boolean daemon, boolean strictExit) {
    this.fixtureScript = fixtureScript;
    this.supportApks = supportApks;
    this.daemon = daemon;
    this.strictExit = strictExit;
  }

  @Override
  public BuiltinProvider<AndroidDeviceScriptFixtureInfoProvider> getProvider() {
    return STARLARK_CONSTRUCTOR;
  }

  public Artifact getFixtureScript() {
    return fixtureScript;
  }

  public NestedSet<Artifact> getSupportApks() {
    return supportApks;
  }

  public boolean getDaemon() {
    return daemon;
  }

  public boolean getStrictExit() {
    return strictExit;
  }
}
