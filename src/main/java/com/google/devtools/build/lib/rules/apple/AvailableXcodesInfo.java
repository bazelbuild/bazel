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
package com.google.devtools.build.lib.rules.apple;

import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;

/** The available xcode versions computed from the {@code avaialable_xcodes} rule. */
@Immutable
public class AvailableXcodesInfo extends NativeInfo {
  /** Skylark name for this provider. */
  public static final String SKYLARK_NAME = "AvailableXcodesInfo";

  /** Provider identifier for {@link AvailableXcodesInfo}. */
  public static final BuiltinProvider<AvailableXcodesInfo> PROVIDER =
      new BuiltinProvider<AvailableXcodesInfo>(SKYLARK_NAME, AvailableXcodesInfo.class) {};

  private final Iterable<XcodeVersionRuleData> availableXcodes;
  private final XcodeVersionRuleData defaultVersion;

  public AvailableXcodesInfo(
      Iterable<XcodeVersionRuleData> availableXcodes, XcodeVersionRuleData defaultVersion) {
    super(PROVIDER);
    this.availableXcodes = availableXcodes;
    this.defaultVersion = defaultVersion;
  }

  /** Returns the available xcode versions from {@code available_xcodes}. */
  public Iterable<XcodeVersionRuleData> getAvailableVersions() {
    return availableXcodes;
  }

  /** Returns the default xcode version from {@code available_xcodes}. */
  public XcodeVersionRuleData getDefaultVersion() {
    return defaultVersion;
  }
}
