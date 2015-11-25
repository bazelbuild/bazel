// Copyright 2015 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Optional;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;

/**
 * Provides a version of xcode based on a combination of the {@code --xcode_version} build flag
 * and a {@code xcode_config} target. This version of xcode should be used for selecting apple
 * toolchains and SDKs.
 */
@Immutable
public final class XcodeConfigProvider implements TransitiveInfoProvider {
  private final Optional<DottedVersion> xcodeVersion;
  
  XcodeConfigProvider(DottedVersion xcodeVersion) {
    this.xcodeVersion = Optional.of(xcodeVersion);
  }
  
  private XcodeConfigProvider() {
    this.xcodeVersion = Optional.absent();
  }
  
  /**
   * Returns a {@link XcodeConfigProvider} with no xcode version specified. The host system
   * default xcode should be used. See {@link #getXcodeVersion}.
   */
  static XcodeConfigProvider hostSystemDefault() {
    return new XcodeConfigProvider();
  }

  /**
   * Returns either an explicit xcode version which should be used in actions which require an
   * apple toolchain, or {@link Optional#absent} if the host system default should be used.
   */
  public Optional<DottedVersion> getXcodeVersion() {
    return xcodeVersion;
  }
}
