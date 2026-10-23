// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.serialization;

import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.cmdline.Label;

/**
 * Dynamic provider interface to resolve platform-specific configuration baselines and mnemonics.
 */
public interface PlatformConfigurationProvider {

  /** Returns the top-level platform label. */
  Label getTopLevelPlatformLabel();

  /**
   * Returns the baseline build options for the given platform label.
   *
   * <p>Configurations are serialized as diffs against this baseline, and the baseline itself is not
   * serialized. So every option in it must either be covered by the configuration checksum in
   * {@code FrontierNodeVersion} or be the same in every build. Otherwise two builds that differ in
   * that option would serialize keys to the same bytes and share cache entries.
   */
  BuildOptions getBaseOptionsForPlatform(
      Label platformLabel, boolean isExec, boolean trimTestOptions) throws SerializationException;

  /**
   * Returns this build's baseline build options for the given platform label.
   *
   * <p>For trimmed configurations this is the same as {@link #getBaseOptionsForPlatform}. Unlike
   * that method, it doesn't replace anything for untrimmed configurations: it keeps this build's
   * test options and full {@code --run_under}. Those are not covered by the configuration checksum
   * in {@code FrontierNodeVersion}, but output directory mnemonics of untrimmed configurations are
   * computed against them, so serialized keys of untrimmed configurations must also record how this
   * baseline differs from {@link #getBaseOptionsForPlatform}.
   */
  BuildOptions getActualBaseOptionsForPlatform(
      Label platformLabel, boolean isExec, boolean trimTestOptions) throws SerializationException;

  /** Resolves the output directory mnemonic for the given configuration options. */
  String resolveMnemonic(BuildOptions targetOptions);
}
