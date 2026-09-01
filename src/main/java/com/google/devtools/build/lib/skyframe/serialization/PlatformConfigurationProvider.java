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

  /** Returns the baseline build options for the given platform label */
  BuildOptions getBaseOptionsForPlatform(
      Label platformLabel, boolean isExec, boolean trimTestOptions) throws SerializationException;

  /** Returns whether test options are trimmed away from the given build options. */
  boolean trimTestOptions(BuildOptions options);

  /** Returns whether platform-based output directory naming is enabled. */
  boolean usePlatformInOutputDir();

  /** Resolves the output directory mnemonic for the given configuration options. */
  String resolveMnemonic(BuildOptions targetOptions);
}
