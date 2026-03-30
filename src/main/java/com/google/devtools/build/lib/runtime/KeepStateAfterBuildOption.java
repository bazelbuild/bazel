// Copyright 2025 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.runtime;

import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;

/**
 * An option class for <code>--keep_state_after_build</code>.
 *
 * <p>This needs to be separate from {@link CommonCommandOptions} because it's accessed from <code>
 * SkyframeExecutor</code> and referencing {@link CommonCommandOptions} would cause a dependency
 * cycle.
 */
public class KeepStateAfterBuildOption extends OptionsBase {
  @Option(
      name = "keep_state_after_build",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.BUILD_TIME_OPTIMIZATION,
      effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE},
      help =
          "If false, Blaze will discard the inmemory state from this build when the build "
              + "finishes. Subsequent builds will not have any incrementality with respect to this "
              + "one.")
  public boolean keepStateAfterBuild;
}
