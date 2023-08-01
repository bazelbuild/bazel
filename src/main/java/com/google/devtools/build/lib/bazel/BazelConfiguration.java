// Copyright 2023 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.bazel;

import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.Fragment;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.RequiresOptions;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;

/** Bazel-specific settings. */
@Immutable
@RequiresOptions(options = {BazelConfiguration.Options.class})
public class BazelConfiguration extends Fragment {

  /** Command-line options. */
  public static class Options extends FragmentOptions {

    @Option(
        name = "incompatible_check_visibility_for_toolchains",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.INPUT_STRICTNESS,
        effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
        metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
        help = "If enabled, visibility checking also applies to toolchain implementations.")
    public boolean checkVisibilityForToolchains;

    @Override
    public FragmentOptions getExec() {
      Options exec = (Options) getDefault();
      exec.checkVisibilityForToolchains = checkVisibilityForToolchains;

      return exec;
    }
  }

  private final Options options;

  public BazelConfiguration(BuildOptions buildOptions) {
    this.options = buildOptions.get(Options.class);
  }

  public boolean checkVisibilityForToolchains() {
    return options.checkVisibilityForToolchains;
  }
}
