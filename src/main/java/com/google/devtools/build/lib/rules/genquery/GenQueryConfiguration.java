// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.genquery;

import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.Fragment;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.RequiresOptions;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;

/** {@link Fragment} for {@link GenQuery}. */
@RequiresOptions(options = {GenQueryConfiguration.GenQueryOptions.class})
public class GenQueryConfiguration extends Fragment {

  /** GenQuery-specific options. */
  public static class GenQueryOptions extends FragmentOptions {
    // TODO(b/410585542): Remove this once there are no more internal users trying to set it.
    @Option(
        name = "experimental_skip_ttvs_for_genquery",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        // Should be OptionMetadataTag.DEPRECATED but test case experimentalOptionsPreservedInExec
        // explicitly asserts that all "experimental_" options use EXPERIMENTAL.
        metadataTags = {OptionMetadataTag.EXPERIMENTAL},
        help = "No-op. Will be removed soon.")
    public boolean skipTtvs;
  }

  public GenQueryConfiguration(BuildOptions buildOptions) {}

  /** Returns whether genquery should load its scope's transitive closure directly. */
  boolean skipTtvs() {
    return true;
  }
}
