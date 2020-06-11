// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.collect.nestedset;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;

/** A {@link BlazeModule} handling options pertaining to {@link NestedSet}. */
public class NestedSetOptionsModule extends BlazeModule {

  /** Command line options controlling the behavior of {@link NestedSet}. */
  public static final class Options extends OptionsBase {
    @Option(
        name = "nested_set_depth_limit",
        defaultValue = "3500",
        documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
        effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
        help =
            "Limit on the depth of NestedSet, which is the internal data structure used to "
                + "implement `depset` in Starlark. If a depset is flattened during evaluation of "
                + "Starlark code or a NestedSet is flattened internally, and that data structure "
                + "has a depth exceeding this limit, then the Bazel invocation will fail.")
    public int nestedSetDepthLimit;
  }

  @Override
  public void beforeCommand(CommandEnvironment env) {
    Options options = env.getOptions().getOptions(Options.class);
    boolean changed = NestedSet.setApplicationDepthLimit(options.nestedSetDepthLimit);
    if (changed) {
      env.getSkyframeExecutor().resetEvaluator();
    }
  }

  @Override
  public Iterable<Class<? extends OptionsBase>> getCommonCommandOptions() {
    return ImmutableList.of(Options.class);
  }
}
