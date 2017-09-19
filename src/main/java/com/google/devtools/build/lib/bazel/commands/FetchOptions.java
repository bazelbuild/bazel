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

package com.google.devtools.build.lib.bazel.commands;

import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;

/**
 * Command-line options for the fetch command.
 */
public class FetchOptions extends OptionsBase {
  @Option(
    name = "keep_going",
    abbrev = 'k',
    defaultValue = "false",
    category = "strategy",
    documentationCategory = OptionDocumentationCategory.EXECUTION_STRATEGY,
    effectTags = {OptionEffectTag.EAGERNESS_TO_EXIT},
    help =
        "Continue as much as possible after an error.  While the target that failed and those "
            + "that depend on it cannot be analyzed, other prerequisites of these targets can be."
  )
  public boolean keepGoing;
}
