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
package com.google.devtools.build.lib.runtime.commands;

import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsClass;

/**
 * The {@code --cquery} option for the {@code build} command. {@code --universe_scope} is provided
 * separately by {@link com.google.devtools.build.lib.query2.common.UniverseScopeOptions} (shared
 * with the query commands).
 */
@OptionsClass
public abstract class BuildCqueryOptions extends OptionsBase {

  @Option(
      name = "cquery",
      defaultValue = "",
      documentationCategory = OptionDocumentationCategory.GENERIC_INPUTS,
      effectTags = {OptionEffectTag.CHANGES_INPUTS},
      help =
          "If set, build evaluates this cquery expression over the post-analysis configured-target"
              + " graph and builds only the matched configured targets. The targets analyzed (the"
              + " universe) are taken from --universe_scope if set, otherwise inferred from the"
              + " patterns in this expression. Command-line target patterns, if any, restrict the"
              + " build to those labels (the expression then selects which of their configured"
              + " instances to build, e.g. via config()). Command-line patterns must be plain"
              + " labels: only --cquery accepts a cquery expression.")
  public abstract String getCquery();
}
