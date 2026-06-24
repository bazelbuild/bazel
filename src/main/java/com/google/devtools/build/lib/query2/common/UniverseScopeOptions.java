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
package com.google.devtools.build.lib.query2.common;

import com.google.devtools.common.options.Converters;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsClass;
import java.util.List;

/**
 * The {@code --universe_scope} option, in its own options class so it can be shared by the {@code
 * query}, {@code cquery}, {@code aquery} and {@code build} commands without colliding.
 *
 * <p>It is deliberately <em>not</em> declared in {@link CommonQueryOptions}: the {@code cquery} and
 * {@code aquery} commands inherit the {@code build} command's options, so if {@code build} and the
 * query commands each surfaced their own {@code --universe_scope} (one via this class, one via
 * {@code CommonQueryOptions}) the option name would collide. Keeping a single definition that every
 * command references avoids that.
 */
@OptionsClass
public abstract class UniverseScopeOptions extends OptionsBase {

  @Option(
      name = "universe_scope",
      defaultValue = "",
      documentationCategory = OptionDocumentationCategory.QUERY,
      converter = Converters.CommaSeparatedOptionListConverter.class,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
      help =
          "A comma-separated set of target patterns (additive and subtractive). The query may be"
              + " performed in the universe defined by the transitive closure of the specified"
              + " targets. This option is used for the query, cquery and build commands.\n"
              + "For cquery and build, the input to this option is the targets all answers are built"
              + " under and so this option may affect configurations and transitions. If this option"
              + " is not specified, the top-level targets are assumed to be the targets parsed from"
              + " the query expression. Note: For cquery, not specifying this option may cause the"
              + " build to break if targets parsed from the query expression are not buildable with"
              + " top-level options.")
  public abstract List<String> getUniverseScope();
}
