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
package com.google.devtools.build.lib.query2;

import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Setting;
import com.google.devtools.common.options.Converters;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

/** Options shared between blaze query and blaze cquery. */
public class CommonQueryOptions extends OptionsBase {
  @Option(
    name = "universe_scope",
    defaultValue = "",
    documentationCategory = OptionDocumentationCategory.QUERY,
    converter = Converters.CommaSeparatedOptionListConverter.class,
    effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
    help =
        "A comma-separated set of target patterns (additive and subtractive). The query may "
            + "be performed in the universe defined by the transitive closure of the specified "
            + "targets. This option is used for the query and cquery commands. \n"
            + "For cquery, the input to this option is the targets all answers are built under and "
            + "so this option may affect configurations and transitions. If this option is not "
            + "specified, the top-level targets are assumed to be the targets parsed from the "
            + "query expression. Note: For cquery, not specifying this option may cause the build "
            + "to break if targets parsed from the query expression are not buildable with "
            + "top-level options."
  )
  public List<String> universeScope;

  @Option(
    name = "host_deps",
    defaultValue = "true",
    documentationCategory = OptionDocumentationCategory.QUERY,
    effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
    help =
        "Query: If disabled, dependencies on 'host configuration' targets will not be included in "
            + "the dependency graph over which the query operates. A 'host configuration' "
            + "dependency edge, such as the one from any 'proto_library' rule to the Protocol "
            + "Compiler, usually points to a tool executed during the build (on the host machine) "
            + "rather than a part of the same 'target' program. \n"
            + "Cquery: If disabled, filters out all configured targets which cross a host "
            + "transition from the top-level target that discovered this configured target. That "
            + "means if the top-level target is in the target configuration, only configured "
            + "targets also in the target configuration will be returned. If the top-level target "
            + "is in the host configuration, only host configured targets will be returned."
  )
  public boolean includeHostDeps;

  @Option(
    name = "implicit_deps",
    defaultValue = "true",
    category = "query",
    documentationCategory = OptionDocumentationCategory.QUERY,
    effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
    help =
        "If enabled, implicit dependencies will be included in the dependency graph over "
            + "which the query operates. An implicit dependency is one that is not explicitly "
            + "specified in the BUILD file but added by blaze."
  )
  public boolean includeImplicitDeps;

  /** Return the current options as a set of QueryEnvironment settings. */
  public Set<Setting> toSettings() {
    Set<Setting> settings = EnumSet.noneOf(Setting.class);
    if (!includeHostDeps) {
      settings.add(Setting.NO_HOST_DEPS);
    }
    if (!includeImplicitDeps) {
      settings.add(Setting.NO_IMPLICIT_DEPS);
    }
    return settings;
  }
}
