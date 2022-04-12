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

package com.google.devtools.build.lib.rules.repository;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.Type.STRING;
import static com.google.devtools.build.lib.packages.Type.STRING_DICT;

import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;

/**
 * Base rule for rules in the WORKSPACE file.
 */
public class WorkspaceBaseRule implements RuleDefinition {
  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment environment) {
    return builder
        .exemptFromConstraintChecking("workspace rules aren't built for target environments")
        .add(
            attr("generator_name", STRING)
                .undocumented("internal")
                .nonconfigurable("internal attributes are non-configurable"))
        .add(
            attr("generator_function", STRING)
                .undocumented("internal")
                .nonconfigurable("internal attributes are non-configurable"))
        .add(
            attr("generator_location", STRING)
                .undocumented("internal")
                .nonconfigurable("internal attributes are non-configurable"))
        /* <!-- #BLAZE_RULE($workspace_base_rule).ATTRIBUTE(repo_mapping) -->
        A dictionary from local repository name to global repository name. This allows controls over
        workspace dependency resolution for dependencies of this repository.

        <p>For example, an entry <code>"@foo": "@bar"</code> declares that, for any time this
        repository depends on <code>"@foo"</code> (such as a dependency on
        <code>"@foo//some:target"</code>), it should actually resolve that dependency within
        globally-declared <code>"@bar"</code> (<code>"@bar//some:target"</code>).
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("repo_mapping", STRING_DICT))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("$workspace_base_rule")
        .type(RuleClassType.ABSTRACT)
        .build();
  }
}
