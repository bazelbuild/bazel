// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.rules.sh;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.Attribute.AllowedValueSet;
import com.google.devtools.build.lib.packages.PredicateWithMessage;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.util.FileTypeSet;
import javax.annotation.Nullable;

/**
 * Rule definitions for rule classes implementing shell support.
 */
public final class BazelShRuleClasses {

  static final ImmutableSet<String> ALLOWED_RULES_IN_DEPS_WITH_WARNING =
      ImmutableSet.of("filegroup", "genrule", "sh_binary", "sh_test", "test_suite");

  /**
   * Common attributes for shell rules.
   */
  public static final class ShRule implements RuleDefinition {
    @Override
    public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment environment) {
      return builder
          /* <!-- #BLAZE_RULE($sh_target).ATTRIBUTE(srcs) -->
          The file containing the shell script.
          <p>
            This attribute must be a singleton list, whose element is the shell script.
            This script must be executable, and may be a source file or a generated file.
            All other files required at runtime (whether scripts or data) belong in the
            <code>data</code> attribute.
          </p>
          <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
          .add(attr("srcs", LABEL_LIST).mandatory().allowedFileTypes(FileTypeSet.ANY_FILE))
          /* <!-- #BLAZE_RULE($sh_target).ATTRIBUTE(deps) -->
          The list of "library" targets to be aggregated into this target.
          See general comments about <code>deps</code>
          at <a href="${link common-definitions#typical.deps}">Typical attributes defined by
          most build rules</a>.
          <p>
            This attribute should be used to list other <code>sh_library</code> rules that provide
            interpreted program source code depended on by the code in <code>srcs</code>. The files
            provided by these rules will be present among the <code>runfiles</code> of this target.
          </p>
          <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
          .override(
              builder
                  .copy("deps")
                  .allowedRuleClasses("sh_library")
                  .allowedRuleClassesWithWarning(ALLOWED_RULES_IN_DEPS_WITH_WARNING)
                  .allowedFileTypes())
          .build();
    }

    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("$sh_target")
          .type(RuleClassType.ABSTRACT)
          .ancestors(BaseRuleClasses.NativeActionCreatingRule.class)
          .build();
    }
  }

  /**
   * Convenience structure for the bash dependency combinations defined
   * by BASH_BINARY_BINDINGS.
   */
  static class BashBinaryBinding {
    public BashBinaryBinding(@Nullable String execPath) {
    }
  }

  /**
   * Attribute value specifying the local system's bash version.
   */
  static final String SYSTEM_BASH_VERSION = "system";

  static final ImmutableMap<String, BashBinaryBinding> BASH_BINARY_BINDINGS =
      ImmutableMap.of(
          // "system": don't package any bash with the target, but rather use whatever is
          // available on the system the script is run on.
          SYSTEM_BASH_VERSION, new BashBinaryBinding("/bin/bash"));

  // TODO(bazel-team): refactor sh_binary and sh_base to have a common root
  // with srcs and bash_version attributes
  static final PredicateWithMessage<Object> BASH_VERSION_ALLOWED_VALUES =
      new AllowedValueSet(BASH_BINARY_BINDINGS.keySet());
}
