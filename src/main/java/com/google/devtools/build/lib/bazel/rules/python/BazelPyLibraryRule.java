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

package com.google.devtools.build.lib.bazel.rules.python;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;

import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.bazel.rules.python.BazelPyRuleClasses.PyBaseRule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.rules.python.PythonConfiguration;

/**
 * Rule definition for the {@code py_library} rule.
 */
public final class BazelPyLibraryRule implements RuleDefinition {
  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
    return builder
        .requiresConfigurationFragments(PythonConfiguration.class)

        /* <!-- #BLAZE_RULE(py_library).ATTRIBUTE(srcs) -->
        The list of source (<code>.py</code>) files that are processed to create the target.
        This includes all your checked-in code and any generated source files.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr("srcs", LABEL_LIST)
                .direct_compile_time_input()
                .allowedFileTypes(BazelPyRuleClasses.PYTHON_SOURCE))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("py_library")
        .ancestors(PyBaseRule.class)
        .factoryClass(BazelPyLibrary.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = py_library, TYPE = LIBRARY, FAMILY = Python) -->

<!-- #END_BLAZE_RULE -->*/
