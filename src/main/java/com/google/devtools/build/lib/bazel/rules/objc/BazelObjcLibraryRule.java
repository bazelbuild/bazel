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

package com.google.devtools.build.lib.bazel.rules.objc;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.rules.objc.ObjcLibraryBaseRule;

/** Rule definition for {@code objc_library}. */
public class BazelObjcLibraryRule implements RuleDefinition {

  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
    return builder.build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("objc_library")
        .factoryClass(BaseRuleClasses.EmptyRuleConfiguredTargetFactory.class)
        .ancestors(ObjcLibraryBaseRule.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = objc_library, TYPE = LIBRARY, FAMILY = Objective-C) -->

<p>This rule produces a static library from the given Objective-C source files.</p>

<!-- #END_BLAZE_RULE -->*/
