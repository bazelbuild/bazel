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

import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.rules.objc.ObjcImportBaseRule;

/** Rule definition for {@code objc_import}. */
public class BazelObjcImportRule implements RuleDefinition {
  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment environment) {
    return builder.build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("objc_import")
        .factoryClass(BazelObjcImport.class)
        .ancestors(ObjcImportBaseRule.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = objc_import, TYPE = LIBRARY, FAMILY = Objective-C) -->

<p>This rule encapsulates an already-compiled static library in the form of an
<code>.a</code> file. It also allows exporting headers and resources using the same
attributes supported by <code>objc_library</code>.</p>

<!-- #END_BLAZE_RULE -->*/
