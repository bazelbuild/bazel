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

package com.google.devtools.build.lib.bazel.rules.objc;

import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.rules.objc.AppleCrosstoolTransition;
import com.google.devtools.build.lib.rules.objc.J2ObjcLibraryBaseRule;
import com.google.devtools.build.lib.rules.objc.ObjcRuleClasses;

/** <code>j2objc_library</code> rule declaration. */
public class BazelJ2ObjcLibraryRule implements RuleDefinition {
  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
    return builder
        .cfg(AppleCrosstoolTransition.APPLE_CROSSTOOL_TRANSITION)
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("j2objc_library")
        .factoryClass(BazelJ2ObjcLibrary.class)
        .ancestors(
            J2ObjcLibraryBaseRule.class,
            ObjcRuleClasses.CrosstoolRule.class,
            ObjcRuleClasses.XcrunRule.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = j2objc_library, TYPE = LIBRARY, FAMILY = Objective-C) -->

<p> This rule uses <a href="https://github.com/google/j2objc">J2ObjC</a> to translate Java source
files to Objective-C, which then can be used used as dependencies of objc_library and objc_binary
rules. Detailed information about J2ObjC itself can be found at  <a href="http://j2objc.org">the
J2ObjC site</a>
</p>
<p>Custom J2ObjC transpilation flags can be specified using the build flag
<code>--j2objc_translation_flags</code> in the command line.
</p>
<p>Please note that the translated files included in a j2objc_library target will be
compiled using the default compilation configuration, the same configuration as for the sources of
an objc_library rule with no compilation options specified in attributes.
</p>
<p>Plus, generated code is de-duplicated at target level, not source level. If you have two
different Java targets that include the same Java source files, you may see a duplicate symbol error
at link time. The correct way to resolve this issue is to move the shared Java source files into a
separate common target that can be depended upon.
</p>


<!-- #END_BLAZE_RULE -->*/
