// Copyright 2014 Google Inc. All rights reserved.
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

package com.google.devtools.build.lib.rules.objc;

import static com.google.devtools.build.lib.packages.ImplicitOutputsFunction.fromFunctions;

import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction.SafeImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.BlazeRule;
import com.google.devtools.build.lib.view.RuleDefinition;
import com.google.devtools.build.lib.view.RuleDefinitionEnvironment;

/**
 * Rule definition for objc_library.
 */
@BlazeRule(name = "objc_library",
    factoryClass = ObjcLibrary.class,
    ancestors = { ObjcRuleClasses.ObjcBaseRule.class,
                  ObjcRuleClasses.ObjcSourcesRule.class })
public class ObjcLibraryRule implements RuleDefinition {
  /**
   * Output that only exists if there is at least one compilable source encapsulated by this rule.
   */
  private static final SafeImplicitOutputsFunction COMPILED_LIBRARY =
      new SafeImplicitOutputsFunction() {
        @Override
        public Iterable<String> getImplicitOutputs(AttributeMap map) {
          return PathFragment.safePathStrings(
              ObjcRuleClasses.outputAFilePackageRelativePath(map).asSet());
        }
      };

  @Override
  public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
    return builder
        /*<!-- #BLAZE_RULE(objc_library).IMPLICIT_OUTPUTS -->
        <ul>
         <li><code><var>name</var>.xcodeproj/project.pbxproj: An Xcode project file which can be
             used to develop or build on a Mac.</li>
        </ul>
        <!-- #END_BLAZE_RULE.IMPLICIT_OUTPUTS -->*/
        .setImplicitOutputsFunction(
            fromFunctions(COMPILED_LIBRARY, ObjcRuleClasses.PBXPROJ))
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = objc_library, TYPE = LIBRARY, FAMILY = Objective-C) -->

${ATTRIBUTE_SIGNATURE}

<p>This rule produces a static library from the given Objective-C source files.</p>

${ATTRIBUTE_DEFINITION}

<!-- #END_BLAZE_RULE -->*/
