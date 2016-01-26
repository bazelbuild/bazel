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
package com.google.devtools.build.lib.bazel.rules.android;

import static com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition.HOST;
import static com.google.devtools.build.lib.packages.Attribute.attr;

import com.google.devtools.build.lib.Constants;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.bazel.rules.cpp.BazelCppRuleClasses;
import com.google.devtools.build.lib.bazel.rules.java.BazelJavaRuleClasses;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.rules.android.AndroidBinaryOnlyRule;
import com.google.devtools.build.lib.rules.android.AndroidConfiguration;
import com.google.devtools.build.lib.rules.android.AndroidRuleClasses;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration;
import com.google.devtools.build.lib.rules.java.JavaConfiguration;

/**
 * Rule class definition for {@code android_binary}.
 */
public class BazelAndroidBinaryRule implements RuleDefinition {

  @Override
  public RuleClass build(Builder builder, RuleDefinitionEnvironment environment) {
    return builder
        .requiresConfigurationFragments(
            AndroidConfiguration.class, JavaConfiguration.class, CppConfiguration.class)
        .add(attr("$debug_keystore", BuildType.LABEL)
            .cfg(HOST)
            .singleArtifact()
            .value(environment.getLabel(
                Constants.TOOLS_REPOSITORY + "//tools/android:debug_keystore")))
        .add(attr(":cc_toolchain_split", BuildType.LABEL)
            .cfg(AndroidRuleClasses.ANDROID_SPLIT_TRANSITION)
            .value(BazelCppRuleClasses.CC_TOOLCHAIN))
        /* <!-- #BLAZE_RULE(android_binary).IMPLICIT_OUTPUTS -->
         <ul>
         <li><code><var>name</var>.apk</code>: An Android application
          package file signed with debug keys and
          <a href="http://developer.android.com/guide/developing/tools/zipalign.html">
          zipaligned</a>, it could be used to develop and debug your application.
          You cannot release your application when signed with the debug keys.</li>
          <li><code><var>name</var>_unsigned.apk</code>: An unsigned version of the
            above file that could be signed with the release keys before release to
            the public.
          </li>
          <li><code><var>name</var>_deploy.jar</code>: A Java archive containing the
            transitive closure of this target.
            <p>The deploy jar contains all the classes that would be found by a
            classloader that searched the runtime classpath of this target
            from beginning to end.</p>
          </li>
          <li><code><var>name</var>_proguard.jar</code>: A Java archive containing
            the result of running ProGuard on the
            <code><var>name</var>_deploy.jar</code>.
            This output is only produced if
            <a href="#android_binary.proguard_specs">proguard_specs</a> attribute is
            specified.
          </li>
          <li><code><var>name</var>_proguard.map</code>: A mapping file result of
            running ProGuard on the <code><var>name</var>_deploy.jar</code>.
            This output is only produced if
            <a href="#android_binary.proguard_specs">proguard_specs</a> attribute is
            specified and
            <a href="#android_binary.proguard_generate_mapping">proguard_generate_mapping</a>
            is set.
          </li>
        </ul>
        <!-- #END_BLAZE_RULE.IMPLICIT_OUTPUTS --> */
        .setImplicitOutputsFunction(AndroidRuleClasses.ANDROID_BINARY_IMPLICIT_OUTPUTS)
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("android_binary")
        .ancestors(
            AndroidBinaryOnlyRule.class,
            BazelJavaRuleClasses.JavaBaseRule.class,
            BazelCppRuleClasses.CcLinkingRule.class)
        .factoryClass(BazelAndroidBinary.class)
        .build();
  }
}
/*<!-- #BLAZE_RULE (NAME = android_binary, TYPE = BINARY, FAMILY = Android) -->

<p>
  Produces Android application package files (.apk).
</p>

${IMPLICIT_OUTPUTS}

<h4 id="android_binary_examples">Examples</h4>
<p>Examples of Android rules can be found in the <code>examples/android</code> directory of the
Bazel source tree.

<!-- #END_BLAZE_RULE -->*/
