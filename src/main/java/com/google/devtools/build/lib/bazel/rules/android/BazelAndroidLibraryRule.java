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

import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.bazel.rules.java.BazelJavaRuleClasses;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.rules.android.AndroidConfiguration;
import com.google.devtools.build.lib.rules.android.AndroidLibraryBaseRule;
import com.google.devtools.build.lib.rules.android.AndroidRuleClasses;
import com.google.devtools.build.lib.rules.java.JavaConfiguration;

/**
 * Definition of the {@code android_library} rule for Bazel.
 */
public class BazelAndroidLibraryRule implements RuleDefinition {

  @Override
  public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
    return builder
        .requiresConfigurationFragments(
            JavaConfiguration.class, AndroidConfiguration.class)
        /* <!-- #BLAZE_RULE(android_library).IMPLICIT_OUTPUTS -->
        <ul>
          <li><code>lib<var>name</var>.jar</code>: A Java archive.</li>
          <li><code>lib<var>name</var>-src.jar</code>: An archive containing the
          sources ("source jar").</li>
          <li><code><var>name</var>.aar</code>: An android 'aar' bundle containing the java archive
          and resources of this target. It does not contain the transitive closure.</li>
        </ul>
        <!-- #END_BLAZE_RULE.IMPLICIT_OUTPUTS --> */
        .setImplicitOutputsFunction(AndroidRuleClasses.ANDROID_LIBRARY_IMPLICIT_OUTPUTS)
        .build();

  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("android_library")
        .ancestors(
            BazelJavaRuleClasses.JavaBaseRule.class,
            AndroidLibraryBaseRule.class)
        .factoryClass(BazelAndroidLibrary.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = android_library, TYPE = LIBRARY, FAMILY = Android) -->

<p>This rule compiles and archives its sources into a <code>.jar</code> file.
  The Android runtime library <code>android.jar</code> is implicitly put on
  the compilation class path.
</p>

<p>If you need to depend on the appcompat library, put
<code>//external:android/appcompat_v4</code> or <code>//external:android/appcompat_v7</code>
in the <code>deps</code> attribute.
</p>
${IMPLICIT_OUTPUTS}


<h4 id="android_library_examples">Examples</h4>
<p>Examples of Android rules can be found in the <code>examples/android</code> directory of the
Bazel source tree.

<!-- #END_BLAZE_RULE -->*/
