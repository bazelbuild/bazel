// Copyright 2016 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.bazel.rules.java.BazelJavaRuleClasses.JavaBaseRule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.rules.android.AarImportBaseRule;

/**
 * Rule definition for the {@code aar_import} rule.
 */
public final class BazelAarImportRule implements RuleDefinition {

  @Override
  public RuleClass build(Builder builder, RuleDefinitionEnvironment environment) {
    return builder
        .removeAttribute("javacopts")
        .removeAttribute("plugins")
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("aar_import")
        .ancestors(AarImportBaseRule.class, JavaBaseRule.class)
        .factoryClass(BazelAarImport.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = aar_import, TYPE = LIBRARY, FAMILY = Android) -->

<p>
  This rule allows the use of <code>.aar</code> files as libraries for
  <code><a href="${link android_library}">android_library</a></code> and
  <code><a href="${link android_binary}">android_binary</a></code> rules.
</p>

<h4 id="aar_import_examples">Examples</h4>

<pre class="code">
    aar_import(
        name = "google-vr-sdk",
        aar = "gvr-android-sdk/libraries/sdk-common-1.10.0.aar",
    )

    android_binary(
        name = "app",
        manifest = "AndroidManifest.xml",
        srcs = glob(["**.java"]),
        deps = [":google-vr-sdk"],
    )
</pre>

<!-- #END_BLAZE_RULE -->*/
