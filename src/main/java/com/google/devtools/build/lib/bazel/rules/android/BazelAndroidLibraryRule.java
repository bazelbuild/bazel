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

import com.google.devtools.build.lib.analysis.Allowlist;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.bazel.rules.java.BazelJavaRuleClasses;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.ToolchainTransitionMode;
import com.google.devtools.build.lib.rules.android.AndroidLibraryBaseRule;
import com.google.devtools.build.lib.rules.android.AndroidRuleClasses;

/**
 * Definition of the {@code android_library} rule for Bazel.
 */
public class BazelAndroidLibraryRule implements RuleDefinition {

  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
    return builder
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
        .useToolchainTransition(ToolchainTransitionMode.ENABLED)
        .add(
            Allowlist.getAttributeFromAllowlistName("allow_deps_without_srcs")
                .value(
                    env.getToolsLabel(
                        "//tools/android:allow_android_library_deps_without_srcs_allowlist")))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("android_library")
        .ancestors(
            BazelJavaRuleClasses.JavaBaseRule.class,
            AndroidLibraryBaseRule.class,
            BazelSdkToolchainRule.class)
        .factoryClass(BazelAndroidLibrary.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = android_library, TYPE = LIBRARY, FAMILY = Android) -->

<p>This rule compiles and archives its sources into a <code>.jar</code> file.
  The Android runtime library <code>android.jar</code> is implicitly put on
  the compilation class path.
</p>

${IMPLICIT_OUTPUTS}


<h4 id="android_library_examples">Examples</h4>
<p>Examples of Android rules can be found in the <code>examples/android</code> directory of the
Bazel source tree.

<p id="android_library_examples.idl_import_root">The following example shows
how to set <code>idl_import_root</code>.
Let <code>//java/bazel/helloandroid/BUILD</code> contain:</p>
<pre class="code">
android_library(
    name = "parcelable",
    srcs = ["MyParcelable.java"], # bazel.helloandroid.MyParcelable

    # MyParcelable.aidl will be used as import for other .aidl
    # files that depend on it, but will not be compiled.
    idl_parcelables = ["MyParcelable.aidl"] # bazel.helloandroid.MyParcelable

    # We don't need to specify idl_import_root since the aidl file
    # which declares bazel.helloandroid.MyParcelable
    # is present at java/bazel/helloandroid/MyParcelable.aidl
    # underneath a java root (java/).
)

android_library(
    name = "foreign_parcelable",
    srcs = ["src/android/helloandroid/OtherParcelable.java"], # android.helloandroid.OtherParcelable
    idl_parcelables = [
        "src/android/helloandroid/OtherParcelable.aidl" # android.helloandroid.OtherParcelable
    ],

    # We need to specify idl_import_root because the aidl file which
    # declares android.helloandroid.OtherParcelable is not positioned
    # at android/helloandroid/OtherParcelable.aidl under a normal java root.
    # Setting idl_import_root to "src" in //java/bazel/helloandroid
    # adds java/bazel/helloandroid/src to the list of roots
    # the aidl compiler will search for imported types.
    idl_import_root = "src",
)

# Here, OtherInterface.aidl has an "import android.helloandroid.CallbackInterface;" statement.
android_library(
    name = "foreign_interface",
    idl_srcs = [
        "src/android/helloandroid/OtherInterface.aidl" # android.helloandroid.OtherInterface
        "src/android/helloandroid/CallbackInterface.aidl" # android.helloandroid.CallbackInterface
    ],

    # As above, idl_srcs which are not correctly positioned under a java root
    # must have idl_import_root set. Otherwise, OtherInterface (or any other
    # interface in a library which depends on this one) will not be able
    # to find CallbackInterface when it is imported.
    idl_import_root = "src",
)

# MyParcelable.aidl is imported by MyInterface.aidl, so the generated
# MyInterface.java requires MyParcelable.class at compile time.
# Depending on :parcelable ensures that aidl compilation of MyInterface.aidl
# specifies the correct import roots and can access MyParcelable.aidl, and
# makes MyParcelable.class available to Java compilation of MyInterface.java
# as usual.
android_library(
    name = "idl",
    idl_srcs = ["MyInterface.aidl"],
    deps = [":parcelable"],
)

# Here, ServiceParcelable uses and thus depends on ParcelableService,
# when it's compiled, but ParcelableService also uses ServiceParcelable,
# which creates a circular dependency.
# As a result, these files must be compiled together, in the same android_library.
android_library(
    name = "circular_dependencies",
    srcs = ["ServiceParcelable.java"],
    idl_srcs = ["ParcelableService.aidl"],
    idl_parcelables = ["ServiceParcelable.aidl"],
)
</pre>

<!-- #END_BLAZE_RULE -->*/
