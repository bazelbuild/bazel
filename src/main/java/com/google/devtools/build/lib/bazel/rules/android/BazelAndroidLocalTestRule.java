// Copyright 2017 The Bazel Authors. All rights reserved.
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

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static com.google.devtools.build.lib.packages.BuildType.TRISTATE;
import static com.google.devtools.build.lib.packages.ImplicitOutputsFunction.fromFunctions;

import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.bazel.rules.java.BazelJavaRuleClasses.BaseJavaBinaryRule;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.packages.RuleClass.ToolchainTransitionMode;
import com.google.devtools.build.lib.packages.StarlarkProviderIdentifier;
import com.google.devtools.build.lib.packages.TriState;
import com.google.devtools.build.lib.rules.android.AndroidFeatureFlagSetProvider;
import com.google.devtools.build.lib.rules.android.AndroidLocalTestBaseRule;
import com.google.devtools.build.lib.rules.config.ConfigFeatureFlagTransitionFactory;
import com.google.devtools.build.lib.rules.java.JavaConfiguration;
import com.google.devtools.build.lib.rules.java.JavaInfo;
import com.google.devtools.build.lib.rules.java.JavaSemantics;

/** Rule definition for Bazel android_local_test */
public class BazelAndroidLocalTestRule implements RuleDefinition {

  protected static final String JUNIT_TESTRUNNER = "//tools/jdk:TestRunner_deploy.jar";

  private static final ImmutableCollection<String> ALLOWED_RULES_IN_DEPS =
      ImmutableSet.of(
          "aar_import",
          "android_library",
          "java_import",
          "java_library",
          "java_lite_proto_library");

  static final ImplicitOutputsFunction ANDROID_ROBOLECTRIC_IMPLICIT_OUTPUTS = fromFunctions(
      JavaSemantics.JAVA_BINARY_CLASS_JAR,
      JavaSemantics.JAVA_BINARY_SOURCE_JAR,
      JavaSemantics.JAVA_BINARY_DEPLOY_JAR);

  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment environment) {
    return builder
        .requiresConfigurationFragments(JavaConfiguration.class)
        .setImplicitOutputsFunction(ANDROID_ROBOLECTRIC_IMPLICIT_OUTPUTS)
        .override(
            attr("deps", LABEL_LIST)
                .allowedFileTypes()
                .allowedRuleClasses(ALLOWED_RULES_IN_DEPS)
                .mandatoryProvidersList(
                    ImmutableList.of(
                        ImmutableList.of(
                            StarlarkProviderIdentifier.forKey(JavaInfo.PROVIDER.getKey())))))
        .override(attr("$testsupport", LABEL).value(environment.getToolsLabel(JUNIT_TESTRUNNER)))
        .add(
            attr("$robolectric_implicit_classpath", LABEL_LIST)
                .value(ImmutableList.of(environment.getToolsLabel("//tools/android:android_jar"))))
        .override(attr("stamp", TRISTATE).value(TriState.NO))
        .removeAttribute("classpath_resources")
        .removeAttribute("create_executable")
        .removeAttribute("deploy_manifest_lines")
        .removeAttribute("distribs")
        .removeAttribute("launcher")
        .removeAttribute("main_class")
        .removeAttribute("resources")
        .removeAttribute("use_testrunner")
        .removeAttribute(":java_launcher")
        .cfg(
            new ConfigFeatureFlagTransitionFactory(AndroidFeatureFlagSetProvider.FEATURE_FLAG_ATTR))
        .useToolchainTransition(ToolchainTransitionMode.ENABLED)
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("android_local_test")
        .type(RuleClassType.TEST)
        .ancestors(
            AndroidLocalTestBaseRule.class,
            BaseJavaBinaryRule.class,
            BaseRuleClasses.TestBaseRule.class)
        .factoryClass(BazelAndroidLocalTest.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = android_local_test, TYPE = TEST, FAMILY = Android) -->

<p>
This rule is for unit testing <code>android_library</code> rules locally
(as opposed to on a device).
It works with the Android Robolectric testing framework.
See the <a href="http://robolectric.org/">Android Robolectric</a> site for details about
writing Robolectric tests.
</p>

${IMPLICIT_OUTPUTS}

<h4 id="android_local_test_examples">Examples</h4>

<p>
To use Robolectric with <code>android_local_test</code>, add
<a href="https://github.com/robolectric/robolectric-bazel/tree/master/bazel">Robolectric's
repository</a> to your <code>WORKSPACE</code> file:
<pre class="code">
http_archive(
    name = "robolectric",
    urls = ["https://github.com/robolectric/robolectric/archive/&lt;COMMIT&gt;.tar.gz"],
    strip_prefix = "robolectric-&lt;COMMIT&gt;",
    sha256 = "&lt;HASH&gt;",
)
load("@robolectric//bazel:robolectric.bzl", "robolectric_repositories")
robolectric_repositories()
</pre>

This pulls in the <code>maven_jar</code> rules needed for Robolectric.

Then each <code>android_local_test</code> rule should depend on
<code>@robolectric//bazel:robolectric</code>. See example below.

</p>

<pre class="code">
android_local_test(
    name = "SampleTest",
    srcs = [
        "SampleTest.java",
    ],
    manifest = "LibManifest.xml",
    deps = [
        ":sample_test_lib",
        "@robolectric//bazel:robolectric",
    ],
)

android_library(
    name = "sample_test_lib",
    srcs = [
         "Lib.java",
    ],
    resource_files = glob(["res/**"]),
    manifest = "AndroidManifest.xml",
)
</pre>

<!-- #END_BLAZE_RULE --> */

/* <!-- #BLAZE_RULE(android_local_test).IMPLICIT_OUTPUTS -->
<ul>
  <li><code><var>name</var>.jar</code>: A Java archive of the test.</li>
  <li><code><var>name</var>-src.jar</code>: An archive containing the sources
    ("source jar").</li>
  <li><code><var>name</var>_deploy.jar</code>: A Java deploy archive suitable
    for deployment (only built if explicitly requested).</li>
</ul>
<!-- #END_BLAZE_RULE.IMPLICIT_OUTPUTS --> */
