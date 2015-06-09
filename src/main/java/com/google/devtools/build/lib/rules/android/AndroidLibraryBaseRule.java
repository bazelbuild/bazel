// Copyright 2015 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.rules.android;

import static com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition.HOST;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.Type.BOOLEAN;
import static com.google.devtools.build.lib.packages.Type.LABEL;
import static com.google.devtools.build.lib.packages.Type.LABEL_LIST;
import static com.google.devtools.build.lib.packages.Type.STRING;

import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.rules.android.AndroidRuleClasses.AndroidAaptBaseRule;
import com.google.devtools.build.lib.rules.android.AndroidRuleClasses.AndroidBaseRule;
import com.google.devtools.build.lib.rules.android.AndroidRuleClasses.AndroidResourceSupportRule;
import com.google.devtools.build.lib.rules.android.AndroidRuleClasses.JackRule;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider;
import com.google.devtools.build.lib.rules.java.JavaSemantics;

/**
 * Rule definition for the android_library rule.
 */
public final class AndroidLibraryBaseRule implements RuleDefinition {
  @Override
  public RuleClass build(RuleClass.Builder builder, final RuleDefinitionEnvironment env) {
    return builder
        /* <!-- #BLAZE_RULE(android_library).ATTRIBUTE(srcs) -->
         The list of source files that are processed to create the target.
        ${SYNOPSIS}
        <p><code>srcs</code> files of type <code>.java</code> are compiled.
        <em>For readability's sake</em>, it is not good to put the name of a
        generated <code>.java</code> source file into the <code>srcs</code>.
        Instead, put the depended-on rule name in the <code>srcs</code>, as
        described below.
        </p>
        <p><code>srcs</code> files of type <code>.srcjar</code> are unpacked and
        compiled. (This is useful if you need to generate a set of .java files with
        a genrule or build extension.)
        </p>
        <p>This rule currently forces source and class compatibility with Java 7,
        although try with resources is not supported.
        </p>
        <p><code>srcs</code> files of type <code>.jar</code> are linked in.
        (This is useful if you have third-party <code>.jar</code> files
        with no source.)
        </p>
        <p>If <code>srcs</code> is omitted, then any dependency specified in
        <code>deps</code> is exported from this rule (see
        <a href="#java_library.exports">java_library's exports</a> for more
        information about exporting dependencies). However, this behavior will be
        deprecated soon; try not to rely on it.
        </p>
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("srcs", LABEL_LIST)
            .direct_compile_time_input()
            .allowedFileTypes(JavaSemantics.JAVA_SOURCE, JavaSemantics.JAR,
                JavaSemantics.SOURCE_JAR))
        /* <!-- #BLAZE_RULE(android_library).ATTRIBUTE(deps) -->
        The list of other libraries to link against.
        ${SYNOPSIS}
        Permitted library types are: <code>android_library</code>,
        <code>java_library</code> with <code>android</code> constraint and
        <code>cc_library</code> wrapping or producing <code>.so</code> native libraries
        for the Android target platform.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .override(builder.copy("deps")
            .allowedRuleClasses(AndroidRuleClasses.ALLOWED_DEPENDENCIES)
            .allowedFileTypes())
        /* <!-- #BLAZE_RULE(android_library).ATTRIBUTE(resources) -->
        The <code>android_resources</code> target assigned to this library.
        ${SYNOPSIS}
        If specified, the resources will be added to any <code>android_binary</code>
        depending on this library.
        <p>Only an <code>android_resource</code> rule with the attribute
        <code>inline_constants</code> set to 0 can be used in this case.</p>
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("resources", LABEL)
            .allowedFileTypes()
            .allowedRuleClasses("android_resources"))
        .add(attr("alwayslink", BOOLEAN).undocumented("purely informational for now"))
        /* <!-- #BLAZE_RULE(android_library).ATTRIBUTE(neverlink) -->
        Only use this library for compilation and not at runtime.
        ${SYNOPSIS}
        The outputs of a rule marked as <code>neverlink</code> will not be used in
        <code>.apk</code> creation. Useful if the library will be provided by the
        runtime environment during execution.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("neverlink", BOOLEAN).value(false))
        /* <!-- #BLAZE_RULE(android_library).ATTRIBUTE(idl_import_root) -->
        Package-relative path to the root of the java package tree containing idl
        sources included in this library.
        ${SYNOPSIS}
        This path will be used as the import root when processing idl sources that
        depend on this library. (See
        <a href="#android_library_examples.idl_import_root">examples</a>.)
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("idl_import_root", STRING))
        /* <!-- #BLAZE_RULE(android_library).ATTRIBUTE(idl_srcs) -->
        List of Android IDL definitions to translate to Java interfaces.
        ${SYNOPSIS}
        After the Java interfaces are generated, they will be compiled together
        with the contents of <code>srcs</code>.
        <p>These files will be made available as imports for any
        <code>android_library</code> target that depends on this library, directly
        or via its transitive closure.</p>
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("idl_srcs", LABEL_LIST).direct_compile_time_input()
            .allowedFileTypes(AndroidRuleClasses.ANDROID_IDL))
        /* <!-- #BLAZE_RULE(android_library).ATTRIBUTE(idl_parcelables) -->
        List of Android IDL definitions to supply as imports.
        ${SYNOPSIS}
        These files will be made available as imports for any
        <code>android_library</code> target that depends on this library, directly
        or via its transitive closure, but will not be translated to Java
        or compiled.
        <p>Only <code>.aidl</code> files that correspond directly to
        <code>.java</code> sources in this library should be included (e.g., custom
        implementations of Parcelable), otherwise <code>idl_srcs</code> should be
        used.</p>
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("idl_parcelables", LABEL_LIST).direct_compile_time_input()
            .allowedFileTypes(AndroidRuleClasses.ANDROID_IDL))
        /* <!-- #BLAZE_RULE(android_library).ATTRIBUTE(proguard_specs) -->
        Files to be used as Proguard specification.
        ${SYNOPSIS}
        These will describe the set of specifications to be used by Proguard. If specified,
        they will be added to any <code>android_binary</code> target depending on this library.

        The files included here must only have idempotent rules, namely -dontnote, -dontwarn,
        assumenosideeffects, and rules that start with -keep. Other options can only appear in
        <code>android_binary</code>'s proguard_specs, to ensure non-tautological merges.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("proguard_specs", LABEL_LIST).legacyAllowAnyFileType())
        .add(attr("$proguard_whitelister", LABEL).cfg(HOST).exec().value(
            new Attribute.ComputedDefault() {
            @Override
            public Object getDefault(AttributeMap rule) {
              return rule.isAttributeValueExplicitlySpecified("proguard_specs")
                  ? env.getLabel("//tools/android:proguard_whitelister")
                  : null;
            }
        }))
        .add(attr("$android_manifest_merge_tool", LABEL).cfg(HOST).exec().value(env.getLabel(
            AndroidRuleClasses.MANIFEST_MERGE_TOOL_LABEL)))
        .advertiseProvider(JavaCompilationArgsProvider.class)
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("$android_library_base")
        .type(RuleClassType.ABSTRACT)
        .ancestors(
            AndroidBaseRule.class,
            AndroidAaptBaseRule.class,
            AndroidResourceSupportRule.class,
            JackRule.class)
        .build();
  }
}
