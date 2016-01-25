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
package com.google.devtools.build.lib.rules.android;

import static com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition.HOST;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static com.google.devtools.build.lib.packages.BuildType.TRISTATE;
import static com.google.devtools.build.lib.packages.ImplicitOutputsFunction.fromTemplates;
import static com.google.devtools.build.lib.syntax.Type.BOOLEAN;
import static com.google.devtools.build.lib.syntax.Type.INTEGER;
import static com.google.devtools.build.lib.syntax.Type.STRING;
import static com.google.devtools.build.lib.syntax.Type.STRING_LIST;
import static com.google.devtools.build.lib.util.FileTypeSet.ANY_FILE;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.Constants;
import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Attribute.LateBoundLabel;
import com.google.devtools.build.lib.packages.Attribute.SplitTransition;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction.SafeImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.packages.TriState;
import com.google.devtools.build.lib.rules.android.AndroidConfiguration.ConfigurationDistinguisher;
import com.google.devtools.build.lib.rules.cpp.CppOptions;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider;
import com.google.devtools.build.lib.rules.java.JavaConfiguration;
import com.google.devtools.build.lib.rules.java.JavaSemantics;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.FileType;

import java.util.ArrayList;
import java.util.List;

import javax.annotation.Nullable;

/**
 * Rule definitions for Android rules.
 */
public final class AndroidRuleClasses {
  public static final SafeImplicitOutputsFunction ANDROID_JAVA_SOURCE_JAR =
      fromTemplates("%{name}.srcjar");
  public static final SafeImplicitOutputsFunction ANDROID_LIBRARY_SOURCE_JAR =
      JavaSemantics.JAVA_LIBRARY_SOURCE_JAR;
  public static final SafeImplicitOutputsFunction ANDROID_LIBRARY_CLASS_JAR =
      JavaSemantics.JAVA_LIBRARY_CLASS_JAR;
  public static final SafeImplicitOutputsFunction ANDROID_LIBRARY_JACK_FILE =
      fromTemplates("lib%{name}.jack");
  public static final SafeImplicitOutputsFunction ANDROID_LIBRARY_AAR =
      fromTemplates("%{name}.aar");
  public static final SafeImplicitOutputsFunction ANDROID_LIBRARY_AAR_CLASSES_JAR =
      fromTemplates("%{name}_aar/classes.jar");
  public static final SafeImplicitOutputsFunction ANDROID_RESOURCES_SOURCE_JAR =
      fromTemplates("%{name}_resources-src.jar");
  public static final SafeImplicitOutputsFunction ANDROID_RESOURCES_CLASS_JAR =
      fromTemplates("%{name}_resources.jar");
  public static final SafeImplicitOutputsFunction ANDROID_RESOURCES_APK =
      fromTemplates("%{name}.ap_");
  public static final SafeImplicitOutputsFunction ANDROID_INCREMENTAL_RESOURCES_APK =
      fromTemplates("%{name}_files/incremental.ap_");
  public static final SafeImplicitOutputsFunction ANDROID_BINARY_APK =
      fromTemplates("%{name}.apk");
  public static final SafeImplicitOutputsFunction ANDROID_BINARY_INCREMENTAL_APK =
      fromTemplates("%{name}_incremental.apk");
  public static final SafeImplicitOutputsFunction ANDROID_BINARY_UNSIGNED_APK =
      fromTemplates("%{name}_unsigned.apk");
  public static final SafeImplicitOutputsFunction ANDROID_BINARY_SIGNED_APK =
      fromTemplates("%{name}_signed.apk");
  public static final SafeImplicitOutputsFunction ANDROID_BINARY_DEPLOY_JAR =
      fromTemplates("%{name}_deploy.jar");
  public static final SafeImplicitOutputsFunction ANDROID_BINARY_PROGUARD_JAR =
      fromTemplates("%{name}_proguard.jar");
  public static final SafeImplicitOutputsFunction ANDROID_BINARY_INSTRUMENTED_JAR =
      fromTemplates("%{name}_instrumented.jar");
  public static final SafeImplicitOutputsFunction ANDROID_TEST_FILTERED_JAR =
      fromTemplates("%{name}_filtered.jar");
  public static final SafeImplicitOutputsFunction ANDROID_R_TXT =
      fromTemplates("%{name}_symbols/R.txt");
  public static final SafeImplicitOutputsFunction ANDROID_SYMBOLS_TXT =
      fromTemplates("%{name}_symbols/local-R.txt");
  public static final SafeImplicitOutputsFunction STUB_APPLICATON_MANIFEST =
      fromTemplates("%{name}_files/stub/AndroidManifest.xml");
  public static final SafeImplicitOutputsFunction FULL_DEPLOY_MARKER =
      fromTemplates("%{name}_files/full_deploy_marker");
  public static final SafeImplicitOutputsFunction INCREMENTAL_DEPLOY_MARKER =
      fromTemplates("%{name}_files/incremental_deploy_marker");
  public static final SafeImplicitOutputsFunction SPLIT_DEPLOY_MARKER =
      fromTemplates("%{name}_files/split_deploy_marker");
  public static final SafeImplicitOutputsFunction MOBILE_INSTALL_ARGS =
      fromTemplates("%{name}_files/mobile_install_args");

  // This needs to be in its own directory because ApkBuilder only has a function (-rf) for source
  // folders but not source files, and it's easiest to guarantee that nothing gets put beside this
  // file in the ApkBuilder invocation in this manner
  public static final SafeImplicitOutputsFunction STUB_APPLICATION_DATA =
      fromTemplates("%{name}_files/stub_application_data/stub_application_data.txt");
  public static final SafeImplicitOutputsFunction DEX_MANIFEST =
      fromTemplates("%{name}_files/dexmanifest.txt");
  public static final SafeImplicitOutputsFunction JAVA_RESOURCES_JAR =
      fromTemplates("%{name}_files/java_resources.jar");
  public static final String MANIFEST_MERGE_TOOL_LABEL =
      Constants.TOOLS_REPOSITORY + "//tools/android:merge_manifests";
  public static final String BUILD_INCREMENTAL_DEXMANIFEST_LABEL =
      Constants.TOOLS_REPOSITORY + "//tools/android:build_incremental_dexmanifest";
  public static final String STUBIFY_MANIFEST_LABEL =
      Constants.TOOLS_REPOSITORY + "//tools/android:stubify_manifest";
  public static final String INCREMENTAL_INSTALL_LABEL =
      Constants.TOOLS_REPOSITORY + "//tools/android:incremental_install";
  public static final String BUILD_SPLIT_MANIFEST_LABEL =
      Constants.TOOLS_REPOSITORY + "//tools/android:build_split_manifest";
  public static final String STRIP_RESOURCES_LABEL =
      Constants.TOOLS_REPOSITORY + "//tools/android:strip_resources";

  public static final Label DEFAULT_ANDROID_SDK =
      Label.parseAbsoluteUnchecked(
          Constants.ANDROID_DEFAULT_SDK);
  public static final Label DEFAULT_INCREMENTAL_STUB_APPLICATION =
      Label.parseAbsoluteUnchecked(
          Constants.TOOLS_REPOSITORY + "//tools/android:incremental_stub_application");
  public static final Label DEFAULT_INCREMENTAL_SPLIT_STUB_APPLICATION =
      Label.parseAbsoluteUnchecked(
          Constants.TOOLS_REPOSITORY + "//tools/android:incremental_split_stub_application");
  public static final Label DEFAULT_RESOURCES_PROCESSOR =
      Label.parseAbsoluteUnchecked(
          Constants.TOOLS_REPOSITORY + "//tools/android:resources_processor");
  public static final Label DEFAULT_AAR_GENERATOR =
      Label.parseAbsoluteUnchecked(Constants.TOOLS_REPOSITORY + "//tools/android:aar_generator");

  public static final LateBoundLabel<BuildConfiguration> ANDROID_SDK =
      new LateBoundLabel<BuildConfiguration>(DEFAULT_ANDROID_SDK, AndroidConfiguration.class) {
        @Override
        public Label getDefault(Rule rule, BuildConfiguration configuration) {
          return configuration.getFragment(AndroidConfiguration.class).getSdk();
        }
      };

  public static final SplitTransition<BuildOptions> ANDROID_SPLIT_TRANSITION =
      new SplitTransition<BuildOptions>() {
        @Override
        public boolean defaultsToSelf() {
          return true;
        }

        private void setCrosstoolToAndroid(BuildOptions output, BuildOptions input) {
          AndroidConfiguration.Options inputAndroidOptions =
              input.get(AndroidConfiguration.Options.class);
          AndroidConfiguration.Options outputAndroidOptions =
              output.get(AndroidConfiguration.Options.class);

          CppOptions cppOptions = output.get(CppOptions.class);
          if (inputAndroidOptions.androidCrosstoolTop != null
              && !cppOptions.crosstoolTop.equals(inputAndroidOptions.androidCrosstoolTop)) {
            if (cppOptions.hostCrosstoolTop == null) {
              cppOptions.hostCrosstoolTop = cppOptions.crosstoolTop;
            }
            cppOptions.crosstoolTop = inputAndroidOptions.androidCrosstoolTop;
          }

          outputAndroidOptions.configurationDistinguisher = ConfigurationDistinguisher.ANDROID;
        }

        @Override
        public List<BuildOptions> split(BuildOptions buildOptions) {
          AndroidConfiguration.Options androidOptions =
              buildOptions.get(AndroidConfiguration.Options.class);
          CppOptions cppOptions = buildOptions.get(CppOptions.class);
          Label androidCrosstoolTop = androidOptions.androidCrosstoolTop;
          if (androidOptions.realFatApkCpus().isEmpty()
              && (androidCrosstoolTop == null
                  || androidCrosstoolTop.equals(cppOptions.crosstoolTop))) {
            return ImmutableList.of();
          }

          if (androidOptions.realFatApkCpus().isEmpty()) {
            BuildOptions splitOptions = buildOptions.clone();
            setCrosstoolToAndroid(splitOptions, buildOptions);
            return ImmutableList.of(splitOptions);
          }

          List<BuildOptions> result = new ArrayList<>();
          for (String cpu : ImmutableSortedSet.copyOf(androidOptions.realFatApkCpus())) {
            BuildOptions splitOptions = buildOptions.clone();
            // Disable fat APKs for the child configurations.
            splitOptions.get(AndroidConfiguration.Options.class).fatApkCpus = ImmutableList.of();

            // Set the cpu & android_cpu.
            // TODO(bazel-team): --android_cpu doesn't follow --cpu right now; it should.
            splitOptions.get(AndroidConfiguration.Options.class).cpu = cpu;
            splitOptions.get(BuildConfiguration.Options.class).cpu = cpu;
            splitOptions.get(CppOptions.class).cppCompiler = androidOptions.cppCompiler;
            setCrosstoolToAndroid(splitOptions, buildOptions);
            result.add(splitOptions);
          }
          return result;
        }
      };

  public static final FileType ANDROID_IDL = FileType.of(".aidl");

  public static final String[] ALLOWED_DEPENDENCIES = {
      "android_library",
      "cc_library",
      "java_import",
      "java_library",
      "proto_library"};

  public static final SafeImplicitOutputsFunction ANDROID_BINARY_IMPLICIT_OUTPUTS =
      new SafeImplicitOutputsFunction() {

        @Override
        public Iterable<String> getImplicitOutputs(AttributeMap rule) {
          boolean mapping = rule.get("proguard_generate_mapping", Type.BOOLEAN);
          List<SafeImplicitOutputsFunction> functions = Lists.newArrayList();
          functions.add(AndroidRuleClasses.ANDROID_BINARY_APK);
          functions.add(AndroidRuleClasses.ANDROID_BINARY_UNSIGNED_APK);
          functions.add(AndroidRuleClasses.ANDROID_BINARY_DEPLOY_JAR);

          // The below is a hack to support configurable attributes (proguard_specs seems like
          // too valuable an attribute to make nonconfigurable, and we don't currently
          // have the ability to know the configuration when determining implicit outputs).
          // An IllegalArgumentException gets triggered if the attribute instance is configurable.
          // We assume, heuristically, that means every configurable value is a non-empty list.
          //
          // TODO(bazel-team): find a stronger approach for this. One simple approach is to somehow
          // receive 'rule' as an AggregatingAttributeMapper instead of a RawAttributeMapper,
          // check that all possible values are non-empty, and simply don't support configurable
          // instances that mix empty and non-empty lists. A more ambitious approach would be
          // to somehow determine implicit outputs after the configuration is known. A third
          // approach is to refactor the Android rule logic to avoid these dependencies in the
          // first place.
          boolean hasProguardSpecs;
          try {
            hasProguardSpecs = !rule.get("proguard_specs", LABEL_LIST).isEmpty();
          } catch (IllegalArgumentException e) {
            // We assume at this point the attribute instance is configurable.
            hasProguardSpecs = true;
          }

          if (hasProguardSpecs) {
            functions.add(AndroidRuleClasses.ANDROID_BINARY_PROGUARD_JAR);
            if (mapping) {
              functions.add(JavaSemantics.JAVA_BINARY_PROGUARD_MAP);
            }
          }
          return fromFunctions(functions).getImplicitOutputs(rule);
        }
      };

  public static final SafeImplicitOutputsFunction ANDROID_LIBRARY_IMPLICIT_OUTPUTS =
      new SafeImplicitOutputsFunction() {
        @Override
        public Iterable<String> getImplicitOutputs(AttributeMap attributes) {

          ImmutableList.Builder<SafeImplicitOutputsFunction> implicitOutputs =
              ImmutableList.builder();

          implicitOutputs.add(
              AndroidRuleClasses.ANDROID_LIBRARY_CLASS_JAR,
              AndroidRuleClasses.ANDROID_LIBRARY_SOURCE_JAR,
              AndroidRuleClasses.ANDROID_LIBRARY_JACK_FILE,
              AndroidRuleClasses.ANDROID_LIBRARY_AAR);

          if (LocalResourceContainer.definesAndroidResources(attributes)) {
            implicitOutputs.add(
                AndroidRuleClasses.ANDROID_JAVA_SOURCE_JAR,
                AndroidRuleClasses.ANDROID_RESOURCES_APK,
                AndroidRuleClasses.ANDROID_R_TXT);
          }

          return fromFunctions(implicitOutputs.build()).getImplicitOutputs(attributes);
        }
      };

  /**
   * Definition of the {@code android_sdk} rule.
   */
  public static final class AndroidSdkRule implements RuleDefinition {
    @Override
    public RuleClass build(Builder builder, RuleDefinitionEnvironment environment) {
      return builder
          .requiresConfigurationFragments(JavaConfiguration.class, AndroidConfiguration.class)
          .setUndocumented()
          // This is the Proguard that comes from the --proguard_top attribute.
          .add(attr(":proguard", LABEL).cfg(HOST).value(JavaSemantics.PROGUARD).exec())
          // This is the Proguard in the BUILD file that contains the android_sdk rule. Used when
          // --proguard_top is not specified.
          .add(attr("proguard", LABEL).mandatory().cfg(HOST).allowedFileTypes(ANY_FILE).exec())
          .add(attr("aapt", LABEL).mandatory().cfg(HOST).allowedFileTypes(ANY_FILE).exec())
          .add(attr("dx", LABEL).mandatory().cfg(HOST).allowedFileTypes(ANY_FILE).exec())
          .add(attr("main_dex_list_creator", LABEL)
              .mandatory().cfg(HOST).allowedFileTypes(ANY_FILE).exec())
          .add(attr("adb", LABEL).mandatory().cfg(HOST).allowedFileTypes(ANY_FILE).exec())
          .add(attr("framework_aidl", LABEL).mandatory().cfg(HOST).allowedFileTypes(ANY_FILE))
          .add(attr("aidl", LABEL).mandatory().cfg(HOST).allowedFileTypes(ANY_FILE).exec())
          .add(attr("android_jar", LABEL).mandatory().cfg(HOST).allowedFileTypes(ANY_FILE))
          .add(attr("shrinked_android_jar", LABEL).mandatory().cfg(HOST).allowedFileTypes(ANY_FILE))
          .add(
              attr("android_jack", LABEL)
                  .cfg(HOST)
                  .allowedFileTypes(ANY_FILE)
                  // TODO(bazel-team): Remove defaults and make mandatory when android_sdk targets
                  // have been updated to include manually specified Jack attributes.
                  .value(environment.getLabel(
                      Constants.TOOLS_REPOSITORY + "//tools/android/jack:android_jack")))
          .add(attr("annotations_jar", LABEL).mandatory().cfg(HOST).allowedFileTypes(ANY_FILE))
          .add(attr("main_dex_classes", LABEL).mandatory().cfg(HOST).allowedFileTypes(ANY_FILE))
          .add(attr("apkbuilder", LABEL).mandatory().cfg(HOST).allowedFileTypes(ANY_FILE).exec())
          .add(attr("zipalign", LABEL).mandatory().cfg(HOST).allowedFileTypes(ANY_FILE).exec())
          .add(
              attr("jack", LABEL)
                  .cfg(HOST)
                  .allowedFileTypes(ANY_FILE)
                  .exec()
                  .value(environment.getLabel(
                      Constants.TOOLS_REPOSITORY + "//tools/android/jack:jack")))
          .add(
              attr("jill", LABEL)
                  .cfg(HOST)
                  .allowedFileTypes(ANY_FILE)
                  .exec()
                  .value(environment.getLabel(
                      Constants.TOOLS_REPOSITORY + "//tools/android/jack:jill")))
          .add(
              attr("resource_extractor", LABEL)
                  .cfg(HOST)
                  .allowedFileTypes(ANY_FILE)
                  .exec()
                  .value(environment.getLabel(
                      Constants.TOOLS_REPOSITORY + "//tools/android/jack:resource_extractor")))
          .build();
    }

    @Override
    public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("android_sdk")
        .ancestors(BaseRuleClasses.BaseRule.class)
        .factoryClass(AndroidSdk.class)
        .build();
    }
  }

  /**
   * Base class for rule definitions using AAPT.
   */
  public static final class AndroidAaptBaseRule implements RuleDefinition {
    @Override
    public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
      return builder
          .add(attr("$android_resources_processor", LABEL).cfg(HOST).exec().value(
              AndroidRuleClasses.DEFAULT_RESOURCES_PROCESSOR))
          .add(attr("$android_aar_generator", LABEL).cfg(HOST).exec().value(
              AndroidRuleClasses.DEFAULT_AAR_GENERATOR))
          .build();
    }

    @Override
    public Metadata getMetadata() {
      return Metadata.builder()
          .name("$android_aapt_base")
          .type(RuleClassType.ABSTRACT)
          .ancestors(AndroidRuleClasses.AndroidBaseRule.class)
          .build();
    }
  }

  /**
   * Base class for rule definitions that support resource declarations.
   */
  public static final class AndroidResourceSupportRule implements RuleDefinition {
    @Override
    public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
      return builder.setUndocumented()
          /* <!-- #BLAZE_RULE($android_resource_support).ATTRIBUTE(manifest) -->
          The name of the Android manifest file, normally <code>AndroidManifest.xml</code>.
          Must be defined if resource_files or assets are defined.
          ${SYNOPSIS}
          <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
          .add(attr("manifest", LABEL).legacyAllowAnyFileType())
          /* <!-- #BLAZE_RULE($android_resource_support).ATTRIBUTE(exports_manifest) -->
          Whether to export manifest entries to <code>android_binary</code> targets
          that depend on this target. <code>uses-permissions</code> attributes are never exported.
          ${SYNOPSIS}
          <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
          .add(attr("exports_manifest", BOOLEAN).value(false))
          /* <!-- #BLAZE_RULE($android_resource_support).ATTRIBUTE(resource_files) -->
          The list of resources to be packaged.
          ${SYNOPSIS}
          This is typically a <code>glob</code> of all files under the
          <code>res</code> directory.
          <br/>
          Generated files (from genrules) can be referenced by
          <a href="../build-ref.html#labels">Label</a> here as well. The only restriction is that
          the generated outputs must be under the same "<code>res</code>" directory as any other
          resource files that are included.
          <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
          .add(attr("resource_files", LABEL_LIST).legacyAllowAnyFileType())
          /* <!-- #BLAZE_RULE($android_resource_support).ATTRIBUTE(assets_dir) -->
          The string giving the path to the files in <code>assets</code>.
          ${SYNOPSIS}
          The pair <code>assets</code> and <code>assets_dir</code> describe packaged
          assets and either both attributes should be provided or none of them.
          <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
          .add(attr("assets_dir", STRING))
          /* <!-- #BLAZE_RULE($android_resource_support).ATTRIBUTE(assets) -->
          The list of assets to be packaged.
          ${SYNOPSIS}
          This is typically a <code>glob</code> of all files under the
          <code>assets</code> directory. You can also reference other rules (any rule that produces
          files) or exported files in the other packages, as long as all those files are under the
          <code>assets_dir</code> directory in the corresponding package.
          <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
          .add(attr("assets", LABEL_LIST).legacyAllowAnyFileType())
          /* <!-- #BLAZE_RULE($android_resource_support).ATTRIBUTE(inline_constants) -->
          Let the compiler inline the constants defined in the generated java sources.
          ${SYNOPSIS}
          This attribute must be set to 0 for all <code>android_library</code> rules
          used directly by an <code>android_binary</code>,
          and for any <code>android_binary</code> that has an <code>android_library</code>
          in its transitive closure.
          <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
          .add(attr("inline_constants", BOOLEAN).value(false))
          /* <!-- #BLAZE_RULE($android_resource_support).ATTRIBUTE(custom_package) -->
          Java package for which java sources will be generated.
          ${SYNOPSIS}
          By default the package is inferred from the directory where the BUILD file
          containing the rule is. You can specify a different package but this is
          highly discouraged since it can introduce classpath conflicts with other
          libraries that will only be detected at runtime.
          <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
          .add(attr("custom_package", STRING))
        .build();
    }

    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("$android_resource_support")
          .type(RuleClassType.ABSTRACT)
          .ancestors(AndroidRuleClasses.AndroidBaseRule.class)
          .build();
    }
  }

  /**
   * Base class for Android rule definitions.
   */
  public static final class AndroidBaseRule implements RuleDefinition {
    @Override
    public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
      return builder
          .add(attr(":android_sdk", LABEL)
              .allowedRuleClasses("android_sdk", "filegroup")
              .value(ANDROID_SDK))
          /* <!-- #BLAZE_RULE($android_base).ATTRIBUTE(plugins) -->
          Java compiler plugins to run at compile-time.
          ${SYNOPSIS}
          Every <code>java_plugin</code> specified in
          the plugins attribute will be run whenever
          this target is built.  Resources generated by
          the plugin will be included in the result jar of
          the target.
          <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
          .add(attr("plugins", LABEL_LIST).cfg(HOST).allowedRuleClasses("java_plugin")
              .legacyAllowAnyFileType())
          .add(attr(":java_plugins", LABEL_LIST)
              .cfg(HOST)
              .allowedRuleClasses("java_plugin")
              .silentRuleClassFilter()
              .value(JavaSemantics.JAVA_PLUGINS))
          /* <!-- #BLAZE_RULE($android_base).ATTRIBUTE(javacopts) -->
          Extra compiler options for this target.
          ${SYNOPSIS}
          Subject to <a href="#make_variables">"Make variable"</a> substitution and
          <a href="common-definitions.html#sh-tokenization">Bourne shell tokenization</a>.
          <p>
          These compiler options are passed to javac after the global compiler options.</p>
          <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
          .add(attr("javacopts", STRING_LIST))
          // TODO(ahumesky): It would be better to put this dependency in //tools/android somehow
          // like all the rest of android tools.
          .add(attr("$jarjar_bin", LABEL).cfg(HOST).exec()
              .value(env.getLabel(
                  Constants.TOOLS_REPOSITORY + "//third_party/java/jarjar:jarjar_bin")))
          .add(attr("$idlclass", LABEL).cfg(HOST).exec()
              .value(env.getLabel(Constants.TOOLS_REPOSITORY + "//tools/android:IdlClass")))
          .build();
    }

    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("$android_base")
          .type(RuleClassType.ABSTRACT)
          .ancestors(BaseRuleClasses.RuleBase.class)
          .build();
    }
  }

  /**
   * Base class for Android rule definitions that produce binaries.
   */
  public static final class AndroidBinaryBaseRule implements RuleDefinition {
    @Override
    public RuleClass build(RuleClass.Builder builder, final RuleDefinitionEnvironment env) {
      return builder
          /* <!-- #BLAZE_RULE($android_binary_base).ATTRIBUTE(srcs) -->
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
          <p>This rule currently forces source and class compatibility with Java 6.
          </p>
          <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
          .add(attr("srcs", LABEL_LIST)
              .direct_compile_time_input()
              .allowedFileTypes(JavaSemantics.JAVA_SOURCE, JavaSemantics.SOURCE_JAR))
          /* <!-- #BLAZE_RULE($android_binary_base).ATTRIBUTE(deps) -->
          The list of other libraries to be linked in to the binary target.
          ${SYNOPSIS}
          Permitted library types are: <code>android_library</code>,
          <code>java_library</code> with <code>android</code> constraint and
          <code>cc_library</code> wrapping or producing <code>.so</code> native libraries for the
          Android target platform.
          <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
          .override(builder.copy("deps")
              .cfg(ANDROID_SPLIT_TRANSITION)
              .allowedRuleClasses(ALLOWED_DEPENDENCIES)
              .allowedFileTypes()
              .aspect(AndroidNeverlinkAspect.class))
          // Proguard rule specifying master list of classes to keep during legacy multidexing.
          .add(attr("$build_incremental_dexmanifest", LABEL).cfg(HOST).exec()
              .value(env.getLabel(AndroidRuleClasses.BUILD_INCREMENTAL_DEXMANIFEST_LABEL)))
          .add(attr("$stubify_manifest", LABEL).cfg(HOST).exec()
              .value(env.getLabel(AndroidRuleClasses.STUBIFY_MANIFEST_LABEL)))
          .add(attr("$shuffle_jars", LABEL).cfg(HOST).exec()
              .value(env.getLabel(Constants.TOOLS_REPOSITORY + "//tools/android:shuffle_jars")))
          .add(attr("$merge_dexzips", LABEL).cfg(HOST).exec()
              .value(env.getLabel(Constants.TOOLS_REPOSITORY + "//tools/android:merge_dexzips")))
          .add(attr("$incremental_install", LABEL).cfg(HOST).exec()
              .value(env.getLabel(INCREMENTAL_INSTALL_LABEL)))
          .add(attr("$build_split_manifest", LABEL).cfg(HOST).exec()
              .value(env.getLabel(BUILD_SPLIT_MANIFEST_LABEL)))
          .add(attr("$strip_resources", LABEL).cfg(HOST).exec()
              .value(env.getLabel(AndroidRuleClasses.STRIP_RESOURCES_LABEL)))
          .add(attr("$incremental_stub_application", LABEL)
              .value(DEFAULT_INCREMENTAL_STUB_APPLICATION))
          .add(attr("$incremental_split_stub_application", LABEL)
              .value(DEFAULT_INCREMENTAL_SPLIT_STUB_APPLICATION))
          /* <!-- #BLAZE_RULE($android_binary_base).ATTRIBUTE(dexopts) -->
          Additional command-line flags for the dx tool when generating classes.dex.
          ${SYNOPSIS}
          Subject to <a href="#make_variables">"Make variable"</a> substitution and
          <a href="common-definitions.html#sh-tokenization">Bourne shell tokenization</a>.
          <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
          .add(attr("dexopts", STRING_LIST))
          /* <!-- #BLAZE_RULE($android_binary_base).ATTRIBUTE(dex_shards) -->
          Number of shards dexing should be decomposed into.
          ${SYNOPSIS}
          This is makes dexing much faster at the expense of app installation and startup time. The
          larger the binary, the more shards should be used. 25 is a good value to start
          experimenting with.
          <p>
          Note that each shard will result in at least one dex in the final app. For this reason,
          setting this to more than 1 is not recommended for release binaries.
          <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
          .add(attr("dex_shards", INTEGER).value(1))
          /* <!-- #BLAZE_RULE($android_binary_base).ATTRIBUTE(main_dex_list_opts) -->
          Command line options to pass to the main dex list builder.
          ${SYNOPSIS}
          Use this option to affect the classes included in the main dex list.
          <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
          .add(attr("main_dex_list_opts", STRING_LIST))
          /* <!-- #BLAZE_RULE($android_binary_base).ATTRIBUTE(main_dex_list) -->

          A text file contains a list of class file names. Classes defined by those class files are
          put in the primary classes.dex. e.g.:<pre class="code">
android/support/multidex/MultiDex$V19.class
android/support/multidex/MultiDex.class
android/support/multidex/MultiDexApplication.class
com/google/common/base/Objects.class
          </pre>
          ${SYNOPSIS}
          Must be used with <code>multidex="manual_main_dex"</code>.
          <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
          .add(attr("main_dex_list", LABEL).legacyAllowAnyFileType())
          /* <!-- #BLAZE_RULE($android_binary_base).ATTRIBUTE(main_dex_proguard_specs) -->
          Files to be used as the Proguard specifications to determine classes that must be kept in
          the main dex.
          ${SYNOPSIS}
          Only allowed if the <code>multidex</code> attribute is set to <code>legacy</code>.
          <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
          .add(attr("main_dex_proguard_specs", LABEL_LIST).legacyAllowAnyFileType())
          /* <!-- #BLAZE_RULE($android_binary_base).ATTRIBUTE(proguard_specs) -->
          Files to be used as Proguard specification.
          ${SYNOPSIS}
          This file will describe the set of specifications to be used by Proguard.
          <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
          .add(attr("proguard_specs", LABEL_LIST).legacyAllowAnyFileType())
          /* <!-- #BLAZE_RULE($android_binary_base).ATTRIBUTE(proguard_generate_mapping) -->
          Whether to generate Proguard mapping file.
          ${SYNOPSIS}
          The mapping file will be generated only if <code>proguard_specs</code> is
          specified. This file will list the mapping between the original and
          obfuscated class, method, and field names.
          <p><em class="harmful">WARNING: If you use this attribute, your Proguard specification
          should contain neither <code>-dontobfuscate</code> nor <code>-printmapping</code>.
          </em>.</p>
          <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
          .add(attr("proguard_generate_mapping", BOOLEAN).value(false)
              .nonconfigurable("value is referenced in an ImplicitOutputsFunction"))
          /* <!-- #BLAZE_RULE($android_binary_base).ATTRIBUTE(proguard_apply_mapping) -->
          File to be used as a mapping for proguard.
          ${SYNOPSIS}
          A mapping file generated by <code>proguard_generate_mapping</code> to be
          re-used to apply the same mapping to a new build.
          <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
          .add(attr("proguard_apply_mapping", LABEL).legacyAllowAnyFileType())
          /* <!-- #BLAZE_RULE($android_binary_base).ATTRIBUTE(legacy_native_support) -->
          Enables legacy native support, where pre-compiled native libraries are copied
          directly into the APK.
          ${SYNOPSIS}
          Possible values:
          <ul>
              <li><code>legacy_native_support = 1</code>: Pre-built .so files found in the
                dependencies of cc_libraries in the transitive closure will be copied into
                the APK without being modified in any way. All cc_libraries in the transitive
                closure of this rule must wrap .so files. (<em class="harmful">deprecated</em> -
                legacy_native_support = 0 will become the default and this attribute will be
                removed in a future Blaze release.)</li>
              <li><code>legacy_native_support = 0</code>: Native dependencies in the transitive
                closure will be linked together into a single lib[ruleName].so
                before being placed in the APK. This ensures that, e.g., only one copy of
                //base will be loaded into memory. This lib[ruleName].so can be loaded
                via System.loadLibrary as normal.</li>
              <li><code>legacy_native_support = -1</code>: Linking is controlled by the
                <a href="blaze-user-manual.html#flag--legacy_android_native_support">
                --[no]legacy_android_native_support</a> Blaze flag.</li>
            </ul>
          <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
          .add(attr("legacy_native_support", TRISTATE).value(TriState.AUTO))
          .add(attr(":extra_proguard_specs", LABEL_LIST).value(JavaSemantics.EXTRA_PROGUARD_SPECS))
          .advertiseProvider(JavaCompilationArgsProvider.class)
          .build();
      }

      @Override
      public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("$android_binary_base")
          .type(RuleClassType.ABSTRACT)
          .ancestors(
              AndroidRuleClasses.AndroidBaseRule.class,
              AndroidAaptBaseRule.class,
              AndroidResourceSupportRule.class)
          .build();
      }
  }

  /**
   * Semantic options for the dexer's multidex behavior.
   */
  public static enum MultidexMode {
    // Build dexes with multidex, assuming native platform support for multidex.
    NATIVE("native"),
    // Build dexes with multidex and implement support at the application level.
    LEGACY("legacy"),
    // Build dexes with multidex, main dex list needs to be manually specified.
    MANUAL_MAIN_DEX("legacy"),
    // Build all dex code into a single classes.dex file.
    OFF("none");

    @Nullable private final String jackFlagValue;

    private MultidexMode(String jackFlagValue) {
      this.jackFlagValue = jackFlagValue;
    }

    /**
     * Returns the attribute value that specifies this mode.
     */
    public String getAttributeValue() {
      return toString().toLowerCase();
    }

    /**
     * Returns whether or not this multidex mode can be passed to Jack.
     */
    public boolean isSupportedByJack() {
      return jackFlagValue != null;
    }

    /**
     * Returns the value that should be passed to Jack's --multi-dex flag.
     *
     * @throws UnsupportedOperationException if the dex mode is not supported by Jack
     *     ({@link #isSupportedByJack()} returns false)
     */
    public String getJackFlagValue() {
      if (!isSupportedByJack()) {
        throw new UnsupportedOperationException();
      }
      return jackFlagValue;
    }

    /**
     * Returns the name of the output dex classes file. In multidex mode, this is an archive
     * of (possibly) multiple files.
     */
    public String getOutputDexFilename() {
      return this == OFF ? "classes.dex" : "classes.dex.zip";
    }

    /**
     * Converts an attribute value to a corresponding mode. Returns null on no match.
     */
    public static MultidexMode fromValue(String value) {
      for (MultidexMode mode : values()) {
        if (mode.getAttributeValue().equals(value)) {
          return mode;
        }
      }
      return null;
    }

    /**
     * Enumerates valid values for the "multidex" attribute.
     */
    public static List<String> getValidValues() {
      List<String> ans = Lists.newArrayList();
      for (MultidexMode mode : MultidexMode.values()) {
        ans.add(mode.getAttributeValue());
      }
      return ans;
    }
  }
}
