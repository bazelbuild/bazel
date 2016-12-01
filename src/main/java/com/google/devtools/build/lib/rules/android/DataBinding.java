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
package com.google.devtools.build.lib.rules.android;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction.Substitution;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction.Template;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.rules.java.JavaPluginInfoProvider;
import com.google.devtools.build.lib.rules.java.JavaTargetAttributes;
import com.google.devtools.build.lib.syntax.Type;
import java.util.ArrayList;
import java.util.List;

/**
 * Support logic for Bazel's
 * <a href="https://developer.android.com/topic/libraries/data-binding/index.html">data binding</a>
 * integration.
 *
 * <p>In short, data binding in Bazel works as follows:
 * <ol>
 *   <li>If a rule enables data binding and has layout resources with data binding expressions,
 *     resource processing invokes the data binding library to preprocess these expressions, then
 *     strips them out before feeding the resources into aapt. A separate "layout info" XML file
 *     gets produced that contains the bindings.</li>
 *   <li>The data binding annotation processor gets activated on Java compilation. This processor
 *     reads a custom-generated <code>DataBindingInfo.java</code> which specifies the path to the
 *     layout info file (as an annotation). The processor reads that file and produces the
 *     corresponding Java classes that end-user code uses to access the resources.</li>
 *   <li>The data binding compile-time and runtime support libraries get linked into the binary's
 *     deploy jar.</li>
 * </ol>
 *
 * <p>For data binding to work, the corresponding support libraries must be checked into the depot
 * via the implicit dependencies specified inside this class.
 *
 * <p>Unless otherwise specified, all methods in this class assume the current rule applies data
 * binding. Callers can intelligently trigger this logic by checking {@link #isEnabled}.
 *
 */
public final class DataBinding {
  /**
   * The rule attribute supplying the data binding runtime/compile-time support libraries.
   */
  private static final String DATABINDING_RUNTIME_ATTR = "$databinding_runtime";

  /**
   * The rule attribute supplying the data binding annotation processor.
   */
  private static final String DATABINDING_ANNOTATION_PROCESSOR_ATTR =
      "$databinding_annotation_processor";

  /**
   * Should data binding support be enabled for this rule?
   *
   * <p>This is true if either the rule or any of its transitive dependencies declares data binding
   * support in its attributes.
   *
   * <p>Data binding incurs additional resource processing and compilation work as well as
   * additional compile/runtime dependencies. But rules with data binding disabled will fail if
   * any data binding expressions appear in their layout resources.
   */
  public static boolean isEnabled(RuleContext ruleContext) {
    if (ruleContext.attributes().has("enable_data_binding", Type.BOOLEAN)
        && ruleContext.attributes().get("enable_data_binding", Type.BOOLEAN)) {
      return true;
    } else {
      return !Iterables.isEmpty(ruleContext.getPrerequisites("deps",
          RuleConfiguredTarget.Mode.TARGET, UsesDataBindingProvider.class));
    }
  }

  /**
   * Returns the file where data binding's resource processing produces binding xml. For
   * example, given:
   *
   * <pre>{@code
   *   <layout>
   *     <data>
   *       <variable name="foo" type="String" />
   *     </data>
   *   </layout>
   *   <LinearLayout>
   *     ...
   *   </LinearLayout>
   * }
   * </pre>
   *
   * <p>data binding strips out and processes this part:
   *
   * <pre>{@code
   *     <data>
   *       <variable name="foo" type="String" />
   *     </data>
   * }
   * </pre>
   *
   * for each layout file with data binding expressions. Since this may produce multiple
   * files, outputs are zipped up into a single container.
   */
  static Artifact getLayoutInfoFile(RuleContext ruleContext) {
    // The data binding library expects this to be called "layout-info.zip".
    return ruleContext.getUniqueDirectoryArtifact("databinding", "layout-info.zip",
        ruleContext.getBinOrGenfilesDirectory());
  }

  /**
   * Adds the support libraries needed to compile/run Java code with data binding.
   *
   * <p>This excludes the annotation processor, which is injected separately as a Java plugin
   * (see {@link #addAnnotationProcessor}).
   */
  static ImmutableList<TransitiveInfoCollection> addSupportLibs(RuleContext ruleContext,
      List<? extends TransitiveInfoCollection> deps) {
    RuleConfiguredTarget.Mode mode = RuleConfiguredTarget.Mode.TARGET;
    return ImmutableList.<TransitiveInfoCollection>builder()
        .addAll(deps)
        .addAll(ruleContext.getPrerequisites(DATABINDING_RUNTIME_ATTR, mode))
        .build();
  }

  /**
   * Adds data binding's annotation processor as a plugin to the given Java compilation context.
   *
   * <p>This, in conjunction with {@link #createAnnotationFile} extends the Java compilation to
   * translate data binding .xml into corresponding classes.
   */
  static void addAnnotationProcessor(RuleContext ruleContext,
      JavaTargetAttributes.Builder attributes) {
    JavaPluginInfoProvider plugin = ruleContext.getPrerequisite(
        DATABINDING_ANNOTATION_PROCESSOR_ATTR, RuleConfiguredTarget.Mode.TARGET,
        JavaPluginInfoProvider.class);
    for (String name : plugin.getProcessorClasses()) {
      // For header compilation (see JavaHeaderCompileAction):
      attributes.addApiGeneratingProcessorName(name);
      // For full compilation:
      attributes.addProcessorName(name);
    }
    // For header compilation (see JavaHeaderCompileAction):
    attributes.addApiGeneratingProcessorPath(plugin.getProcessorClasspath());
    // For full compilation:
    attributes.addProcessorPath(plugin.getProcessorClasspath());
    attributes.addAdditionalOutputs(getMetadataOutputs(ruleContext));
  }

  /**
   * Creates and returns the generated Java source that data binding's annotation processor
   * reads to translate layout info xml (from {@link #getLayoutInfoFile} into the classes that
   * end user code consumes.
   */
  static Artifact createAnnotationFile(RuleContext ruleContext, boolean isLibrary) {
    Template template =
        Template.forResource(DataBinding.class, "databinding_annotation_template.txt");

    List<Substitution> subs = new ArrayList<>();
    subs.add(Substitution.of("%module_package%", AndroidCommon.getJavaPackage(ruleContext)));
    // TODO(gregce): clarify or remove the sdk root
    subs.add(Substitution.of("%sdk_root%", "/not/used"));
    subs.add(Substitution.of("%layout_info_dir%",
        getLayoutInfoFile(ruleContext).getExecPath().getParentDirectory().toString()));
    subs.add(Substitution.of("%export_class_list_to%", "/tmp/exported_classes")); // Unused.
    subs.add(Substitution.of("%is_library%", Boolean.toString(isLibrary)));
    subs.add(Substitution.of("%min_sdk%", "14")); // TODO(gregce): update this

    Artifact output = ruleContext.getPackageRelativeArtifact(
        String.format("databinding/%s/DataBindingInfo.java", ruleContext.getLabel().getName()),
        ruleContext.getConfiguration().getGenfilesDirectory());

    ruleContext.registerAction
        (new TemplateExpansionAction(ruleContext.getActionOwner(), output, template, subs, false));

    return output;
  }

  /**
   * Adds the appropriate {@link UsesDataBindingProvider} for a rule if it should expose one.
   *
   * <p>A rule exposes {@link UsesDataBindingProvider} if either it or its deps set
   * {@code enable_data_binding = 1}.
   */
  public static void maybeAddProvider(RuleConfiguredTargetBuilder builder,
      RuleContext ruleContext) {
    // Expose the data binding provider if this rule either applies data binding or exports a dep
    // that applies it.
    List<Artifact> dataBindingMetadataOutputs = new ArrayList<>();
    if (DataBinding.isEnabled(ruleContext)) {
      dataBindingMetadataOutputs.addAll(getMetadataOutputs(ruleContext));
    }
    if (ruleContext.attributes().has("exports", BuildType.LABEL_LIST)) {
      for (UsesDataBindingProvider provider : ruleContext.getPrerequisites("exports",
          RuleConfiguredTarget.Mode.TARGET, UsesDataBindingProvider.class)) {
        dataBindingMetadataOutputs.addAll(provider.getMetadataOutputs());
      }
    }
    if (!dataBindingMetadataOutputs.isEmpty()) {
      // QUESTION(gregce): does a rule need to propagate the metadata outputs of its deps, or do
      // they get integrated automatically into its own outputs?
      builder.addProvider(UsesDataBindingProvider.class,
          new UsesDataBindingProvider(dataBindingMetadataOutputs));
    }
  }

  /**
   * Annotation processing creates the following metadata files that describe how data binding is
   * applied. The full file paths include prefixes as implemented in {@link #getMetadataOutputs}.
   */
  private static final ImmutableList<String> METADATA_OUTPUT_SUFFIXES = ImmutableList.<String>of(
      "setter_store.bin", "layoutinfo.bin", "br.bin");

  /**
   * Returns metadata outputs from this rule's annotation processing that describe what it did with
   * data binding. This is used by parent rules to ensure consistent binding patterns.
   *
   * <p>>For example, if an {@code android_binary} depends on an {@code android_library} in a
   * different package, the {@code android_library}'s version gets packaged with the application
   * jar, even though (due to resource merging) both modules compile against their own instances.
   */
  public static List<Artifact> getMetadataOutputs(RuleContext ruleContext) {
    ImmutableList.Builder<Artifact> outputs = ImmutableList.<Artifact>builder();
    String javaPackage = AndroidCommon.getJavaPackage(ruleContext);
    Label ruleLabel = ruleContext.getRule().getLabel();
    String pathPrefix =
        String.format(
            "_javac/%s/lib%s_classes/%s/%s-",
            ruleLabel.getName(),
            ruleLabel.getPackageIdentifier().getPackageFragment().getBaseName(),
            javaPackage.replace('.', '/'),
            javaPackage);
    for (String suffix : METADATA_OUTPUT_SUFFIXES) {
      outputs.add(ruleContext.getBinArtifact(pathPrefix + suffix));
    }
    return outputs.build();
  }

  /**
   * Processes deps that also apply data binding.
   *
   * @param ruleContext the current rule
   * @param attributes java compilation attributes. The directories of the deps' metadata outputs
   *     (see {@link #getMetadataOutputs}) are added to this rule's annotation processor classpath.
   * @return the deps' metadata outputs. These need to be staged as compilation inputs to the
   *     current rule.
   */
  static ImmutableList<Artifact> processDeps(RuleContext ruleContext,
      JavaTargetAttributes.Builder attributes) {
    ImmutableList.Builder<Artifact> dataBindingJavaInputs = ImmutableList.<Artifact>builder();
    dataBindingJavaInputs.add(DataBinding.getLayoutInfoFile(ruleContext));
    for (UsesDataBindingProvider p : ruleContext.getPrerequisites("deps",
        RuleConfiguredTarget.Mode.TARGET, UsesDataBindingProvider.class)) {
      for (Artifact dataBindingDepMetadata : p.getMetadataOutputs()) {
        attributes.addProcessorPathDir(dataBindingDepMetadata.getExecPath().getParentDirectory());
        dataBindingJavaInputs.add(dataBindingDepMetadata);
      }
    }
    return dataBindingJavaInputs.build();
  }
}

