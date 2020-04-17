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
package com.google.devtools.build.lib.rules.android.databinding;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitionMode;
import com.google.devtools.build.lib.analysis.actions.ActionConstructionContext;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.analysis.actions.SymlinkAction;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.rules.android.AndroidCommon;
import com.google.devtools.build.lib.rules.android.AndroidConfiguration;
import com.google.devtools.build.lib.rules.android.AndroidDataContext;
import com.google.devtools.build.lib.rules.android.AndroidResources;
import com.google.devtools.build.lib.util.ResourceFileLoader;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.List;
import javax.annotation.Nullable;

/**
 * Support logic for Bazel's <a
 * href="https://developer.android.com/topic/libraries/data-binding/index.html">data binding</a>
 * integration.
 *
 * <p>In short, data binding in Bazel works as follows:
 *
 * <ol>
 *   <li>If a rule enables data binding and has layout resources with data binding expressions,
 *       resource processing invokes the data binding library to preprocess these expressions, then
 *       strips them out before feeding the resources into aapt. A separate "layout info" XML file
 *       gets produced that contains the bindings.
 *   <li>The data binding annotation processor gets activated on Java compilation. This processor
 *       reads a custom-generated <code>DataBindingInfo.java</code> which specifies the path to the
 *       layout info file (as an annotation). The processor reads that file and produces the
 *       corresponding Java classes that end-user code uses to access the resources.
 *   <li>The data binding compile-time and runtime support libraries get linked into the binary's
 *       deploy jar.
 * </ol>
 *
 * <p>For data binding to work, the corresponding support libraries must be checked into the depot
 * via the implicit dependencies specified inside this class.
 */
public final class DataBinding {

  /** The rule attribute supplying data binding's annotation processor. */
  public static final String DATABINDING_ANNOTATION_PROCESSOR_ATTR =
      "$databinding_annotation_processor";

  /** The rule attribute supplying data binding's build helper (exec). */
  public static final String DATABINDING_EXEC_PROCESSOR_ATTR = "$databinding_exec";

  private static final String ENABLE_DATA_BINDING_ATTR = "enable_data_binding";

  /** The directory where the annotation processor looks for dep metadata. */
  public static final String DEP_METADATA_INPUT_DIR = "dependent-lib-artifacts";

  /** The directory where the annotation processor writes metadata output for the current rule. */
  public static final String METADATA_OUTPUT_DIR = "bin-files";

  @VisibleForTesting
  public static final DataBindingContext DISABLED_V1_CONTEXT = new DisabledDataBindingV1Context();

  private static final DataBindingContext DISABLED_V2_CONTEXT = new DisabledDataBindingV2Context();

  /** Supplies a databinding context from a rulecontext. */
  public static DataBindingContext contextFrom(
      RuleContext ruleContext, AndroidConfiguration androidConfig) {

    return contextFrom(isEnabled(ruleContext), ruleContext, androidConfig);
  }

  /** Supplies a databinding context from an action context. */
  public static DataBindingContext contextFrom(
      boolean enabled, ActionConstructionContext context, AndroidConfiguration androidConfig) {

    if (enabled) {
      if (androidConfig.useDataBindingV2()) {
        return new DataBindingV2Context(context, androidConfig.useDataBindingUpdatedArgs());
      } else {
        return new DataBindingV1Context(context, androidConfig.useDataBindingUpdatedArgs());
      }
    } else {
      if (androidConfig.useDataBindingV2()) {
        return DISABLED_V2_CONTEXT;
      } else {
        return DISABLED_V1_CONTEXT;
      }
    }
  }

  /**
   * Should data binding support be enabled for this rule?
   *
   * <p>Data binding incurs additional resource processing and compilation work as well as
   * additional compile/runtime dependencies. But rules with data binding disabled will fail if data
   * binding expressions appear in their layout resources.
   */
  private static boolean isEnabled(RuleContext ruleContext) {
    return ruleContext.attributes().has(ENABLE_DATA_BINDING_ATTR, Type.BOOLEAN)
        && Boolean.TRUE.equals(
            ruleContext.attributes().get(ENABLE_DATA_BINDING_ATTR, Type.BOOLEAN));
  }

  /** Supplies a disabled (no-op) DataBindingContext. */
  public static DataBindingContext getDisabledDataBindingContext(AndroidDataContext ctx) {
    if (ctx.useDataBindingV2()) {
      return DISABLED_V2_CONTEXT;
    } else {
      return DISABLED_V1_CONTEXT;
    }
  }

  /** Returns this rule's data binding base output dir (as an execroot-relative path). */
  static PathFragment getDataBindingExecPath(RuleContext ruleContext) {
    return ruleContext
        .getBinOrGenfilesDirectory()
        .getExecPath()
        .getRelative(ruleContext.getUniqueDirectory("databinding"));
  }

  @VisibleForTesting
  public static Artifact getLayoutInfoFile(ActionConstructionContext actionConstructionContext) {
    return actionConstructionContext.getUniqueDirectoryArtifact("databinding", "layout-info.zip");
  }

  /** Returns an artifact for the specified output under a standardized data binding base dir. */
  static Artifact getDataBindingArtifact(RuleContext ruleContext, String relativePath) {
    PathFragment binRelativeBasePath =
        getDataBindingExecPath(ruleContext)
            .relativeTo(ruleContext.getBinOrGenfilesDirectory().getExecPath());
    return ruleContext.getDerivedArtifact(
        binRelativeBasePath.getRelative(relativePath), ruleContext.getBinOrGenfilesDirectory());
  }

  static ImmutableList<Artifact> getAnnotationFile(RuleContext ruleContext) {
    // Add this rule's annotation processor input. If the rule doesn't have direct resources,
    // there's no direct data binding info, so there's strictly no need for annotation processing.
    // But it's still important to process the deps' .bin files so any Java class references get
    // re-referenced so they don't get filtered out of the compilation classpath by JavaBuilder
    // (which filters out classpath .jars that "aren't used": see --reduce_classpath). If data
    // binding didn't reprocess a library's data binding expressions redundantly up the dependency
    // chain (meaning each depender processes them again as if they were its own), this problem
    // wouldn't happen.
    try {
      String contents =
          ResourceFileLoader.loadResource(
              DataBinding.class, "databinding_annotation_template.txt");
      Artifact annotationFile = getDataBindingArtifact(ruleContext, "DataBindingInfo.java");
      ruleContext.registerAction(
          FileWriteAction.create(ruleContext, annotationFile, contents, false));
      return ImmutableList.of(annotationFile);
    } catch (IOException e) {
      ruleContext.ruleError("Cannot load annotation processor template: " + e.getMessage());
      return ImmutableList.of();
    }
  }

  /** Returns the data binding resource processing output from deps under the given attribute. */
  static List<Artifact> getTransitiveMetadata(RuleContext ruleContext, String attr) {
    ImmutableList.Builder<Artifact> dataBindingMetadataOutputs = ImmutableList.builder();
    if (ruleContext.attributes().has(attr, BuildType.LABEL_LIST)) {
      for (UsesDataBindingProvider provider :
          ruleContext.getPrerequisites(
              attr, TransitionMode.TARGET, UsesDataBindingProvider.PROVIDER)) {
        dataBindingMetadataOutputs.addAll(provider.getMetadataOutputs());
      }
    }
    return dataBindingMetadataOutputs.build();
  }

  /**
   * Returns metadata outputs from this rule's annotation processing that describe what it did with
   * data binding. This is used by parent rules to ensure consistent binding patterns.
   *
   * <p>>For example, if {@code foo.AndroidBinary} depends on {@code foo.lib.AndroidLibrary} and the
   * library defines data binding expression {@code Bar}, compiling the library produces Java class
   * {@code foo.lib.Bar}. But since the binary applies data binding over the merged resources of its
   * deps, that means the binary also sees {@code Bar}, so it compiles it into {@code foo.Bar}. This
   * would be a class redefinition conflict. But by feeding the library's metadata outputs into the
   * binary's compilation, enough information is available to only use the first version.
   */
  static ImmutableList<Artifact> getMetadataOutputs(
      RuleContext ruleContext, boolean useUpdatedArgs, List<String> metadataOutputSuffixes) {

    if (!AndroidResources.definesAndroidResources(ruleContext.attributes())) {
      // If this rule doesn't define local resources, no resource processing was done, so it
      // doesn't produce data binding output.
      return ImmutableList.of();
    }
    ImmutableList.Builder<Artifact> outputs = ImmutableList.<Artifact>builder();
    String javaPackage = AndroidCommon.getJavaPackage(ruleContext);
    for (String suffix : metadataOutputSuffixes) {
      // The annotation processor automatically creates files with this naming pattern under the
      // {@code -Aandroid.databinding.generationalFileOutDir} base directory.
      if (useUpdatedArgs) {
        outputs.add(
            getDataBindingArtifact(
                ruleContext, String.format("%s/%s-%s", METADATA_OUTPUT_DIR, javaPackage, suffix)));
      } else {
        outputs.add(
            getDataBindingArtifact(
                ruleContext,
                String.format(
                    "%s/%s-%s-%s", METADATA_OUTPUT_DIR, javaPackage, javaPackage, suffix)));
      }
    }
    return outputs.build();
  }

  @Nullable
  static Artifact getMetadataOutput(
      RuleContext ruleContext, boolean useUpdatedArgs, String metadataOutputSuffix) {

    if (!AndroidResources.definesAndroidResources(ruleContext.attributes())) {
      // If this rule doesn't define local resources, no resource processing was done, so it
      // doesn't produce data binding output.
      return null;
    }
    String javaPackage = AndroidCommon.getJavaPackage(ruleContext);

    // The annotation processor automatically creates files with this naming pattern under the
    // {@code -Aandroid.databinding.generationalFileOutDir} base directory.
    if (useUpdatedArgs) {
      return getDataBindingArtifact(
          ruleContext,
          String.format("%s/%s-%s", METADATA_OUTPUT_DIR, javaPackage, metadataOutputSuffix));
    } else {
      return getDataBindingArtifact(
          ruleContext,
          String.format(
              "%s/%s-%s-%s", METADATA_OUTPUT_DIR, javaPackage, javaPackage, metadataOutputSuffix));
    }
  }

  /**
   * Data binding's annotation processor reads the transitive metadata outputs of the target's deps
   * (see {@link #getMetadataOutputs(RuleContext, List<String>)}) in the directory specified by the
   * processor flag {@code -Aandroid.databinding.bindingBuildFolder}. Since dependencies don't
   * generate their outputs under a common directory, we symlink them into a common place here.
   *
   * @return the symlink paths of the transitive dep metadata outputs for this rule
   */
  static Artifact symlinkDepsMetadataIntoOutputTree(RuleContext ruleContext, Artifact depMetadata) {

    Label ruleLabel = ruleContext.getRule().getLabel();
    Artifact symlink =
        getDataBindingArtifact(
            ruleContext,
            String.format(
                "%s/%s", DEP_METADATA_INPUT_DIR, depMetadata.getRootRelativePathString()));
    ruleContext.registerAction(
        SymlinkAction.toArtifact(
            ruleContext.getActionOwner(),
            depMetadata,
            symlink,
            String.format(
                "Symlinking dep metadata output %s for %s", depMetadata.getFilename(), ruleLabel)));
    return symlink;
  }

}
