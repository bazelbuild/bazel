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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.ActionConstructionContext;
import com.google.devtools.build.lib.analysis.actions.SymlinkAction;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.rules.android.AndroidCommon;
import com.google.devtools.build.lib.rules.android.AndroidConfiguration;
import com.google.devtools.build.lib.rules.android.AndroidResources;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.List;

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

  private static final String ENABLE_DATA_BINDING_ATTR = "enable_data_binding";

  private static final DataBindingContext DISABLED_CONTEXT = new DisabledDataBindingV1Context();

  /** Supplies a databinding context from a rulecontext. */
  public static DataBindingContext contextFrom(
      RuleContext ruleContext, AndroidConfiguration androidConfig) {
    if (isEnabled(ruleContext)) {
      if (androidConfig.useDataBindingV2()) {
        return asEnabledDataBindingV2ContextFrom(ruleContext);
      }
      return asEnabledDataBindingV1ContextFrom(ruleContext);
    }
    return asDisabledDataBindingContext();
  }

  /** Supplies a databinding context from an action context. */
  public static DataBindingContext contextFrom(
      boolean enabled, ActionConstructionContext context, AndroidConfiguration androidConfig) {
    if (enabled) {
      if (androidConfig.useDataBindingV2()) {
        return asEnabledDataBindingV2ContextFrom(context);
      }
      return asEnabledDataBindingV1ContextFrom(context);
    }
    return asDisabledDataBindingContext();
  }

  /** Supplies an enabled DataBindingContext from the action context. */
  private static DataBindingContext asEnabledDataBindingV1ContextFrom(
      ActionConstructionContext actionContext) {
    return new DataBindingV1Context(actionContext);
  }

  private static DataBindingContext asEnabledDataBindingV2ContextFrom(
      ActionConstructionContext actionContext) {
    return new DataBindingV2Context(actionContext);
  }

  /** Supplies a disabled (no-op) DataBindingContext. */
  public static DataBindingContext asDisabledDataBindingContext() {
    return DISABLED_CONTEXT;
  }

  /**
   * Annotation processing creates the following metadata files that describe how data binding is
   * applied. The full file paths include prefixes as implemented in {@link #getMetadataOutputs}.
   */
  private static final ImmutableList<String> METADATA_OUTPUT_SUFFIXES =
      ImmutableList.of("setter_store.bin", "layoutinfo.bin", "br.bin");

  /** The directory where the annotation processor looks for dep metadata. */
  private static final String DEP_METADATA_INPUT_DIR = "dependent-lib-artifacts";

  /** The directory where the annotation processor write metadata output for the current rule. */
  private static final String METADATA_OUTPUT_DIR = "bin-files";

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

  /** Returns this rule's data binding base output dir (as an execroot-relative path). */
  static PathFragment getDataBindingExecPath(RuleContext ruleContext) {
    return ruleContext
        .getBinOrGenfilesDirectory()
        .getExecPath()
        .getRelative(ruleContext.getUniqueDirectory("databinding"));
  }

  /** Returns an artifact for the specified output under a standardized data binding base dir. */
  static Artifact getDataBindingArtifact(RuleContext ruleContext, String relativePath) {
    PathFragment binRelativeBasePath =
        getDataBindingExecPath(ruleContext)
            .relativeTo(ruleContext.getBinOrGenfilesDirectory().getExecPath());
    return ruleContext.getDerivedArtifact(
        binRelativeBasePath.getRelative(relativePath), ruleContext.getBinOrGenfilesDirectory());
  }

  /** Turns a key/value pair into a javac annotation processor flag received by data binding. */
  static String createProcessorFlag(String flag, String value) {
    return String.format("-Aandroid.databinding.%s=%s", flag, value);
  }

  /**
   * Adds the appropriate {@link UsesDataBindingProvider} for a rule if it should expose one.
   *
   * <p>A rule exposes {@link UsesDataBindingProvider} if either it or its deps set {@code
   * enable_data_binding = 1}.
   */
  static void maybeAddProvider(
      List<Artifact> dataBindingMetadataOutputs,
      RuleConfiguredTargetBuilder builder,
      RuleContext ruleContext) {
    // Expose the data binding provider if there are outputs.
    dataBindingMetadataOutputs.addAll(getTransitiveMetadata(ruleContext, "exports"));
    if (!AndroidResources.definesAndroidResources(ruleContext.attributes())) {
      // If this rule doesn't declare direct resources, no resource processing is run so no data
      // binding outputs are produced. In that case, we need to explicitly propagate data binding
      // outputs from the deps to make sure they continue up the build graph.
      dataBindingMetadataOutputs.addAll(getTransitiveMetadata(ruleContext, "deps"));
    }
    if (!dataBindingMetadataOutputs.isEmpty()) {
      builder.addNativeDeclaredProvider(new UsesDataBindingProvider(dataBindingMetadataOutputs));
    }
  }

  /** Returns the data binding resource processing output from deps under the given attribute. */
  static List<Artifact> getTransitiveMetadata(RuleContext ruleContext, String attr) {
    ImmutableList.Builder<Artifact> dataBindingMetadataOutputs = ImmutableList.builder();
    if (ruleContext.attributes().has(attr, BuildType.LABEL_LIST)) {
      for (UsesDataBindingProvider provider :
          ruleContext.getPrerequisites(
              attr, RuleConfiguredTarget.Mode.TARGET, UsesDataBindingProvider.PROVIDER)) {
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
  static List<Artifact> getMetadataOutputs(RuleContext ruleContext) {
    if (!AndroidResources.definesAndroidResources(ruleContext.attributes())) {
      // If this rule doesn't define local resources, no resource processing was done, so it
      // doesn't produce data binding output.
      return ImmutableList.of();
    }
    ImmutableList.Builder<Artifact> outputs = ImmutableList.<Artifact>builder();
    String javaPackage = AndroidCommon.getJavaPackage(ruleContext);
    for (String suffix : METADATA_OUTPUT_SUFFIXES) {
      // The annotation processor automatically creates files with this naming pattern under the
      // {@code -Aandroid.databinding.generationalFileOutDir} base directory.
      outputs.add(
          getDataBindingArtifact(
              ruleContext,
              String.format("%s/%s-%s-%s", METADATA_OUTPUT_DIR, javaPackage, javaPackage, suffix)));
    }
    return outputs.build();
  }

  /**
   * Data binding's annotation processor reads the transitive metadata outputs of the target's deps
   * (see {@link #getMetadataOutputs(RuleContext)}) in the directory specified by the processor flag
   * {@code -Aandroid.databinding.bindingBuildFolder}. Since dependencies don't generate their
   * outputs under a common directory, we symlink them into a common place here.
   *
   * @return the symlink paths of the transitive dep metadata outputs for this rule
   */
  static Artifact symlinkDepsMetadataIntoOutputTree(
      RuleContext ruleContext, Artifact depMetadata) {
    Label ruleLabel = ruleContext.getRule().getLabel();
    Artifact symlink =
        getDataBindingArtifact(
            ruleContext,
            String.format(
                "%s/%s", DEP_METADATA_INPUT_DIR, depMetadata.getRootRelativePathString()));
    ruleContext.registerAction(SymlinkAction.toArtifact(
        ruleContext.getActionOwner(),
        depMetadata,
        symlink,
        String.format(
            "Symlinking dep metadata output %s for %s", depMetadata.getFilename(), ruleLabel)));
    return symlink;
  }
}
