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
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.ActionConstructionContext;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.analysis.actions.SymlinkAction;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.rules.java.JavaInfo;
import com.google.devtools.build.lib.rules.java.JavaPluginInfoProvider;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.ResourceFileLoader;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.BiConsumer;
import java.util.function.Consumer;

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
  static final String DATABINDING_ANNOTATION_PROCESSOR_ATTR = "$databinding_annotation_processor";

  private static final String ENABLE_DATA_BINDING_ATTR = "enable_data_binding";

  /** Contains Android Databinding configuration and resource generation information. */
  public interface DataBindingContext {

    /**
     * Returns the file where data binding's resource processing produces binding xml. For example,
     * given:
     *
     * <pre>{@code
     * <layout>
     *   <data>
     *     <variable name="foo" type="String" />
     *   </data>
     * </layout>
     * <LinearLayout>
     *   ...
     * </LinearLayout>
     * }</pre>
     *
     * <p>data binding strips out and processes this part:
     *
     * <pre>{@code
     * <data>
     *   <variable name="foo" type="String" />
     * </data>
     * }</pre>
     *
     * for each layout file with data binding expressions. Since this may produce multiple files,
     * outputs are zipped up into a single container.
     */
    default void supplyLayoutInfo(Consumer<Artifact> consumer) {}

    /** The javac flags that are needed to configure data binding's annotation processor. */
    default void supplyJavaCoptsUsing(
        RuleContext ruleContext, boolean isBinary, Consumer<Iterable<String>> consumer) {}

    /**
     * Adds data binding's annotation processor as a plugin to the given Java compilation context.
     *
     * <p>This extends the Java compilation to translate data binding .xml into corresponding
     * classes.
     */
    default void supplyAnnotationProcessor(
        RuleContext ruleContext, BiConsumer<JavaPluginInfoProvider, Iterable<Artifact>> consumer) {}

    /**
     * Processes deps that also apply data binding.
     *
     * @param ruleContext the current rule
     * @return the deps' metadata outputs. These need to be staged as compilation inputs to the
     *     current rule.
     */
    default ImmutableList<Artifact> processDeps(RuleContext ruleContext) {
      return ImmutableList.of();
    }

    /**
     * Creates and adds the generated Java source for data binding annotation processor to read and
     * translate layout info xml (from {@link #supplyLayoutInfo(Consumer)} into the classes that end
     * user code consumes.
     *
     * <p>This triggers the annotation processor. Annotation processor settings are configured
     * separately in {@link #supplyJavaCoptsUsing(RuleContext, boolean, Consumer)}.
     */
    default ImmutableList<Artifact> addAnnotationFileToSrcs(
        ImmutableList<Artifact> srcs, RuleContext ruleContext) {
      return srcs;
    };

    /**
     * Adds the appropriate {@link UsesDataBindingProvider} for a rule if it should expose one.
     *
     * <p>A rule exposes {@link UsesDataBindingProvider} if either it or its deps set {@code
     * enable_data_binding = 1}.
     */
    default void addProvider(RuleConfiguredTargetBuilder builder, RuleContext ruleContext) {
      maybeAddProvider(new ArrayList<>(), builder, ruleContext);
    }

    default AndroidResources processResources(AndroidResources resources) {
      return resources;
    }
  }

  private static final class EnabledDataBindingV1Context implements DataBindingContext {

    private final ActionConstructionContext actionConstructionContext;

    private EnabledDataBindingV1Context(ActionConstructionContext actionConstructionContext) {
      this.actionConstructionContext = actionConstructionContext;
    }

    @Override
    public void supplyLayoutInfo(Consumer<Artifact> consumer) {
      consumer.accept(layoutInfoFile());
    }

    Artifact layoutInfoFile() {
      return actionConstructionContext.getUniqueDirectoryArtifact("databinding", "layout-info.zip");
    }

    @Override
    public void supplyJavaCoptsUsing(
        RuleContext ruleContext, boolean isBinary, Consumer<Iterable<String>> consumer) {
      ImmutableList.Builder<String> flags = ImmutableList.builder();
      String metadataOutputDir = getDataBindingExecPath(ruleContext).getPathString();

      // Directory where the annotation processor looks for deps metadata output. The annotation
      // processor automatically appends {@link DEP_METADATA_INPUT_DIR} to this path. Individual
      // files can be anywhere under this directory, recursively.
      flags.add(createProcessorFlag("bindingBuildFolder", metadataOutputDir));

      // Directory where the annotation processor should write this rule's metadata output. The
      // annotation processor automatically appends {@link METADATA_OUTPUT_DIR} to this path.
      flags.add(createProcessorFlag("generationalFileOutDir", metadataOutputDir));

      // Path to the Android SDK installation (if available).
      flags.add(createProcessorFlag("sdkDir", "/not/used"));

      // Whether the current rule is a library or binary.
      flags.add(createProcessorFlag("artifactType", isBinary ? "APPLICATION" : "LIBRARY"));

      // The path where data binding's resource processor wrote its output (the data binding XML
      // expressions). The annotation processor reads this file to translate that XML into Java.
      flags.add(createProcessorFlag("xmlOutDir", getDataBindingExecPath(ruleContext).toString()));

      // Unused.
      flags.add(createProcessorFlag("exportClassListTo", "/tmp/exported_classes"));

      // The Java package for the current rule.
      flags.add(createProcessorFlag("modulePackage", AndroidCommon.getJavaPackage(ruleContext)));

      // The minimum Android SDK compatible with this rule.
      flags.add(createProcessorFlag("minApi", "14")); // TODO(gregce): update this

      // If enabled, produces cleaner output for Android Studio.
      flags.add(createProcessorFlag("printEncodedErrors", "0"));

      consumer.accept(flags.build());
    }

    @Override
    public void supplyAnnotationProcessor(
        RuleContext ruleContext, BiConsumer<JavaPluginInfoProvider, Iterable<Artifact>> consumer) {
      consumer.accept(
          JavaInfo.getProvider(
              JavaPluginInfoProvider.class,
              ruleContext.getPrerequisite(
                  DATABINDING_ANNOTATION_PROCESSOR_ATTR, RuleConfiguredTarget.Mode.HOST)),
          getMetadataOutputs(ruleContext));
    }

    @Override
    public ImmutableList<Artifact> processDeps(RuleContext ruleContext) {
      ImmutableList.Builder<Artifact> dataBindingJavaInputs = ImmutableList.builder();
      if (AndroidResources.definesAndroidResources(ruleContext.attributes())) {
        dataBindingJavaInputs.add(layoutInfoFile());
      }
      for (Artifact dataBindingDepMetadata : getTransitiveMetadata(ruleContext, "deps")) {
        dataBindingJavaInputs.add(
            symlinkDepsMetadataIntoOutputTree(ruleContext, dataBindingDepMetadata));
      }
      return dataBindingJavaInputs.build();
    }

    @Override
    public ImmutableList<Artifact> addAnnotationFileToSrcs(
        ImmutableList<Artifact> srcs, RuleContext ruleContext) {
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
        return ImmutableList.<Artifact>builder().addAll(srcs).add(annotationFile).build();
      } catch (IOException e) {
        ruleContext.ruleError("Cannot load annotation processor template: " + e.getMessage());
        return ImmutableList.of();
      }
    }

    @Override
    public void addProvider(RuleConfiguredTargetBuilder builder, RuleContext ruleContext) {
      List<Artifact> dataBindingMetadataOutputs =
          Lists.newArrayList(getMetadataOutputs(ruleContext));
      maybeAddProvider(dataBindingMetadataOutputs, builder, ruleContext);
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (o == null || getClass() != o.getClass()) {
        return false;
      }
      EnabledDataBindingV1Context that = (EnabledDataBindingV1Context) o;
      return Objects.equals(actionConstructionContext, that.actionConstructionContext);
    }

    @Override
    public int hashCode() {
      return actionConstructionContext.hashCode();
    }

    @Override
    public AndroidResources processResources(AndroidResources resources) {
      return resources;
    }
  }

  private static class EnabledDataBindingV2Context implements DataBindingContext {

    private final ActionConstructionContext actionContext;

    private EnabledDataBindingV2Context(ActionConstructionContext actionContext) {
      this.actionContext = actionContext;
      throw new UnsupportedOperationException("V2 not implemented yet.");
    }
    // TODO(b/112038432): Enable databinding v2.
  }

  private static final DataBindingContext DISABLED_CONTEXT = new DataBindingContext() {};

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
    return new EnabledDataBindingV1Context(actionContext);
  }

  private static DataBindingContext asEnabledDataBindingV2ContextFrom(
      ActionConstructionContext actionContext) {
    return new EnabledDataBindingV2Context(actionContext);
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
  private static PathFragment getDataBindingExecPath(RuleContext ruleContext) {
    return ruleContext
        .getBinOrGenfilesDirectory()
        .getExecPath()
        .getRelative(ruleContext.getUniqueDirectory("databinding"));
  }

  /** Returns an artifact for the specified output under a standardized data binding base dir. */
  private static Artifact getDataBindingArtifact(RuleContext ruleContext, String relativePath) {
    PathFragment binRelativeBasePath =
        getDataBindingExecPath(ruleContext)
            .relativeTo(ruleContext.getBinOrGenfilesDirectory().getExecPath());
    return ruleContext.getDerivedArtifact(
        binRelativeBasePath.getRelative(relativePath), ruleContext.getBinOrGenfilesDirectory());
  }

  /** Turns a key/value pair into a javac annotation processor flag received by data binding. */
  private static String createProcessorFlag(String flag, String value) {
    return String.format("-Aandroid.databinding.%s=%s", flag, value);
  }

  /**
   * Adds the appropriate {@link UsesDataBindingProvider} for a rule if it should expose one.
   *
   * <p>A rule exposes {@link UsesDataBindingProvider} if either it or its deps set {@code
   * enable_data_binding = 1}.
   */
  private static void maybeAddProvider(
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
  private static List<Artifact> getTransitiveMetadata(RuleContext ruleContext, String attr) {
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
  private static List<Artifact> getMetadataOutputs(RuleContext ruleContext) {
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
  private static Artifact symlinkDepsMetadataIntoOutputTree(
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
