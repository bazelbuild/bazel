// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.java;

import static com.google.devtools.build.lib.analysis.config.BuildConfiguration.StrictDepsMode.OFF;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
import com.google.devtools.build.lib.analysis.AnalysisEnvironment;
import com.google.devtools.build.lib.analysis.AnalysisUtils;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.StrictDepsMode;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgs.ClasspathType;
import com.google.devtools.build.lib.rules.java.JavaConfiguration.JavaClasspathMode;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import javax.annotation.Nullable;

/**
 * A helper class for compiling Java targets. It contains method to create the
 * various intermediate Artifacts for using ijars and source ijars.
 * <p>
 * Also supports the creation of resource and source only Jars.
 */
public final class JavaCompilationHelper extends BaseJavaCompilationHelper {

  /**
   * Maximum memory to use for GenClass for generating the gen jar.
   */
  private static final String GENCLASS_MAX_MEMORY = "-Xmx64m";

  private JavaTargetAttributes.Builder attributes;
  private JavaTargetAttributes builtAttributes;
  private final ImmutableList<String> customJavacOpts;
  private final ImmutableList<String> customJavacJvmOpts;
  private final List<Artifact> translations = new ArrayList<>();
  private boolean translationsFrozen;
  private final JavaSemantics semantics;

  public JavaCompilationHelper(RuleContext ruleContext, JavaSemantics semantics,
      ImmutableList<String> javacOpts, JavaTargetAttributes.Builder attributes,
      String implicitAttributesSuffix) {
    super(ruleContext, implicitAttributesSuffix);
    this.attributes = attributes;
    this.customJavacOpts = javacOpts;
    this.customJavacJvmOpts = javaToolchain.getJavacJvmOptions();
    this.semantics = semantics;
  }

  public JavaCompilationHelper(RuleContext ruleContext, JavaSemantics semantics,
      ImmutableList<String> javacOpts, JavaTargetAttributes.Builder attributes) {
    this(ruleContext, semantics, javacOpts, attributes, "");
  }

  public JavaCompilationHelper(RuleContext ruleContext, JavaSemantics semantics,
      JavaTargetAttributes.Builder attributes) {
    this(ruleContext, semantics, getDefaultJavacOptsFromRule(ruleContext), attributes);
  }

  public JavaTargetAttributes getAttributes() {
    if (builtAttributes == null) {
      builtAttributes = attributes.build();
    }
    return builtAttributes;
  }

  public RuleContext getRuleContext() {
    return ruleContext;
  }

  private AnalysisEnvironment getAnalysisEnvironment() {
    return ruleContext.getAnalysisEnvironment();
  }

  private BuildConfiguration getConfiguration() {
    return ruleContext.getConfiguration();
  }

  private JavaConfiguration getJavaConfiguration() {
    return ruleContext.getFragment(JavaConfiguration.class);
  }

  /**
   * Creates the Action that compiles Java source files.
   *
   * @param outputJar the class jar Artifact to create with the Action
   * @param manifestProtoOutput the output artifact for the manifest proto emitted from JavaBuilder
   * @param gensrcOutputJar the generated sources jar Artifact to create with the Action
   *        (null if no sources will be generated).
   * @param outputDepsProto the compiler-generated jdeps file to create with the Action
   *        (null if not requested)
   * @param outputMetadata metadata file (null if no instrumentation is needed).
   */
  public void createCompileAction(
      Artifact outputJar,
      Artifact manifestProtoOutput,
      @Nullable Artifact gensrcOutputJar,
      @Nullable Artifact outputDepsProto,
      @Nullable Artifact outputMetadata) {

    JavaTargetAttributes attributes = getAttributes();
    JavaCompileAction.Builder builder = createJavaCompileActionBuilder(semantics);
    builder.setClasspathEntries(attributes.getCompileTimeClassPath());
    builder.addResources(attributes.getResources());
    builder.addClasspathResources(attributes.getClassPathResources());
    builder.setBootclasspathEntries(getBootclasspathOrDefault());
    builder.setExtdirInputs(getExtdirInputs());
    builder.setLangtoolsJar(getLangtoolsJar());
    builder.setJavaBuilderJar(getJavaBuilderJar());
    builder.addTranslations(getTranslations());
    builder.setOutputJar(outputJar);
    builder.setManifestProtoOutput(manifestProtoOutput);
    builder.setGensrcOutputJar(gensrcOutputJar);
    builder.setOutputDepsProto(outputDepsProto);
    builder.setMetadata(outputMetadata);
    builder.setInstrumentationJars(getInstrumentationJars());
    builder.addSourceFiles(attributes.getSourceFiles());
    builder.addSourceJars(attributes.getSourceJars());
    builder.setJavacOpts(customJavacOpts);
    builder.setJavacJvmOpts(customJavacJvmOpts);
    builder.setCompressJar(true);
    builder.setSourceGenDirectory(sourceGenDir(outputJar));
    builder.setTempDirectory(tempDir(outputJar));
    builder.setClassDirectory(classDir(outputJar));
    builder.addProcessorPaths(attributes.getProcessorPath());
    builder.addProcessorNames(attributes.getProcessorNames());
    builder.setStrictJavaDeps(attributes.getStrictJavaDeps());
    builder.addDirectJars(attributes.getDirectJars());
    builder.addCompileTimeDependencyArtifacts(attributes.getCompileTimeDependencyArtifacts());
    builder.setRuleKind(attributes.getRuleKind());
    builder.setTargetLabel(attributes.getTargetLabel());
    getAnalysisEnvironment().registerAction(builder.build());
  }

  /** Returns the bootclasspath explicit set in attributes if present, or else the default. */
  private ImmutableList<Artifact> getBootclasspathOrDefault() {
    JavaTargetAttributes attributes = getAttributes();
    if (!attributes.getBootClassPath().isEmpty()) {
      return attributes.getBootClassPath();
    } else {
      return getBootClasspath();
    }
  }

  /**
   * Creates the Action that compiles Java source files and optionally instruments them for
   * coverage.
   *
   * @param outputJar the class jar Artifact to create with the Action
   * @param manifestProtoOutput the output artifact for the manifest proto emitted from JavaBuilder
   * @param gensrcJar the generated sources jar Artifact to create with the Action
   * @param outputDepsProto the compiler-generated jdeps file to create with the Action
   * @param javaArtifactsBuilder the build to store the instrumentation metadata in
   */
  public void createCompileActionWithInstrumentation(
      Artifact outputJar,
      Artifact manifestProtoOutput,
      @Nullable Artifact gensrcJar,
      @Nullable Artifact outputDepsProto,
      JavaCompilationArtifacts.Builder javaArtifactsBuilder) {
    createCompileAction(
        outputJar,
        manifestProtoOutput,
        gensrcJar,
        outputDepsProto,
        createInstrumentationMetadata(outputJar, javaArtifactsBuilder));
  }

  /**
   * Creates the instrumentation metadata artifact if needed.
   *
   * @return the instrumentation metadata artifact or null if instrumentation is
   *         disabled
   */
  public Artifact createInstrumentationMetadata(Artifact outputJar,
      JavaCompilationArtifacts.Builder javaArtifactsBuilder) {
    // If we need to instrument the jar, add additional output (the coverage metadata file) to the
    // JavaCompileAction.
    Artifact instrumentationMetadata = null;
    if (shouldInstrumentJar()) {
      instrumentationMetadata = semantics.createInstrumentationMetadataArtifact(
          getRuleContext(), outputJar);

      if (instrumentationMetadata != null) {
        javaArtifactsBuilder.addInstrumentationMetadata(instrumentationMetadata);
      }
    }
    return instrumentationMetadata;
  }

  private boolean shouldInstrumentJar() {
    // TODO(bazel-team): What about source jars?
    return getConfiguration().isCodeCoverageEnabled() && attributes.hasSourceFiles() &&
        getConfiguration().getInstrumentationFilter().isIncluded(
            getRuleContext().getLabel().toString());
  }

  private boolean shouldUseHeaderCompilation() {
    if (!getJavaConfiguration().useHeaderCompilation()) {
      return false;
    }
    if (!attributes.hasSourceFiles() && !attributes.hasSourceJars()) {
      return false;
    }
    if (javaToolchain.getHeaderCompiler() == null) {
      getRuleContext().ruleError(
          "header compilation was requested but it is not support by the current Java toolchain;"
              + " see the java_toolchain.header_compiler attribute");
      return false;
    }
    return true;
  }

  /**
   * Creates the Action that compiles ijars from source.
   *
   * @param runtimeJar the jar output of this java compilation, used to create output-relative
   *     paths for new artifacts.
   */
  private Artifact createHeaderCompilationAction(
      Artifact runtimeJar, JavaCompilationArtifacts.Builder artifactBuilder) {

    Artifact headerJar =
        getAnalysisEnvironment()
            .getDerivedArtifact(
                FileSystemUtils.replaceExtension(runtimeJar.getRootRelativePath(), "-hjar.jar"),
                runtimeJar.getRoot());
    Artifact headerDeps =
        getAnalysisEnvironment()
            .getDerivedArtifact(
                FileSystemUtils.replaceExtension(runtimeJar.getRootRelativePath(), "-hjar.jdeps"),
                runtimeJar.getRoot());

    JavaTargetAttributes attributes = getAttributes();
    JavaHeaderCompileActionBuilder builder = new JavaHeaderCompileActionBuilder(getRuleContext());
    builder.addSourceFiles(attributes.getSourceFiles());
    builder.addSourceJars(attributes.getSourceJars());
    builder.setClasspathEntries(attributes.getCompileTimeClassPath());
    builder.addAllBootclasspathEntries(getBootclasspathOrDefault());
    builder.addAllExtClasspathEntries(getExtdirInputs());
    // TODO(cushon): restrict to only API-generating annotation processors
    builder.addProcessorPaths(attributes.getProcessorPath());
    builder.addProcessorNames(attributes.getProcessorNames());
    builder.setJavacOpts(getJavacOpts());
    builder.setTempDirectory(tempDir(headerJar));
    builder.setOutputJar(headerJar);
    builder.setOutputDepsProto(headerDeps);
    builder.setStrictJavaDeps(attributes.getStrictJavaDeps());
    builder.addCompileTimeDependencyArtifacts(attributes.getCompileTimeDependencyArtifacts());
    builder.addDirectJars(attributes.getDirectJars());
    builder.setRuleKind(attributes.getRuleKind());
    builder.setTargetLabel(attributes.getTargetLabel());
    builder.build();

    artifactBuilder.setCompileTimeDependencies(headerDeps);
    return headerJar;
  }

  /**
   * Returns the artifact for a jar file containing class files that were generated by
   * annotation processors.
   */
  public Artifact createGenJar(Artifact outputJar) {
    return getRuleContext().getDerivedArtifact(
        FileSystemUtils.appendWithoutExtension(outputJar.getRootRelativePath(), "-gen"),
        outputJar.getRoot());
  }

  /**
   * Returns the artifact for a jar file containing source files that were generated by
   * annotation processors.
   */
  public Artifact createGensrcJar(Artifact outputJar) {
    return getRuleContext().getDerivedArtifact(
        FileSystemUtils.appendWithoutExtension(outputJar.getRootRelativePath(), "-gensrc"),
        outputJar.getRoot());
  }

  /**
   * Returns whether this target uses annotation processing.
   */
  public boolean usesAnnotationProcessing() {
    JavaTargetAttributes attributes = getAttributes();
    return getJavacOpts().contains("-processor") || !attributes.getProcessorNames().isEmpty();
  }

  /**
   * Returns the artifact for the manifest proto emitted from JavaBuilder. For example, for a
   * class jar foo.jar, returns "foo.jar_manifest_proto".
   *
   * @param outputJar The artifact for the class jar emitted form JavaBuilder
   * @return The output artifact for the manifest proto emitted from JavaBuilder
   */
  public Artifact createManifestProtoOutput(Artifact outputJar) {
    return getRuleContext().getDerivedArtifact(
        FileSystemUtils.appendExtension(outputJar.getRootRelativePath(), "_manifest_proto"),
        outputJar.getRoot());
  }

  /**
   * Creates the action for creating the gen jar.
   *
   * @param classJar The artifact for the class jar emitted from JavaBuilder
   * @param manifestProto The artifact for the manifest proto emitted from JavaBuilder
   * @param genClassJar The artifact for the gen jar to output
   */
  public void createGenJarAction(Artifact classJar, Artifact manifestProto,
      Artifact genClassJar) {
    getRuleContext().registerAction(new SpawnAction.Builder()
      .addInput(manifestProto)
      .addInput(classJar)
      .addOutput(genClassJar)
      .addTransitiveInputs(getHostJavabaseInputsNonStatic(getRuleContext()))
      .setJarExecutable(
          getRuleContext().getHostConfiguration().getFragment(Jvm.class).getJavaExecutable(),
          getGenClassJar(ruleContext),
          ImmutableList.of("-client", GENCLASS_MAX_MEMORY))
      .setCommandLine(CustomCommandLine.builder()
          .addExecPath("--manifest_proto", manifestProto)
          .addExecPath("--class_jar", classJar)
          .addExecPath("--output_jar", genClassJar)
          .add("--temp_dir").addPath(tempDir(genClassJar))
          .build())
      .useParameterFile(ParameterFileType.SHELL_QUOTED)
      .setProgressMessage("Building genclass jar " + genClassJar.prettyPrint())
      .setMnemonic("JavaSourceJar")
      .build(getRuleContext()));
  }

  /** Returns the GenClass deploy jar Artifact. */
  private Artifact getGenClassJar(RuleContext ruleContext) {
    Artifact genClass = javaToolchain.getGenClass();
    if (genClass != null) {
      return genClass;
    }
    return ruleContext.getPrerequisiteArtifact("$genclass", Mode.HOST);
  }

  /**
   * Creates the jdeps file artifact if needed. Returns null if the target can't emit dependency
   * information (i.e there is no compilation step, the target acts as an alias).
   *
   * @param outputJar output jar artifact used to derive the name
   * @return the jdeps file artifact or null if the target can't generate such a file
   */
  public Artifact createOutputDepsProtoArtifact(Artifact outputJar,
      JavaCompilationArtifacts.Builder builder) {
    if (!generatesOutputDeps()) {
      return null;
    }

    Artifact outputDepsProtoArtifact =
        getRuleContext()
            .getDerivedArtifact(
                FileSystemUtils.replaceExtension(outputJar.getRootRelativePath(), ".jdeps"),
                outputJar.getRoot());

    builder.setCompileTimeDependencies(outputDepsProtoArtifact);
    return outputDepsProtoArtifact;
  }

  /**
   * Returns whether this target emits dependency information. Compilation must occur, so certain
   * targets acting as aliases have to be filtered out.
   */
  private boolean generatesOutputDeps() {
    return getJavaConfiguration().getGenerateJavaDeps() &&
        (attributes.hasSourceFiles() || attributes.hasSourceJars());
  }

  /**
   * Creates an Action that packages all of the resources into a Jar. This
   * includes the declared resources, the classpath resources and the translated
   * messages.
   *
   * <p>The resource jar artifact is derived from the given original jar, by
   * prepending the given prefix and appending the given suffix. The new jar
   * uses the same root as the original jar.
   */
  // TODO(bazel-team): Extract this method to make it easier to create simple
  // zip/jar archives without having to first create a JavaCompilationhelper and
  // JavaTargetAttributes.
  public Artifact createResourceJarAction(Artifact resourceJar) {
    JavaTargetAttributes attributes = getAttributes();
    JavaCompileAction.Builder builder = createJavaCompileActionBuilder(semantics);
    builder.setOutputJar(resourceJar);
    builder.addResources(attributes.getResources());
    builder.addClasspathResources(attributes.getClassPathResources());
    builder.setExtdirInputs(getExtdirInputs());
    builder.setLangtoolsJar(getLangtoolsJar());
    builder.addTranslations(getTranslations());
    builder.setCompressJar(true);
    builder.setTempDirectory(tempDir(resourceJar));
    builder.setClassDirectory(classDir(resourceJar));
    builder.setJavaBuilderJar(getJavaBuilderJar());
    builder.setJavacOpts(getDefaultJavacOptsFromRule(getRuleContext()));
    builder.setJavacJvmOpts(javaToolchain.getJavacJvmOptions());
    getAnalysisEnvironment().registerAction(builder.build());
    return resourceJar;
  }

  private JavaCompileAction.Builder createJavaCompileActionBuilder(
      JavaSemantics semantics) {
    JavaCompileAction.Builder builder = new JavaCompileAction.Builder(ruleContext, semantics);
    builder.setJavaExecutable(
        ruleContext.getHostConfiguration().getFragment(Jvm.class).getJavaExecutable());
    builder.setJavaBaseInputs(getHostJavabaseInputsNonStatic(ruleContext));
    return builder;
  }

  /**
   * Produces a derived directory where source files generated by annotation processors should be
   * stored.
   */
  private PathFragment sourceGenDir(Artifact outputJar) {
    return workDir(outputJar, "_sourcegenfiles");
  }

  private PathFragment tempDir(Artifact outputJar) {
    return workDir(outputJar, "_temp");
  }

  private PathFragment classDir(Artifact outputJar) {
    return workDir(outputJar, "_classes");
  }

  /**
   * For an output jar and a suffix, produces a derived directory under
   * {@code bin} directory with a given suffix.
   *
   * <p>Note that this won't work if a rule produces two jars with the same basename.
   */
  private PathFragment workDir(Artifact outputJar, String suffix) {
    String basename = FileSystemUtils.removeExtension(outputJar.getExecPath().getBaseName());
    return getConfiguration().getBinDirectory().getExecPath()
        .getRelative(ruleContext.getUniqueDirectory("_javac"))
        .getRelative(basename + suffix);
  }

  /**
   * Creates an Action that packages the Java source files into a Jar.  If {@code gensrcJar} is
   * non-null, includes the contents of the {@code gensrcJar} with the output source jar.
   *
   * @param outputJar the Artifact to create with the Action
   * @param gensrcJar the generated sources jar Artifact that should be included with the
   *        sources in the output Artifact.  May be null.
   */
  public void createSourceJarAction(Artifact outputJar, @Nullable Artifact gensrcJar) {
    JavaTargetAttributes attributes = getAttributes();
    Collection<Artifact> resourceJars = new ArrayList<>(attributes.getSourceJars());
    if (gensrcJar != null) {
      resourceJars.add(gensrcJar);
    }
    Map<PathFragment, Artifact> resources = new LinkedHashMap<>();
    for (Artifact sourceFile : attributes.getSourceFiles()) {
      resources.put(semantics.getDefaultJavaResourcePath(sourceFile.getRootRelativePath()), sourceFile);
    }
    SingleJarActionBuilder.createSourceJarAction(ruleContext, resources, resourceJars, outputJar);
  }

  /**
   * Creates the actions that produce the interface jar. Adds the jar artifacts to the given
   * JavaCompilationArtifacts builder.
   *
   * @return the header jar (if requested), or ijar (if requested), or else the class jar
   */
  public Artifact createCompileTimeJarAction(
      Artifact runtimeJar, JavaCompilationArtifacts.Builder builder) {
    Artifact jar;
    if (shouldUseHeaderCompilation()) {
      jar = createHeaderCompilationAction(runtimeJar, builder);
    } else if (getJavaConfiguration().getUseIjars()) {
      jar = createIjarAction(runtimeJar, false);
    } else {
      jar = runtimeJar;
    }
    builder.addCompileTimeJar(jar);
    return jar;
  }

  private void addArgsAndJarsToAttributes(JavaCompilationArgs args, Iterable<Artifact> directJars) {
    // Can only be non-null when isStrict() returns true.
    if (directJars != null) {
      attributes.addDirectCompileTimeClassPathEntries(directJars);
      attributes.addDirectJars(directJars);
    }

    attributes.merge(args);
  }

  private void addLibrariesToAttributesInternal(Iterable<? extends TransitiveInfoCollection> deps) {
    JavaCompilationArgs args = JavaCompilationArgs.builder()
        .addTransitiveTargets(deps).build();

    NestedSet<Artifact> directJars = isStrict()
        ? getNonRecursiveCompileTimeJarsFromCollection(deps)
        : null;
    addArgsAndJarsToAttributes(args, directJars);
  }

  private void addProvidersToAttributesInternal(
      Iterable<? extends SourcesJavaCompilationArgsProvider> deps, boolean isNeverLink) {
    JavaCompilationArgs args = JavaCompilationArgs.builder()
        .addSourcesTransitiveCompilationArgs(deps, true,
            isNeverLink ? ClasspathType.COMPILE_ONLY : ClasspathType.BOTH)
        .build();

    NestedSet<Artifact> directJars = isStrict()
        ? getNonRecursiveCompileTimeJarsFromProvider(deps, isNeverLink)
        : null;
    addArgsAndJarsToAttributes(args, directJars);
  }

  private boolean isStrict() {
    return getStrictJavaDeps() != OFF;
  }

  private NestedSet<Artifact> getNonRecursiveCompileTimeJarsFromCollection(
      Iterable<? extends TransitiveInfoCollection> deps) {
    JavaCompilationArgs.Builder builder = JavaCompilationArgs.builder();
    builder.addTransitiveTargets(deps, /*recursive=*/false);
    return builder.build().getCompileTimeJars();
  }

  private NestedSet<Artifact> getNonRecursiveCompileTimeJarsFromProvider(
      Iterable<? extends SourcesJavaCompilationArgsProvider> deps, boolean isNeverLink) {
    return JavaCompilationArgs.builder()
        .addSourcesTransitiveCompilationArgs(deps, false,
            isNeverLink ? ClasspathType.COMPILE_ONLY : ClasspathType.BOTH)
        .build().getCompileTimeJars();
  }

  static void addDependencyArtifactsToAttributes(
      JavaTargetAttributes.Builder attributes,
      Iterable<? extends JavaCompilationArgsProvider> deps) {
    NestedSetBuilder<Artifact> result = NestedSetBuilder.stableOrder();
    for (JavaCompilationArgsProvider provider : deps) {
      result.addTransitive(provider.getCompileTimeJavaDependencyArtifacts());
    }
    attributes.addCompileTimeDependencyArtifacts(result.build());
  }

  /**
   * Adds the compile time and runtime Java libraries in the transitive closure
   * of the deps to the attributes.
   *
   * @param deps the dependencies to be included as roots of the transitive
   *        closure
   */
  public void addLibrariesToAttributes(Iterable<? extends TransitiveInfoCollection> deps) {
    // Enforcing strict Java dependencies: when the --strict_java_deps flag is
    // WARN or ERROR, or is DEFAULT and strict_java_deps attribute is unset,
    // we use a stricter javac compiler to perform direct deps checks.
    attributes.setStrictJavaDeps(getStrictJavaDeps());
    addLibrariesToAttributesInternal(deps);

    JavaClasspathMode classpathMode = getJavaConfiguration().getReduceJavaClasspath();
    if (isStrict() && classpathMode != JavaClasspathMode.OFF) {
      addDependencyArtifactsToAttributes(
          attributes, AnalysisUtils.getProviders(deps, JavaCompilationArgsProvider.class));
    }
  }

  public void addProvidersToAttributes(Iterable<? extends SourcesJavaCompilationArgsProvider> deps,
      boolean isNeverLink) {
    // see addLibrariesToAttributes() for explanation
    attributes.setStrictJavaDeps(getStrictJavaDeps());
    addProvidersToAttributesInternal(deps, isNeverLink);
  }

  /**
   * Determines whether to enable strict_java_deps.
   *
   * @return filtered command line flag value, defaulting to ERROR
   */
  public StrictDepsMode getStrictJavaDeps() {
    return getJavaConfiguration().getFilteredStrictJavaDeps();
  }

  /**
   * Gets the value of the "javacopts" attribute combining them with the
   * default options. If the current rule has no javacopts attribute, this
   * method only returns the default options.
   */
  @VisibleForTesting
  public ImmutableList<String> getJavacOpts() {
    return customJavacOpts;
  }

  /**
   * Obtains the standard list of javac opts needed to build {@code rule}.
   *
   * This method must only be called during initialization.
   *
   * @param ruleContext a rule context
   * @return a list of options to provide to javac
   */
  private static ImmutableList<String> getDefaultJavacOptsFromRule(RuleContext ruleContext) {
    return ImmutableList.copyOf(Iterables.concat(
        JavaToolchainProvider.fromRuleContext(ruleContext).getJavacOptions(),
        ruleContext.getTokenizedStringListAttr("javacopts")));
  }

  public void setTranslations(Collection<Artifact> translations) {
    Preconditions.checkArgument(!translationsFrozen);
    this.translations.addAll(translations);
  }

  private ImmutableList<Artifact> getTranslations() {
    translationsFrozen = true;
    return ImmutableList.copyOf(translations);
  }
}
