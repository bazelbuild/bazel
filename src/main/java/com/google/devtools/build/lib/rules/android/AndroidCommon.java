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

import static com.google.devtools.build.lib.analysis.config.BuildConfiguration.StrictDepsMode.DEFAULT;
import static com.google.devtools.build.lib.analysis.config.BuildConfiguration.StrictDepsMode.ERROR;
import static com.google.devtools.build.lib.analysis.config.BuildConfiguration.StrictDepsMode.STRICT;

import com.google.common.base.Optional;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.analysis.AnalysisUtils;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.OutputGroupProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.StrictDepsMode;
import com.google.devtools.build.lib.collect.IterablesChain;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.rules.android.AndroidResourcesProvider.ResourceContainer;
import com.google.devtools.build.lib.rules.android.AndroidResourcesProvider.ResourceType;
import com.google.devtools.build.lib.rules.android.AndroidRuleClasses.MultidexMode;
import com.google.devtools.build.lib.rules.cpp.CcLinkParams;
import com.google.devtools.build.lib.rules.cpp.CcLinkParamsProvider;
import com.google.devtools.build.lib.rules.cpp.CcLinkParamsStore;
import com.google.devtools.build.lib.rules.cpp.CcNativeLibraryProvider;
import com.google.devtools.build.lib.rules.cpp.CppFileTypes;
import com.google.devtools.build.lib.rules.cpp.LinkerInput;
import com.google.devtools.build.lib.rules.java.ClasspathConfiguredFragment;
import com.google.devtools.build.lib.rules.java.JavaCcLinkParamsProvider;
import com.google.devtools.build.lib.rules.java.JavaCommon;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgs;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgs.ClasspathType;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider;
import com.google.devtools.build.lib.rules.java.JavaCompilationArtifacts;
import com.google.devtools.build.lib.rules.java.JavaCompilationHelper;
import com.google.devtools.build.lib.rules.java.JavaNativeLibraryProvider;
import com.google.devtools.build.lib.rules.java.JavaRuleOutputJarsProvider;
import com.google.devtools.build.lib.rules.java.JavaRuntimeJarProvider;
import com.google.devtools.build.lib.rules.java.JavaSemantics;
import com.google.devtools.build.lib.rules.java.JavaSourceJarsProvider;
import com.google.devtools.build.lib.rules.java.JavaTargetAttributes;
import com.google.devtools.build.lib.rules.java.JavaUtil;
import com.google.devtools.build.lib.rules.java.SourcesJavaCompilationArgsProvider;
import com.google.devtools.build.lib.rules.test.InstrumentedFilesCollector.InstrumentationSpec;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Set;

import javax.annotation.Nullable;

/**
 * A helper class for android rules.
 *
 * <p>Helps create the java compilation as well as handling the exporting of the java compilation
 * artifacts to the other rules.
 */
public class AndroidCommon {

  public static final InstrumentationSpec ANDROID_COLLECTION_SPEC = JavaCommon.JAVA_COLLECTION_SPEC
      .withDependencyAttributes("deps", "data", "exports", "runtime_deps", "binary_under_test");

  public static final Set<String> TRANSITIVE_ATTRIBUTES = ImmutableSet.of(
      "deps",
      "exports"
  );

  public static final <T extends TransitiveInfoProvider> Iterable<T> getTransitivePrerequisites(
      RuleContext ruleContext, Mode mode, final Class<T> classType) {
    IterablesChain.Builder<T> builder = IterablesChain.builder();
    for (String attr : TRANSITIVE_ATTRIBUTES) {
      if (ruleContext.getAttribute(attr) != null) {
        builder.add(ruleContext.getPrerequisites(attr, mode, classType));
      }
    }
    return builder.build();
  }

  public static final Iterable<TransitiveInfoCollection> collectTransitiveInfo(
      RuleContext ruleContext, Mode mode) {
    ImmutableList.Builder<TransitiveInfoCollection> builder = ImmutableList.builder();
    for (String attr : TRANSITIVE_ATTRIBUTES) {
      if (ruleContext.getAttribute(attr) != null) {
        builder.addAll(ruleContext.getPrerequisites(attr, mode));
      }
    }
    return builder.build();
  }

  private final RuleContext ruleContext;
  private final JavaCommon javaCommon;
  private final boolean asNeverLink;
  private final boolean exportDeps;

  private NestedSet<Artifact> compileTimeDependencyArtifacts;
  private NestedSet<Artifact> filesToBuild;
  private NestedSet<Artifact> transitiveNeverlinkLibraries =
      NestedSetBuilder.emptySet(Order.STABLE_ORDER);
  private Iterable<Artifact> topLevelSourceJars = ImmutableList.of();
  private NestedSet<Artifact> transitiveSourceJars = NestedSetBuilder.emptySet(Order.STABLE_ORDER);
  private JavaCompilationArgs javaCompilationArgs = JavaCompilationArgs.EMPTY_ARGS;
  private JavaCompilationArgs recursiveJavaCompilationArgs = JavaCompilationArgs.EMPTY_ARGS;
  private JackCompilationHelper jackCompilationHelper;
  private Artifact classJar;
  private Artifact iJar;
  private Artifact srcJar;
  private Artifact genClassJar;
  private Artifact genSourceJar;
  private Artifact resourceClassJar;
  private Artifact resourceIJar;
  private Artifact resourceSourceJar;
  private Artifact outputDepsProto;

  private Artifact manifestProtoOutput;
  private AndroidIdlHelper idlHelper;

  public AndroidCommon(JavaCommon javaCommon) {
    this(javaCommon, JavaCommon.isNeverLink(javaCommon.getRuleContext()), false);
  }

  /**
   * Creates a new AndroidCommon.
   * @param common the JavaCommon instance
   * @param asNeverLink Boolean to indicate if this rule should be treated as a compile time dep
   *    by consuming rules.
   * @param exportDeps Boolean to indicate if the dependencies should be treated as "exported" deps.
   */
  public AndroidCommon(JavaCommon common, boolean asNeverLink, boolean exportDeps) {
    this.ruleContext = common.getRuleContext();
    this.asNeverLink = asNeverLink;
    this.exportDeps = exportDeps;
    this.javaCommon = common;
  }

  /**
   * Collects the transitive neverlink dependencies.
   *
   * @param ruleContext the context of the rule neverlink deps are to be computed for
   * @param deps the targets to be treated as dependencies
   * @param runtimeJars the runtime jars produced by the rule (non-transitive)
   *
   * @return a nested set of the neverlink deps.
   */
  public static NestedSet<Artifact> collectTransitiveNeverlinkLibraries(
      RuleContext ruleContext, Iterable<? extends TransitiveInfoCollection> deps,
      ImmutableList<Artifact> runtimeJars) {
    NestedSetBuilder<Artifact> builder = NestedSetBuilder.naiveLinkOrder();

    for (AndroidNeverLinkLibrariesProvider provider : AnalysisUtils.getProviders(deps,
        AndroidNeverLinkLibrariesProvider.class)) {
      builder.addTransitive(provider.getTransitiveNeverLinkLibraries());
    }

    if (JavaCommon.isNeverLink(ruleContext)) {
      builder.addAll(runtimeJars);
      for (JavaCompilationArgsProvider provider : AnalysisUtils.getProviders(
          deps, JavaCompilationArgsProvider.class)) {
        builder.addTransitive(provider.getRecursiveJavaCompilationArgs().getRuntimeJars());
      }
    }

    return builder.build();
  }

  /**
   * Creates an action that converts {@code jarToDex} to a dex file. The output will be stored in
   * the {@link com.google.devtools.build.lib.actions.Artifact} {@code dxJar}.
   */
  public static void createDexAction(
      RuleContext ruleContext,
      Artifact jarToDex, Artifact classesDex, List<String> dexOptions, boolean multidex,
      Artifact mainDexList) {
    List<String> args = new ArrayList<>();
    args.add("--dex");
    // Add --no-locals to coverage builds. Otherwise local variable debug information is not
    // preserved, which leads to runtime errors.
    if (ruleContext.getConfiguration().isCodeCoverageEnabled()) {
      args.add("--no-locals");
    }

    // Multithreaded dex does not work when using --multi-dex.
    if (!multidex) {
      // Multithreaded dex tends to run faster, but only up to about 5 threads (at which point the
      // law of diminishing returns kicks in). This was determined experimentally, with 5-thread dex
      // performing about 25% faster than 1-thread dex.
      args.add("--num-threads=5");
    }

    args.addAll(dexOptions);
    if (multidex) {
      args.add("--multi-dex");
      if (mainDexList != null) {
        args.add("--main-dex-list=" + mainDexList.getExecPathString());
      }
    }
    args.add("--output=" + classesDex.getExecPathString());
    args.add(jarToDex.getExecPathString());

    SpawnAction.Builder builder = new SpawnAction.Builder()
        .setExecutable(AndroidSdkProvider.fromRuleContext(ruleContext).getDx())
        .addInput(jarToDex)
        .addOutput(classesDex)
        .addArguments(args)
        .setProgressMessage("Converting " + jarToDex.getExecPathString() + " to dex format")
        .setMnemonic("AndroidDexer")
        .setResources(ResourceSet.createWithRamCpuIo(4096.0, 5.0, 0.0));
    if (mainDexList != null) {
      builder.addInput(mainDexList);
    }
    ruleContext.registerAction(builder.build(ruleContext));
  }

  public static AndroidIdeInfoProvider createAndroidIdeInfoProvider(
      RuleContext ruleContext,
      AndroidSemantics semantics,
      AndroidIdlHelper idlHelper,
      ResourceApk resourceApk,
      Artifact zipAlignedApk,
      Iterable<Artifact> apksUnderTest) {
    AndroidIdeInfoProvider.Builder ideInfoProviderBuilder =
        new AndroidIdeInfoProvider.Builder()
            .setIdlClassJar(idlHelper.getIdlClassJar())
            .setIdlSourceJar(idlHelper.getIdlSourceJar())
            .addIdlParcelables(idlHelper.getIdlParcelables())
            .addIdlSrcs(idlHelper.getIdlSources())
            .addIdlGeneratedJavaFiles(idlHelper.getIdlGeneratedJavaSources())
            .addAllApksUnderTest(apksUnderTest);

    if (zipAlignedApk != null) {
      ideInfoProviderBuilder.setApk(zipAlignedApk);
    }

    // If the rule defines resources, put those in the IDE info. Otherwise, proxy the data coming
    // from the android_resources rule in its direct dependencies, if such a thing exists.
    if (LocalResourceContainer.definesAndroidResources(ruleContext.attributes())) {
      ideInfoProviderBuilder
          .addResourceSources(resourceApk.getPrimaryResource().getArtifacts(ResourceType.RESOURCES))
          .addAssetSources(
              resourceApk.getPrimaryResource().getArtifacts(ResourceType.ASSETS),
              getAssetDir(ruleContext))
          // Sets the possibly merged manifest and the raw manifest.
          .setGeneratedManifest(resourceApk.getPrimaryResource().getManifest())
          .setManifest(ruleContext.getPrerequisiteArtifact("manifest", Mode.TARGET));
    } else {
      semantics.addNonLocalResources(ruleContext, resourceApk, ideInfoProviderBuilder);
    }

    return ideInfoProviderBuilder.build();
  }

  public static String getJavaPackage(RuleContext ruleContext) {
    if (ruleContext.attributes().isAttributeValueExplicitlySpecified("custom_package")) {
      return ruleContext.attributes().get("custom_package", Type.STRING);
    } else {
      PathFragment nameFragment = ruleContext.getRule().getPackage().getNameFragment();
      String packageName = JavaUtil.getJavaFullClassname(nameFragment);
      if (packageName != null) {
        return packageName;
      } else {
        // This is a workaround for libraries that don't follow the standard Bazel package format
        return nameFragment.getPathString().replace('/', '.');
      }
    }
  }

  Artifact compileDexWithJack(
      MultidexMode mode, Optional<Artifact> mainDexList, Collection<Artifact> proguardSpecs) {
    return jackCompilationHelper.compileAsDex(mode, mainDexList, proguardSpecs);
  }

  private void compileResources(
      JavaSemantics javaSemantics,
      Artifact resourcesJar,
      JavaCompilationArtifacts.Builder artifactsBuilder,
      JavaTargetAttributes.Builder attributes,
      NestedSetBuilder<Artifact> filesBuilder) throws InterruptedException {
    compileResourceJar(javaSemantics, resourcesJar);
    // Add the compiled resource jar to the classpath of the main compilation.
    attributes.addDirectJars(ImmutableList.of(resourceClassJar));
    attributes.addDirectCompileTimeClassPathEntries(ImmutableList.of(resourceClassJar));
    // Add the compiled resource jar to the classpath of consuming targets.
    artifactsBuilder.addCompileTimeJar(resourceClassJar);
    // Combined resource constants needs to come even before our own classes that may contain
    // local resource constants.
    artifactsBuilder.addRuntimeJar(resourceClassJar);
    // Add the compiled resource jar as a declared output of the rule.
    filesBuilder.add(resourceSourceJar);
    filesBuilder.add(resourceClassJar);
  }

  private void compileResourceJar(JavaSemantics javaSemantics, Artifact resourcesJar)
      throws InterruptedException {
    resourceSourceJar = ruleContext.getImplicitOutputArtifact(
        AndroidRuleClasses.ANDROID_RESOURCES_SOURCE_JAR);
    resourceClassJar = ruleContext.getImplicitOutputArtifact(
        AndroidRuleClasses.ANDROID_RESOURCES_CLASS_JAR);

    JavaCompilationArtifacts.Builder javaArtifactsBuilder = new JavaCompilationArtifacts.Builder();
    JavaTargetAttributes.Builder javacAttributes = new JavaTargetAttributes.Builder(javaSemantics)
        .addSourceJar(resourcesJar);
    JavaCompilationHelper javacHelper = new JavaCompilationHelper(
        ruleContext, javaSemantics, getJavacOpts(), javacAttributes);

    Artifact outputDepsProto =
        javacHelper.createOutputDepsProtoArtifact(resourceClassJar, javaArtifactsBuilder);
    javacHelper.createCompileActionWithInstrumentation(
        resourceClassJar,
        null /* manifestProtoOutput */,
        null /* genSourceJar */,
        outputDepsProto,
        javaArtifactsBuilder);
    javacHelper.createSourceJarAction(resourceSourceJar, null);
    resourceIJar = javacHelper.createCompileTimeJarAction(resourceClassJar, javaArtifactsBuilder);
  }

  private void createJarJarActions(
      JavaTargetAttributes.Builder attributes,
      Iterable<ResourceContainer> resourceContainers,
      String originalPackage,
      Artifact binaryResourcesJar) {
    // Now use jarjar for the rest of the resources. We need to make a copy
    // of the final generated resources for each of the targets included in
    // the transitive closure of this binary.
    for (ResourceContainer otherContainer : resourceContainers) {
      if (otherContainer.getLabel().equals(ruleContext.getLabel())) {
        continue;
      }

      Artifact resourcesJar = createResourceJarArtifact(ruleContext, otherContainer, ".jar");
      // combined resource constants copy needs to come before library classes that may contain
      // their local resource constants
      attributes.addRuntimeClassPathEntry(resourcesJar);

      Artifact jarJarRuleFile = createResourceJarArtifact(
          ruleContext, otherContainer, ".jar_jarjar_rules.txt");

      String jarJarRule = String.format("rule %s.* %s.@1",
          originalPackage, otherContainer.getJavaPackage());
      ruleContext.registerAction(new FileWriteAction(
          ruleContext.getActionOwner(), jarJarRuleFile, jarJarRule, false));

      FilesToRunProvider jarjar =
          ruleContext.getExecutablePrerequisite("$jarjar_bin", Mode.HOST);

      ruleContext.registerAction(new SpawnAction.Builder()
          .setExecutable(jarjar)
          .addArgument("process")
          .addInputArgument(jarJarRuleFile)
          .addInputArgument(binaryResourcesJar)
          .addOutputArgument(resourcesJar)
          .setProgressMessage("Repackaging jar")
          .setMnemonic("AndroidRepackageJar")
          .build(ruleContext));
    }
  }

  private static Artifact createResourceJarArtifact(RuleContext ruleContext,
      ResourceContainer container, String fileNameSuffix) {

    String artifactName = container.getLabel().getName() + fileNameSuffix;

    // Since the Java sources are generated by combining all resources with the
    // ones included in the binary, the path of the artifact has to be unique
    // per binary and per library (not only per library).
    Artifact artifact = ruleContext.getUniqueDirectoryArtifact("resource_jars",
        container.getLabel().getPackageIdentifier().getPathFragment().getRelative(artifactName),
        ruleContext.getBinOrGenfilesDirectory());
    return artifact;
  }

  public JavaTargetAttributes init(
      JavaSemantics javaSemantics, AndroidSemantics androidSemantics,
      ResourceApk resourceApk,
      boolean addCoverageSupport, boolean collectJavaCompilationArgs) throws InterruptedException {

    classJar = ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_LIBRARY_CLASS_JAR);
    idlHelper = new AndroidIdlHelper(ruleContext, classJar);

    javaCommon.initializeJavacOpts(androidSemantics.getJavacArguments());
    JavaTargetAttributes.Builder attributes = javaCommon
        .initCommon(idlHelper.getIdlGeneratedJavaSources())
        .setBootClassPath(ImmutableList.of(
            AndroidSdkProvider.fromRuleContext(ruleContext).getAndroidJar()));

    JavaCompilationArtifacts.Builder artifactsBuilder = new JavaCompilationArtifacts.Builder();
    NestedSetBuilder<Artifact> filesBuilder = NestedSetBuilder.<Artifact>stableOrder();

    Artifact resourcesJar = resourceApk.getResourceJavaSrcJar();
    if (resourcesJar != null) {
      filesBuilder.add(resourcesJar);
      compileResources(javaSemantics, resourcesJar, artifactsBuilder, attributes, filesBuilder);
      if (resourceApk.isLegacy()) {
        // Repackages the R.java for each dependency package and places the resultant jars before
        // the dependency libraries to ensure that the generated resource ids are correct.
        createJarJarActions(attributes, resourceApk.getResourceDependencies().getResources(),
            resourceApk.getPrimaryResource().getJavaPackage(), resourceClassJar);
      }
    }

    if (disallowDepsWithoutSrcs(ruleContext.getRule().getRuleClass())
        && ruleContext.attributes().get("srcs", BuildType.LABEL_LIST).isEmpty()
        && ruleContext.attributes().get("idl_srcs", BuildType.LABEL_LIST).isEmpty()
        && !ruleContext.attributes().get("deps", BuildType.LABEL_LIST).isEmpty()) {
      ruleContext.attributeError("deps", "deps not allowed without srcs; move to exports?");
    }

    JavaCompilationHelper helper = initAttributes(attributes, javaSemantics);
    if (ruleContext.hasErrors()) {
      return null;
    }

    if (addCoverageSupport) {
      androidSemantics.addCoverageSupport(ruleContext, this, javaSemantics, true,
          attributes, artifactsBuilder);
      if (ruleContext.hasErrors()) {
        return null;
      }
    }

    jackCompilationHelper = initJack(helper.getAttributes());
    if (ruleContext.hasErrors()) {
      return null;
    }

    initJava(helper, artifactsBuilder, collectJavaCompilationArgs, filesBuilder);
    if (ruleContext.hasErrors()) {
      return null;
    }
    return helper.getAttributes();
  }

  private boolean disallowDepsWithoutSrcs(String ruleClass) {
    return ruleClass.equals("android_library")
        && !ruleContext.getFragment(AndroidConfiguration.class).allowSrcsLessAndroidLibraryDeps();
  }

  private JavaCompilationHelper initAttributes(
      JavaTargetAttributes.Builder attributes, JavaSemantics semantics) {
    JavaCompilationHelper helper = new JavaCompilationHelper(
        ruleContext, semantics, javaCommon.getJavacOpts(), attributes);
    
    helper.addLibrariesToAttributes(javaCommon.targetsTreatedAsDeps(ClasspathType.COMPILE_ONLY));
    helper.addProvidersToAttributes(
        JavaCommon.compilationArgsFromSources(ruleContext), asNeverLink);
    attributes.setStrictJavaDeps(getStrictAndroidDeps());
    attributes.setRuleKind(ruleContext.getRule().getRuleClass());
    attributes.setTargetLabel(ruleContext.getLabel());

    JavaCommon.validateConstraint(ruleContext, "android",
        javaCommon.targetsTreatedAsDeps(ClasspathType.BOTH));
    ruleContext.checkSrcsSamePackage(true);
    return helper;
  }

  private StrictDepsMode getStrictAndroidDeps() {
    // Get command line strict_android_deps option
    StrictDepsMode strict = ruleContext.getFragment(AndroidConfiguration.class).getStrictDeps();
    // Use option if anything but DEFAULT, which is now equivalent to ERROR.
    return (strict != DEFAULT && strict != STRICT) ? strict : ERROR;
  }

  JackCompilationHelper initJack(JavaTargetAttributes attributes) throws InterruptedException {
    AndroidSdkProvider sdk = AndroidSdkProvider.fromRuleContext(ruleContext);
    return new JackCompilationHelper.Builder()
        // blaze infrastructure
        .setRuleContext(ruleContext)
        // configuration
        .setOutputArtifact(
            ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_LIBRARY_JACK_FILE))
        // tools
        .setAndroidSdk(sdk)
        // sources
        .addJavaSources(attributes.getSourceFiles())
        .addSourceJars(attributes.getSourceJars())
        .addResources(attributes.getResources())
        .addProcessorNames(attributes.getProcessorNames())
        .addProcessorClasspathJars(attributes.getProcessorPath())
        .addExports(JavaCommon.getExports(ruleContext))
        .addClasspathDeps(javaCommon.targetsTreatedAsDeps(ClasspathType.COMPILE_ONLY))
        .addRuntimeDeps(javaCommon.targetsTreatedAsDeps(ClasspathType.RUNTIME_ONLY))
        .build();
  }

  private void initJava(
      JavaCompilationHelper helper,
      JavaCompilationArtifacts.Builder javaArtifactsBuilder,
      boolean collectJavaCompilationArgs,
      NestedSetBuilder<Artifact> filesBuilder) throws InterruptedException {

    JavaTargetAttributes attributes = helper.getAttributes();
    if (ruleContext.hasErrors()) {
      // Avoid leaving filesToBuild set to null, otherwise we'll get a NullPointerException masking
      // the real error.
      filesToBuild = filesBuilder.build();
      return;
    }

    Artifact jar = null;
    if (attributes.hasSourceFiles() || attributes.hasSourceJars() || attributes.hasResources()) {
      // We only want to add a jar to the classpath of a dependent rule if it has content.
      javaArtifactsBuilder.addRuntimeJar(classJar);
      jar = classJar;
    }

    filesBuilder.add(classJar);

    manifestProtoOutput = helper.createManifestProtoOutput(classJar);

    // The gensrc jar is created only if the target uses annotation processing. Otherwise,
    // it is null, and the source jar action will not depend on the compile action.
    if (helper.usesAnnotationProcessing()) {
      genClassJar = helper.createGenJar(classJar);
      genSourceJar = helper.createGensrcJar(classJar);
      helper.createGenJarAction(classJar, manifestProtoOutput, genClassJar);
    }

    srcJar = ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_LIBRARY_SOURCE_JAR);
    helper.createSourceJarAction(srcJar, genSourceJar);

    outputDepsProto = helper.createOutputDepsProtoArtifact(classJar, javaArtifactsBuilder);
    helper.createCompileActionWithInstrumentation(classJar, manifestProtoOutput, genSourceJar,
        outputDepsProto, javaArtifactsBuilder);

    compileTimeDependencyArtifacts = 
        javaCommon.collectCompileTimeDependencyArtifacts(outputDepsProto);
    filesToBuild = filesBuilder.build();

    if ((attributes.hasSourceFiles() || attributes.hasSourceJars()) && jar != null) {
      iJar = helper.createCompileTimeJarAction(jar, javaArtifactsBuilder);
    }
    javaCommon.setJavaCompilationArtifacts(javaArtifactsBuilder.build());

    javaCommon.setClassPathFragment(
        new ClasspathConfiguredFragment(
            javaCommon.getJavaCompilationArtifacts(), attributes, asNeverLink));

    transitiveNeverlinkLibraries = collectTransitiveNeverlinkLibraries(
        ruleContext,
        javaCommon.getDependencies(),
        javaCommon.getJavaCompilationArtifacts().getRuntimeJars());
    topLevelSourceJars = ImmutableList.of(srcJar);
    transitiveSourceJars = javaCommon.collectTransitiveSourceJars(srcJar);

    if (collectJavaCompilationArgs) {
      boolean hasSources = attributes.hasSourceFiles() || attributes.hasSourceJars();
      this.javaCompilationArgs =
          collectJavaCompilationArgs(exportDeps, asNeverLink, hasSources);
      this.recursiveJavaCompilationArgs = collectJavaCompilationArgs(
          true, asNeverLink, /* hasSources */ true);
    }
  }
  
  public RuleConfiguredTargetBuilder addTransitiveInfoProviders(
      RuleConfiguredTargetBuilder builder,
      AndroidSemantics androidSemantics,
      ResourceApk resourceApk,
      Artifact zipAlignedApk,
      Iterable<Artifact> apksUnderTest) {

    javaCommon.addTransitiveInfoProviders(builder, filesToBuild, classJar, ANDROID_COLLECTION_SPEC);
    javaCommon.addGenJarsProvider(builder, genClassJar, genSourceJar);
    idlHelper.addTransitiveInfoProviders(builder, classJar, manifestProtoOutput);

    JavaRuleOutputJarsProvider.Builder outputJarsBuilder = JavaRuleOutputJarsProvider.builder()
        .addOutputJar(classJar, iJar, srcJar)
        .setJdeps(outputDepsProto);
    if (resourceClassJar != null && resourceIJar != null && resourceSourceJar != null) {
      outputJarsBuilder.addOutputJar(resourceClassJar, resourceIJar, resourceSourceJar);
    }

    return builder
        .setFilesToBuild(filesToBuild)
        .add(JavaRuleOutputJarsProvider.class, outputJarsBuilder.build())
        .add(
            JavaRuntimeJarProvider.class,
            new JavaRuntimeJarProvider(javaCommon.getJavaCompilationArtifacts().getRuntimeJars()))
        .add(RunfilesProvider.class, RunfilesProvider.simple(getRunfiles()))
        .add(
            AndroidResourcesProvider.class, resourceApk.toResourceProvider(ruleContext.getLabel()))
        .add(
            AndroidIdeInfoProvider.class,
            createAndroidIdeInfoProvider(ruleContext, androidSemantics, idlHelper,
                resourceApk, zipAlignedApk, apksUnderTest))
        .add(
            JavaCompilationArgsProvider.class,
            new JavaCompilationArgsProvider(
                javaCompilationArgs,
                recursiveJavaCompilationArgs,
                compileTimeDependencyArtifacts,
                NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER)))
        .add(
            JackLibraryProvider.class,
            asNeverLink
                ? jackCompilationHelper.compileAsNeverlinkLibrary()
                : jackCompilationHelper.compileAsLibrary())
        .addOutputGroup(
            OutputGroupProvider.HIDDEN_TOP_LEVEL, collectHiddenTopLevelArtifacts(ruleContext))
        .addOutputGroup(JavaSemantics.SOURCE_JARS_OUTPUT_GROUP, transitiveSourceJars);
  }
  
  private Runfiles getRunfiles() {
    // TODO(bazel-team): why return any Runfiles in the neverlink case?
    if (asNeverLink) {
      return new Runfiles.Builder(ruleContext.getWorkspaceName())
          .addRunfiles(ruleContext, RunfilesProvider.DEFAULT_RUNFILES)
          .build();
    }
    return javaCommon.getRunfiles(asNeverLink);
  }

  public static PathFragment getAssetDir(RuleContext ruleContext) {
    return new PathFragment(ruleContext.attributes().get(
        AndroidResourcesProvider.ResourceType.ASSETS.getAttribute() + "_dir",
        Type.STRING));
  }

  public static NestedSet<LinkerInput> collectTransitiveNativeLibraries(
      Iterable<? extends TransitiveInfoCollection> deps) {
    NestedSetBuilder<LinkerInput> builder = NestedSetBuilder.stableOrder();
    for (TransitiveInfoCollection dep : deps) {
      AndroidNativeLibraryProvider android = dep.getProvider(AndroidNativeLibraryProvider.class);
      if (android != null) {
        builder.addTransitive(android.getTransitiveAndroidNativeLibraries());
        continue;
      }

      JavaNativeLibraryProvider java = dep.getProvider(JavaNativeLibraryProvider.class);
      if (java != null) {
        builder.addTransitive(java.getTransitiveJavaNativeLibraries());
        continue;
      }

      CcNativeLibraryProvider cc = dep.getProvider(CcNativeLibraryProvider.class);
      if (cc != null) {
        for (LinkerInput input : cc.getTransitiveCcNativeLibraries()) {
          Artifact library = input.getOriginalLibraryArtifact();
          String name = library.getFilename();
          if (CppFileTypes.SHARED_LIBRARY.matches(name)
              || CppFileTypes.VERSIONED_SHARED_LIBRARY.matches(name)) {
            builder.add(input);
          }
        }
        continue;
      }
    }

    return builder.build();
  }

  public static AndroidResourcesProvider getAndroidResources(RuleContext ruleContext) {
    if (!ruleContext.attributes().has("resources", BuildType.LABEL)) {
      return null;
    }

    TransitiveInfoCollection prerequisite = ruleContext.getPrerequisite("resources", Mode.TARGET);
    if (prerequisite == null) {
      return null;
    }

    return prerequisite.getProvider(AndroidResourcesProvider.class);
  }

  /**
   * Collects Java compilation arguments for this target.
   *
   * @param recursive Whether to scan dependencies recursively.
   * @param isNeverLink Whether the target has the 'neverlink' attr.
   * @param hasSrcs If false, deps are exported (deprecated behaviour)
   */
  private JavaCompilationArgs collectJavaCompilationArgs(boolean recursive, boolean isNeverLink,
      boolean hasSrcs) {
    Iterable<SourcesJavaCompilationArgsProvider> fromSrcs = 
        ImmutableList.<SourcesJavaCompilationArgsProvider> of();
    return javaCommon.collectJavaCompilationArgs(recursive, isNeverLink, fromSrcs, !hasSrcs);
  }

  public ImmutableList<String> getJavacOpts() {
    return javaCommon.getJavacOpts();
  }

  public Artifact getGenClassJar() {
    return genClassJar;
  }

  @Nullable public Artifact getGenSourceJar() {
    return genSourceJar;
  }

  public ImmutableList<Artifact> getRuntimeJars() {
    return javaCommon.getJavaCompilationArtifacts().getRuntimeJars();
  }

  public Artifact getInstrumentedJar() {
    return javaCommon.getJavaCompilationArtifacts().getInstrumentedJar();
  }

  public NestedSet<Artifact> getTransitiveNeverLinkLibraries() {
    return transitiveNeverlinkLibraries;
  }

  public Iterable<Artifact> getTopLevelSourceJars() {
    return topLevelSourceJars;
  }

  public NestedSet<Artifact> getTransitiveSourceJars() {
    return transitiveSourceJars;
  }

  public JavaSourceJarsProvider getJavaSourceJarsProvider() {
    return new JavaSourceJarsProvider(getTransitiveSourceJars(), getTopLevelSourceJars());
  }

  public boolean isNeverLink() {
    return asNeverLink;
  }

  public CcLinkParamsStore getCcLinkParamsStore() {
    return getCcLinkParamsStore(javaCommon.targetsTreatedAsDeps(ClasspathType.BOTH));
  }

  public static CcLinkParamsStore getCcLinkParamsStore(
      final Iterable<? extends TransitiveInfoCollection> deps) {
    return new CcLinkParamsStore() {
      @Override
      protected void collect(CcLinkParams.Builder builder, boolean linkingStatically,
                             boolean linkShared) {
        builder.addTransitiveTargets(deps,
            // Link in Java-specific C++ code in the transitive closure
            JavaCcLinkParamsProvider.TO_LINK_PARAMS,
            // Link in Android-specific C++ code (e.g., android_libraries) in the transitive closure
            AndroidCcLinkParamsProvider.TO_LINK_PARAMS,
            // Link in non-language-specific C++ code in the transitive closure
            CcLinkParamsProvider.TO_LINK_PARAMS);
      }
    };
  }

  private NestedSet<Artifact> collectHiddenTopLevelArtifacts(RuleContext ruleContext) {
    NestedSetBuilder<Artifact> builder = NestedSetBuilder.stableOrder();
    for (OutputGroupProvider provider :
        getTransitivePrerequisites(ruleContext, Mode.TARGET, OutputGroupProvider.class)) {
      builder.addTransitive(provider.getOutputGroup(OutputGroupProvider.HIDDEN_TOP_LEVEL));
    }
    return builder.build();
  }
}
