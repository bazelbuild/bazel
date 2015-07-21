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

import static com.google.devtools.build.lib.analysis.config.BuildConfiguration.StrictDepsMode.DEFAULT;
import static com.google.devtools.build.lib.analysis.config.BuildConfiguration.StrictDepsMode.ERROR;
import static com.google.devtools.build.lib.analysis.config.BuildConfiguration.StrictDepsMode.STRICT;

import com.google.common.base.Optional;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MiddlemanFactory;
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
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.StrictDepsMode;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction.SafeImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.rules.android.AndroidResourcesProvider.ResourceContainer;
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
import com.google.devtools.build.lib.rules.java.JavaRuntimeJarProvider;
import com.google.devtools.build.lib.rules.java.JavaSemantics;
import com.google.devtools.build.lib.rules.java.JavaSourceJarsProvider;
import com.google.devtools.build.lib.rules.java.JavaTargetAttributes;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import javax.annotation.Nullable;

/**
 * A helper class for android rules.
 *
 * <p>Helps create the java compilation as well as handling the exporting of the java compilation
 * artifacts to the other rules.
 */
public class AndroidCommon {
  private final RuleContext ruleContext;
  private final JavaCommon javaCommon;

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
  private Artifact srcJar;
  private Artifact genJar;
  private Artifact gensrcJar;

  private Collection<Artifact> idls;
  private AndroidIdlProvider transitiveIdlImportData;
  private NestedSet<ResourceContainer> transitiveResources;
  private Map<Artifact, Artifact> translatedIdlSources = ImmutableMap.of();
  private boolean asNeverLink;
  private boolean exportDeps;

  public AndroidCommon(RuleContext ruleContext, JavaCommon javaCommon) {
    this.ruleContext = ruleContext;
    this.javaCommon = javaCommon;
    this.asNeverLink = JavaCommon.isNeverLink(ruleContext);
  }

  /**
   * Creates a new AndroidCommon.
   * @param ruleContext The rule context associated with this instance.
   * @param common the JavaCommon instance
   * @param asNeverLink Boolean to indicate if this rule should be treated as a compile time dep
   *    by consuming rules.
   * @param exportDeps Boolean to indicate if the dependencies should be treated as "exported" deps.
   */
  public AndroidCommon(
      RuleContext ruleContext, JavaCommon common, boolean asNeverLink, boolean exportDeps) {
    this.ruleContext = ruleContext;
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

  Artifact compileDexWithJack(
      MultidexMode mode, Optional<Artifact> mainDexList, Collection<Artifact> proguardSpecs) {
    return jackCompilationHelper.compileAsDex(mode, mainDexList, proguardSpecs);
  }

  private void compileResources(
      JavaSemantics javaSemantics,
      JavaCompilationArtifacts.Builder artifactsBuilder,
      JavaTargetAttributes.Builder attributes,
      NestedSet<ResourceContainer> resourceContainers,
      ResourceContainer updatedResources) {
      Artifact binaryResourcesJar =
          ruleContext.getImplicitOutputArtifact(JavaSemantics.JAVA_BINARY_CLASS_JAR);
      compileResourceJar(javaSemantics, binaryResourcesJar, updatedResources.getJavaSourceJar());
      // combined resource constants needs to come even before own classes that may contain
      // local resource constants
      artifactsBuilder.addRuntimeJar(binaryResourcesJar);
      // Repackages the R.java for each dependency package and places the resultant jars
      // before the dependency libraries to ensure that the generated resource ids are
      // correct.
      createJarJarActions(attributes, resourceContainers,
          updatedResources.getJavaPackage(),
          binaryResourcesJar);
  }

  private void compileResourceJar(
      JavaSemantics javaSemantics, Artifact binaryResourcesJar, Artifact javaSourceJar) {
      JavaCompilationArtifacts.Builder javaArtifactsBuilder =
          new JavaCompilationArtifacts.Builder();
      JavaTargetAttributes.Builder javacJarAttributes =
          new JavaTargetAttributes.Builder(javaSemantics);
      javacJarAttributes.addSourceJar(javaSourceJar);
      JavaCompilationHelper javacHelper = new JavaCompilationHelper(
          ruleContext, javaSemantics, getJavacOpts(), javacJarAttributes);
      Artifact outputDepsProto =
          javacHelper.createOutputDepsProtoArtifact(binaryResourcesJar, javaArtifactsBuilder);

      javacHelper.createCompileActionWithInstrumentation(
          binaryResourcesJar,
          null /* manifestProtoOutput */,
          null /* gensrcJar */,
          outputDepsProto,
          javaArtifactsBuilder);
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
    PathFragment resourceJarsPathFragment = ruleContext.getUniqueDirectory("resource_jars");
    PathFragment artifactPathFragment = resourceJarsPathFragment.getRelative(
        container.getLabel().getPackageFragment().getRelative(artifactName));

    Artifact artifact = ruleContext.getAnalysisEnvironment()
        .getDerivedArtifact(artifactPathFragment, ruleContext.getBinOrGenfilesDirectory());
    return artifact;
  }

  public JavaTargetAttributes init(
      JavaSemantics javaSemantics, AndroidSemantics androidSemantics,
      ResourceApk resourceApk, AndroidIdlProvider transitiveIdlImportData,
      boolean addCoverageSupport, boolean collectJavaCompilationArgs,
      SafeImplicitOutputsFunction genClassJarImplicitOutput) {
    ImmutableList<Artifact> extraSources =
        resourceApk.isLegacy() || resourceApk.getResourceJavaSrcJar() == null
            ? ImmutableList.<Artifact>of()
            : ImmutableList.of(resourceApk.getResourceJavaSrcJar());
    JavaTargetAttributes.Builder attributes = init(
        androidSemantics,
        transitiveIdlImportData,
        resourceApk.getTransitiveResources(),
        extraSources);
    JavaCompilationArtifacts.Builder artifactsBuilder = new JavaCompilationArtifacts.Builder();
    if (resourceApk.isLegacy()) {
      compileResources(javaSemantics, artifactsBuilder, attributes,
          resourceApk.getTransitiveResources(), resourceApk.getPrimaryResource());
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

    jackCompilationHelper = initJack(helper.getAttributes(), javaSemantics);
    if (ruleContext.hasErrors()) {
      return null;
    }

    initJava(
        helper, artifactsBuilder, collectJavaCompilationArgs, resourceApk.getResourceJavaSrcJar(),
        genClassJarImplicitOutput);
    if (ruleContext.hasErrors()) {
      return null;
    }
    return helper.getAttributes();
  }

  private JavaTargetAttributes.Builder init(
      AndroidSemantics androidSemantics,
      AndroidIdlProvider transitiveIdlImportData,
      NestedSet<AndroidResourcesProvider.ResourceContainer> transitiveResources,
      Collection<Artifact> extraArtifacts) {
    this.transitiveIdlImportData = transitiveIdlImportData;
    this.transitiveResources = transitiveResources;

    ImmutableList.Builder<Artifact> extraSrcsBuilder =
        new ImmutableList.Builder<Artifact>().addAll(extraArtifacts);

    idls = getIdlSrcs(ruleContext);
    if (!idls.isEmpty() && !ruleContext.hasErrors()) {
      translatedIdlSources = generateTranslatedIdlArtifacts(ruleContext, idls);
    }

    javaCommon.initializeJavacOpts(androidSemantics.getJavacArguments());
    JavaTargetAttributes.Builder attributes = javaCommon.initCommon(
        extraSrcsBuilder.addAll(translatedIdlSources.values()).build());

    attributes.setBootClassPath(ImmutableList.of(
        AndroidSdkProvider.fromRuleContext(ruleContext).getAndroidJar()));
    return attributes;
  }

  private JavaCompilationHelper initAttributes(
      JavaTargetAttributes.Builder attributes, JavaSemantics semantics) {
    JavaCompilationHelper helper = new JavaCompilationHelper(
        ruleContext, semantics, javaCommon.getJavacOpts(), attributes);
    Iterable<? extends TransitiveInfoCollection> deps =
        javaCommon.targetsTreatedAsDeps(ClasspathType.BOTH);
    helper.addLibrariesToAttributes(deps);
    helper.addProvidersToAttributes(javaCommon.compilationArgsFromSources(), asNeverLink);
    attributes.setStrictJavaDeps(getStrictAndroidDeps());
    attributes.setRuleKind(ruleContext.getRule().getRuleClass());
    attributes.setTargetLabel(ruleContext.getLabel());

    JavaCommon.validateConstraint(ruleContext, "android", deps);
    ruleContext.checkSrcsSamePackage(true);
    return helper;
  }

  private StrictDepsMode getStrictAndroidDeps() {
    // Get command line strict_android_deps option
    StrictDepsMode strict = ruleContext.getFragment(AndroidConfiguration.class).getStrictDeps();
    // Use option if anything but DEFAULT, which is now equivalent to ERROR.
    return (strict != DEFAULT && strict != STRICT) ? strict : ERROR;
  }

  JackCompilationHelper initJack(JavaTargetAttributes attributes, JavaSemantics javaSemantics) {
    Map<PathFragment, Artifact> resourcesMap = new LinkedHashMap<>();
    for (Artifact resource : attributes.getResources()) {
      resourcesMap.put(javaSemantics.getJavaResourcePath(resource.getRootRelativePath()), resource);
    }
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
        .addCompiledJars(attributes.getJarFiles())
        .addResources(ImmutableMap.copyOf(resourcesMap))
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
      @Nullable Artifact additionalSourceJar,
      SafeImplicitOutputsFunction genClassJarImplicitOutput) {
    NestedSetBuilder<Artifact> filesBuilder = NestedSetBuilder.<Artifact>stableOrder();
    if (additionalSourceJar != null) {
      filesBuilder.add(additionalSourceJar);
    }

    JavaTargetAttributes attributes = helper.getAttributes();
    if (ruleContext.hasErrors()) {
      // Avoid leaving filesToBuild set to null, otherwise we'll get a NullPointerException masking
      // the real error.
      filesToBuild = filesBuilder.build();
      return;
    }

    if (attributes.hasJarFiles()) {
      // This rule is repackaging some source jars as a java library

      javaArtifactsBuilder.addRuntimeJars(attributes.getJarFiles());
      javaArtifactsBuilder.addCompileTimeJars(attributes.getCompileTimeJarFiles());

      filesBuilder.addAll(attributes.getJarFiles());
    }

    Artifact jar = null;
    classJar = ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_LIBRARY_CLASS_JAR);
    if (attributes.hasSourceFiles() || attributes.hasSourceJars() || attributes.hasResources()) {
      // We only want to add a jar to the classpath of a dependent rule if it has content.
      javaArtifactsBuilder.addRuntimeJar(classJar);
      jar = classJar;
    }

    filesBuilder.add(classJar);

    // The gensrc jar is created only if the target uses annotation processing. Otherwise,
    // it is null, and the source jar action will not depend on the compile action.
    gensrcJar = helper.createGensrcJar(classJar);
    Artifact manifestProtoOutput = helper.createManifestProtoOutput(classJar);

    // AndroidBinary will pass its -gen.jar output, and AndroidLibrary will pass its own.
    genJar = ruleContext.getImplicitOutputArtifact(genClassJarImplicitOutput);
    helper.createGenJarAction(classJar, manifestProtoOutput, genJar);

    srcJar = ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_LIBRARY_SOURCE_JAR);
    helper.createSourceJarAction(srcJar, gensrcJar);

    NestedSetBuilder<Artifact> compileTimeDependenciesBuilder = NestedSetBuilder.stableOrder();
    Artifact outputDepsProto = helper.createOutputDepsProtoArtifact(classJar, javaArtifactsBuilder);
    if (outputDepsProto != null) {
      compileTimeDependenciesBuilder.add(outputDepsProto);
    }
    helper.createCompileActionWithInstrumentation(classJar, manifestProtoOutput, gensrcJar,
        outputDepsProto, javaArtifactsBuilder);

    compileTimeDependencyArtifacts = compileTimeDependenciesBuilder.build();
    filesToBuild = filesBuilder.build();

    if ((attributes.hasSourceFiles() || attributes.hasSourceJars()) && jar != null) {
      helper.createCompileTimeJarAction(jar, outputDepsProto, javaArtifactsBuilder);
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
      this.javaCompilationArgs = collectJavaCompilationArgs(
          ruleContext, exportDeps, asNeverLink, attributes.hasSourceFiles());
      this.recursiveJavaCompilationArgs = collectJavaCompilationArgs(
          ruleContext, true, asNeverLink, /* hasSources */ true);
    }
  }

  public RuleConfiguredTargetBuilder addTransitiveInfoProviders(
      RuleConfiguredTargetBuilder builder) {
    if (!idls.isEmpty()) {
      generateAndroidIdlActions(
          ruleContext, idls, transitiveIdlImportData, translatedIdlSources);
    }

    Runfiles runfiles = new Runfiles.Builder()
        .addRunfiles(ruleContext, RunfilesProvider.DEFAULT_RUNFILES)
        .build();

    javaCommon.addTransitiveInfoProviders(builder, filesToBuild, classJar);

    return builder
        .setFilesToBuild(filesToBuild)
        .add(
            JavaRuntimeJarProvider.class,
            new JavaRuntimeJarProvider(javaCommon.getJavaCompilationArtifacts().getRuntimeJars()))
        .add(RunfilesProvider.class, RunfilesProvider.simple(runfiles))
        .add(
            AndroidResourcesProvider.class,
            new AndroidResourcesProvider(ruleContext.getLabel(), transitiveResources))
        .add(AndroidIdlProvider.class, transitiveIdlImportData)
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
        .addOutputGroup(JavaSemantics.SOURCE_JARS_OUTPUT_GROUP, transitiveSourceJars)
        .addOutputGroup(JavaSemantics.GENERATED_JARS_OUTPUT_GROUP, genJar);
  }

  public static PathFragment getAssetDir(RuleContext ruleContext) {
    return new PathFragment(ruleContext.attributes().get(
        AndroidResourcesProvider.ResourceType.ASSETS.getAttribute() + "_dir",
        Type.STRING));
  }

  public static ImmutableList<Artifact> getIdlParcelables(RuleContext ruleContext) {
    return ruleContext.getRule().isAttrDefined("idl_parcelables", Type.LABEL_LIST)
        ? ImmutableList.copyOf(ruleContext.getPrerequisiteArtifacts(
            "idl_parcelables", Mode.TARGET).filter(AndroidRuleClasses.ANDROID_IDL).list())
        : ImmutableList.<Artifact>of();
  }

  public static Collection<Artifact> getIdlSrcs(RuleContext ruleContext) {
    if (!ruleContext.getRule().isAttrDefined("idl_srcs", Type.LABEL_LIST)) {
      return ImmutableList.of();
    }
    checkIdlSrcsSamePackage(ruleContext);
    return ruleContext.getPrerequisiteArtifacts(
        "idl_srcs", Mode.TARGET).filter(AndroidRuleClasses.ANDROID_IDL).list();
  }

  public static void checkIdlSrcsSamePackage(RuleContext ruleContext) {
    PathFragment packageName = ruleContext.getLabel().getPackageFragment();
    Collection<Artifact> idls = ruleContext
        .getPrerequisiteArtifacts("idl_srcs", Mode.TARGET)
        .filter(AndroidRuleClasses.ANDROID_IDL)
        .list();
    for (Artifact idl : idls) {
      Label idlLabel = idl.getOwner();
      if (!packageName.equals(idlLabel.getPackageFragment())) {
        ruleContext.attributeError("idl_srcs", "do not import '" + idlLabel + "' directly. "
            + "You should either move the file to this package or depend on "
            + "an appropriate rule there");
      }
    }
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
    TransitiveInfoCollection prerequisite = ruleContext.getPrerequisite("resources", Mode.TARGET);
    return prerequisite != null
        ? prerequisite.getProvider(AndroidResourcesProvider.class)
        : null;
  }

  public static NestedSet<Artifact> getApplicationApks(RuleContext ruleContext) {
    NestedSetBuilder<Artifact> applicationApksBuilder = NestedSetBuilder.stableOrder();
    for (ApkProvider dep : ruleContext.getPrerequisites("deps", Mode.TARGET, ApkProvider.class)) {
      applicationApksBuilder.addTransitive(dep.getTransitiveApks());
    }
    return applicationApksBuilder.build();
  }

  private ImmutableMap<Artifact, Artifact> generateTranslatedIdlArtifacts(
      RuleContext ruleContext, Collection<Artifact> idls) {
    ImmutableMap.Builder<Artifact, Artifact> outputJavaSources = ImmutableMap.builder();
    PathFragment rulePackage = ruleContext.getRule().getLabel().getPackageFragment();
    String ruleName = ruleContext.getRule().getName();
    // for each aidl file use aggregated preprocessed files to generate Java code
    for (Artifact idl : idls) {
      // Reconstruct the package tree under <rule>_aidl to avoid a name conflict
      // if the same AIDL files are used in multiple targets.
      PathFragment javaOutputPath = FileSystemUtils.replaceExtension(
          rulePackage.getRelative(ruleName + "_aidl").getRelative(idl.getRootRelativePath()),
          ".java");
      Artifact output = ruleContext.getAnalysisEnvironment().getDerivedArtifact(
          javaOutputPath, ruleContext.getConfiguration().getGenfilesDirectory());
      outputJavaSources.put(idl, output);
    }
    return outputJavaSources.build();
  }

  private JavaCompilationArgs collectJavaCompilationArgs(RuleContext ruleContext,
      boolean recursive, boolean neverLink, boolean hasSrcs) {
    JavaCompilationArgs.Builder builder = JavaCompilationArgs.builder()
        .merge(javaCommon.getJavaCompilationArtifacts(), neverLink);
    if (recursive || !hasSrcs) {
      builder.addTransitiveTargets(ruleContext.getPrerequisites("deps", Mode.TARGET), recursive,
          neverLink ? ClasspathType.COMPILE_ONLY : ClasspathType.BOTH);
    }
    return builder.build();
  }

  private void generateAndroidIdlActions(RuleContext ruleContext,
      Collection<Artifact> idls, AndroidIdlProvider transitiveIdlImportData,
      Map<Artifact, Artifact> translatedIdlSources) {
    AndroidSdkProvider sdk = AndroidSdkProvider.fromRuleContext(ruleContext);
    Set<Artifact> preprocessedIdls = new LinkedHashSet<>();
    List<String> preprocessedArgs = new ArrayList<>();

    // add imports
    for (String idlImport : transitiveIdlImportData.getTransitiveIdlImportRoots()) {
      preprocessedArgs.add("-I" + idlImport);
    }

    // preprocess each aidl file
    preprocessedArgs.add("-p" + sdk.getFrameworkAidl().getExecPathString());
    PathFragment rulePackage = ruleContext.getRule().getLabel().getPackageFragment();
    String ruleName = ruleContext.getRule().getName();
    for (Artifact idl : idls) {
      // Reconstruct the package tree under <rule>_aidl to avoid a name conflict
      // if the source AIDL files are also generated.
      PathFragment preprocessedPath = rulePackage.getRelative(ruleName + "_aidl")
          .getRelative(idl.getRootRelativePath());
      Artifact preprocessed = ruleContext.getAnalysisEnvironment().getDerivedArtifact(
          preprocessedPath, ruleContext.getConfiguration().getGenfilesDirectory());
      preprocessedIdls.add(preprocessed);
      preprocessedArgs.add("-p" + preprocessed.getExecPathString());

      createAndroidIdlPreprocessAction(ruleContext, idl, preprocessed);
    }

    // aggregate all preprocessed aidl files
    MiddlemanFactory middlemanFactory = ruleContext.getAnalysisEnvironment().getMiddlemanFactory();
    Artifact preprocessedIdlsMiddleman = middlemanFactory.createAggregatingMiddleman(
        ruleContext.getActionOwner(), "AndroidIDLMiddleman", preprocessedIdls,
        ruleContext.getConfiguration().getMiddlemanDirectory());

    for (Artifact idl : translatedIdlSources.keySet()) {
      createAndroidIdlAction(ruleContext, idl,
          transitiveIdlImportData.getTransitiveIdlImports(),
          preprocessedIdlsMiddleman, translatedIdlSources.get(idl), preprocessedArgs);
    }
  }

  private void createAndroidIdlPreprocessAction(RuleContext ruleContext,
      Artifact idl, Artifact preprocessed) {
    AndroidSdkProvider sdk = AndroidSdkProvider.fromRuleContext(ruleContext);
    ruleContext.registerAction(new SpawnAction.Builder()
        .setExecutable(sdk.getAidl())
        // Note the below may be an overapproximation of the actual runfiles, due to "conditional
        // artifacts" (see Runfiles.PruningManifest).
        // TODO(bazel-team): When using getFilesToRun(), the middleman is
        // not expanded. Fix by providing code to expand and use getFilesToRun here.
        .addInput(idl)
        .addOutput(preprocessed)
        .addArgument("--preprocess")
        .addArgument(preprocessed.getExecPathString())
        .addArgument(idl.getExecPathString())
        .setProgressMessage("Android IDL preprocessing")
        .setMnemonic("AndroidIDLPreprocess")
        .build(ruleContext));
  }

  private void createAndroidIdlAction(RuleContext ruleContext,
      Artifact idl, Iterable<Artifact> idlImports, Artifact preprocessedIdls,
      Artifact output, List<String> preprocessedArgs) {
    AndroidSdkProvider sdk = AndroidSdkProvider.fromRuleContext(ruleContext);
    ruleContext.registerAction(new SpawnAction.Builder()
        .setExecutable(sdk.getAidl())
        .addInput(idl)
        .addInputs(idlImports)
        .addInput(preprocessedIdls)
        .addInput(sdk.getFrameworkAidl())
        .addOutput(output)
        .addArgument("-b") // Fail if trying to compile a parcelable.
        .addArguments(preprocessedArgs)
        .addArgument(idl.getExecPathString())
        .addArgument(output.getExecPathString())
        .setProgressMessage("Android IDL generation")
        .setMnemonic("AndroidIDLGnerate")
        .build(ruleContext));
  }

  public ImmutableList<String> getJavacOpts() {
    return javaCommon.getJavacOpts();
  }

  public Artifact getGenJar() {
    return genJar;
  }

  @Nullable public Artifact getGensrcJar() {
    return gensrcJar;
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
        ruleContext.getPrerequisites("deps", Mode.TARGET, OutputGroupProvider.class)) {
      builder.addTransitive(provider.getOutputGroup(OutputGroupProvider.HIDDEN_TOP_LEVEL));
    }
    return builder.build();
  }
}
