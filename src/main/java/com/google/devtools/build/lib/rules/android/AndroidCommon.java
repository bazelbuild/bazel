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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.analysis.AnalysisUtils;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.OutputGroupProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesCollector.InstrumentationSpec;
import com.google.devtools.build.lib.analysis.whitelisting.Whitelist;
import com.google.devtools.build.lib.collect.IterablesChain;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.AggregatingAttributeMapper;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.NativeProvider;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.TriState;
import com.google.devtools.build.lib.rules.android.AndroidConfiguration.AndroidAaptVersion;
import com.google.devtools.build.lib.rules.android.ResourceContainer.ResourceType;
import com.google.devtools.build.lib.rules.cpp.CcLinkParams;
import com.google.devtools.build.lib.rules.cpp.CcLinkParamsInfo;
import com.google.devtools.build.lib.rules.cpp.CcLinkParamsStore;
import com.google.devtools.build.lib.rules.java.ClasspathConfiguredFragment;
import com.google.devtools.build.lib.rules.java.JavaCcLinkParamsProvider;
import com.google.devtools.build.lib.rules.java.JavaCommon;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgs;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgs.ClasspathType;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider;
import com.google.devtools.build.lib.rules.java.JavaCompilationArtifacts;
import com.google.devtools.build.lib.rules.java.JavaCompilationHelper;
import com.google.devtools.build.lib.rules.java.JavaGenJarsProvider;
import com.google.devtools.build.lib.rules.java.JavaInfo;
import com.google.devtools.build.lib.rules.java.JavaRuleOutputJarsProvider;
import com.google.devtools.build.lib.rules.java.JavaRuleOutputJarsProvider.OutputJar;
import com.google.devtools.build.lib.rules.java.JavaRuntimeJarProvider;
import com.google.devtools.build.lib.rules.java.JavaSemantics;
import com.google.devtools.build.lib.rules.java.JavaSkylarkApiProvider;
import com.google.devtools.build.lib.rules.java.JavaSourceJarsProvider;
import com.google.devtools.build.lib.rules.java.JavaTargetAttributes;
import com.google.devtools.build.lib.rules.java.JavaUtil;
import com.google.devtools.build.lib.rules.java.proto.GeneratedExtensionRegistryProvider;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Collection;
import java.util.List;

/**
 * A helper class for android rules.
 *
 * <p>Helps create the java compilation as well as handling the exporting of the java compilation
 * artifacts to the other rules.
 */
public class AndroidCommon {

  public static final InstrumentationSpec ANDROID_COLLECTION_SPEC = JavaCommon.JAVA_COLLECTION_SPEC
      .withDependencyAttributes("deps", "data", "exports", "runtime_deps", "binary_under_test");

  public static final ImmutableSet<String> TRANSITIVE_ATTRIBUTES =
      ImmutableSet.of("deps", "exports");

  public static final <T extends TransitiveInfoProvider> Iterable<T> getTransitivePrerequisites(
      RuleContext ruleContext, Mode mode, final Class<T> classType) {
    IterablesChain.Builder<T> builder = IterablesChain.builder();
    AttributeMap attributes = ruleContext.attributes();
    for (String attr : TRANSITIVE_ATTRIBUTES) {
      if (attributes.has(attr, BuildType.LABEL_LIST)) {
        builder.add(ruleContext.getPrerequisites(attr, mode, classType));
      }
    }
    return builder.build();
  }

  public static final <T extends Info> Iterable<T> getTransitivePrerequisites(
      RuleContext ruleContext, Mode mode, NativeProvider<T> key) {
    IterablesChain.Builder<T> builder = IterablesChain.builder();
    AttributeMap attributes = ruleContext.attributes();
    for (String attr : TRANSITIVE_ATTRIBUTES) {
      if (attributes.has(attr, BuildType.LABEL_LIST)) {
        builder.add(ruleContext.getPrerequisites(attr, mode, key));
      }
    }
    return builder.build();
  }

  public static final String RESOURCES_WHITELIST_NAME = "android_resources";

  private final RuleContext ruleContext;
  private final JavaCommon javaCommon;
  private final boolean asNeverLink;
  private final boolean exportDeps;

  private NestedSet<Artifact> compileTimeDependencyArtifacts;
  private NestedSet<Artifact> filesToBuild;
  private NestedSet<Artifact> transitiveNeverlinkLibraries =
      NestedSetBuilder.emptySet(Order.STABLE_ORDER);
  private JavaCompilationArgs javaCompilationArgs = JavaCompilationArgs.EMPTY_ARGS;
  private JavaCompilationArgs recursiveJavaCompilationArgs = JavaCompilationArgs.EMPTY_ARGS;
  private NestedSet<Artifact> jarsProducedForRuntime;
  private Artifact classJar;
  private Artifact iJar;
  private Artifact srcJar;
  private Artifact genClassJar;
  private Artifact genSourceJar;
  private Artifact resourceClassJar;
  private Artifact resourceSourceJar;
  private Artifact outputDepsProto;
  private GeneratedExtensionRegistryProvider generatedExtensionRegistryProvider;
  private final JavaSourceJarsProvider.Builder javaSourceJarsProviderBuilder =
      JavaSourceJarsProvider.builder();
  private final JavaRuleOutputJarsProvider.Builder javaRuleOutputJarsProviderBuilder =
      JavaRuleOutputJarsProvider.builder();
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
      for (JavaCompilationArgsProvider provider :
          JavaInfo.getProvidersFromListOfTargets(JavaCompilationArgsProvider.class, deps)) {
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
    CustomCommandLine.Builder commandLine = CustomCommandLine.builder();
    commandLine.add("--dex");

    // Multithreaded dex does not work when using --multi-dex.
    if (!multidex) {
      // Multithreaded dex tends to run faster, but only up to about 5 threads (at which point the
      // law of diminishing returns kicks in). This was determined experimentally, with 5-thread dex
      // performing about 25% faster than 1-thread dex.
      commandLine.add("--num-threads=5");
    }

    commandLine.addAll(dexOptions);
    if (multidex) {
      commandLine.add("--multi-dex");
      if (mainDexList != null) {
        commandLine.addPrefixedExecPath("--main-dex-list=", mainDexList);
      }
    }
    commandLine.addPrefixedExecPath("--output=", classesDex);
    commandLine.addExecPath(jarToDex);

    SpawnAction.Builder builder =
        new SpawnAction.Builder()
            .useDefaultShellEnvironment()
            .setExecutable(AndroidSdkProvider.fromRuleContext(ruleContext).getDx())
            .addInput(jarToDex)
            .addOutput(classesDex)
            .setProgressMessage("Converting %s to dex format", jarToDex.getExecPathString())
            .setMnemonic("AndroidDexer")
            .addCommandLine(commandLine.build())
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
      OutputJar resourceJar,
      Artifact aar,
      ResourceApk resourceApk,
      Artifact zipAlignedApk,
      Iterable<Artifact> apksUnderTest,
      NativeLibs nativeLibs) {
    AndroidIdeInfoProvider.Builder ideInfoProviderBuilder =
        new AndroidIdeInfoProvider.Builder()
            .setIdlClassJar(idlHelper.getIdlClassJar())
            .setIdlSourceJar(idlHelper.getIdlSourceJar())
            .setResourceJar(resourceJar)
            .setAar(aar)
            .setNativeLibs(nativeLibs.getMap())
            .addIdlImportRoot(idlHelper.getIdlImportRoot())
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
          .setDefinesAndroidResources(true)
          .addResourceSources(resourceApk.getPrimaryResource().getArtifacts(ResourceType.RESOURCES))
          .addAssetSources(
              resourceApk.getPrimaryResource().getArtifacts(ResourceType.ASSETS),
              getAssetDir(ruleContext))
          // Sets the possibly merged manifest and the raw manifest.
          .setGeneratedManifest(resourceApk.getPrimaryResource().getManifest())
          .setManifest(ruleContext.getPrerequisiteArtifact("manifest", Mode.TARGET))
          .setJavaPackage(getJavaPackage(ruleContext))
          .setResourceApk(resourceApk.getArtifact());
    } else {
      semantics.addNonLocalResources(ruleContext, resourceApk, ideInfoProviderBuilder);
    }

    return ideInfoProviderBuilder.build();
  }

  public static String getJavaPackage(RuleContext ruleContext) {
    AttributeMap attributes = ruleContext.attributes();
    if (attributes.isAttributeValueExplicitlySpecified("custom_package")) {
      return attributes.get("custom_package", Type.STRING);
    }
    return getDefaultJavaPackage(ruleContext.getRule());
  }

  public static Iterable<String> getPossibleJavaPackages(Rule rule) {
    AggregatingAttributeMapper attributes = AggregatingAttributeMapper.of(rule);
    if (attributes.isAttributeValueExplicitlySpecified("custom_package")) {
      return attributes.visitAttribute("custom_package", Type.STRING);
    }
    return ImmutableList.of(getDefaultJavaPackage(rule));
  }

  private static String getDefaultJavaPackage(Rule rule) {
    PathFragment nameFragment = rule.getPackage().getNameFragment();
    String packageName = JavaUtil.getJavaFullClassname(nameFragment);
    if (packageName != null) {
      return packageName;
    } else {
      // This is a workaround for libraries that don't follow the standard Bazel package format
      return nameFragment.getPathString().replace('/', '.');
    }
  }

  static PathFragment getSourceDirectoryRelativePathFromResource(Artifact resource) {
    PathFragment resourceDir = LocalResourceContainer.findResourceDir(resource);
    if (resourceDir == null) {
      return null;
    }
    return trimTo(resource.getRootRelativePath(), resourceDir);
  }

  /**
   * Finds the rightmost occurrence of the needle and returns subfragment of the haystack from
   * left to the end of the occurrence inclusive of the needle.
   *
   * <pre>
   * `Example:
   *   Given the haystack:
   *     res/research/handwriting/res/values/strings.xml
   *   And the needle:
   *     res
   *   Returns:
   *     res/research/handwriting/res
   * </pre>
   */
  static PathFragment trimTo(PathFragment haystack, PathFragment needle) {
    if (needle.equals(PathFragment.EMPTY_FRAGMENT)) {
      return haystack;
    }
    // Compute the overlap offset for duplicated parts of the needle.
    int[] overlap = new int[needle.segmentCount() + 1];
    // Start overlap at -1, as it will cancel out the increment in the search.
    // See http://en.wikipedia.org/wiki/Knuth%E2%80%93Morris%E2%80%93Pratt_algorithm for the
    // details.
    overlap[0] = -1;
    for (int i = 0, j = -1; i < needle.segmentCount(); j++, i++, overlap[i] = j) {
      while (j >= 0 && !needle.getSegment(i).equals(needle.getSegment(j))) {
        // Walk the overlap until the bound is found.
        j = overlap[j];
      }
    }
    // TODO(corysmith): reverse the search algorithm.
    // Keep the index of the found so that the rightmost index is taken.
    int found = -1;
    for (int i = 0, j = 0; i < haystack.segmentCount(); i++) {

      while (j >= 0 && !haystack.getSegment(i).equals(needle.getSegment(j))) {
        // Not matching, walk the needle index to attempt another match.
        j = overlap[j];
      }
      j++;
      // Needle index is exhausted, so the needle must match.
      if (j == needle.segmentCount()) {
        // Record the found index + 1 to be inclusive of the end index.
        found = i + 1;
        // Subtract one from the needle index to restart the search process
        j = j - 1;
      }
    }
    if (found != -1) {
      // Return the subsection of the haystack.
      return haystack.subFragment(0, found);
    }
    throw new IllegalArgumentException(String.format("%s was not found in %s", needle, haystack));
  }

  public static NestedSetBuilder<Artifact> collectTransitiveNativeLibsZips(
      RuleContext ruleContext) {
    NestedSetBuilder<Artifact> transitiveAarNativeLibs = NestedSetBuilder.naiveLinkOrder();
    Iterable<NativeLibsZipsProvider> providers = getTransitivePrerequisites(
        ruleContext, Mode.TARGET, NativeLibsZipsProvider.class);
    for (NativeLibsZipsProvider nativeLibsZipsProvider : providers) {
      transitiveAarNativeLibs.addTransitive(nativeLibsZipsProvider.getAarNativeLibs());
    }
    return transitiveAarNativeLibs;
  }

  static boolean getExportsManifest(RuleContext ruleContext) {
    // AndroidLibraryBaseRule has exports_manifest but AndroidBinaryBaseRule does not.
    // ResourceContainers are built for both, so we must check if exports_manifest is present.
    if (!ruleContext.attributes().has("exports_manifest", BuildType.TRISTATE)) {
      return false;
    }
    TriState attributeValue = ruleContext.attributes().get("exports_manifest", BuildType.TRISTATE);

    // If the rule does not have the Android configuration fragment, we default to false.
    boolean exportsManifestDefault =
        ruleContext.isLegalFragment(AndroidConfiguration.class)
            && ruleContext.getFragment(AndroidConfiguration.class).getExportsManifestDefault();
    return attributeValue == TriState.YES
        || (attributeValue == TriState.AUTO && exportsManifestDefault);
  }

  /** Returns the artifact for the debug key for signing the APK. */
  static Artifact getApkDebugSigningKey(RuleContext ruleContext) {
    return ruleContext.getHostPrerequisiteArtifact("debug_key");
  }

  private void compileResources(
      JavaSemantics javaSemantics,
      ResourceApk resourceApk,
      Artifact resourcesJar,
      JavaCompilationArtifacts.Builder artifactsBuilder,
      JavaTargetAttributes.Builder attributes,
      NestedSetBuilder<Artifact> filesBuilder,
      boolean useRClassGenerator)
      throws InterruptedException, RuleErrorException {
    compileResourceJar(javaSemantics, resourceApk, resourcesJar, useRClassGenerator);
    // Add the compiled resource jar to the classpath of the main compilation.
    attributes.addDirectJars(NestedSetBuilder.create(Order.STABLE_ORDER, resourceClassJar));
    // Add the compiled resource jar to the classpath of consuming targets.
    // We don't actually use the ijar. That is almost the same as the resource class jar
    // except for <clinit>, but it takes time to build and waiting for that to build would
    // just delay building the rest of the library.
    artifactsBuilder.addCompileTimeJarAsFullJar(resourceClassJar);

    // Add the compiled resource jar as a declared output of the rule.
    filesBuilder.add(resourceSourceJar);
    filesBuilder.add(resourceClassJar);
  }

  private void compileResourceJar(
      JavaSemantics javaSemantics, ResourceApk resourceApk, Artifact resourcesJar,
      boolean useRClassGenerator)
      throws InterruptedException, RuleErrorException {
    resourceSourceJar = ruleContext.getImplicitOutputArtifact(
        AndroidRuleClasses.ANDROID_RESOURCES_SOURCE_JAR);
    resourceClassJar = ruleContext.getImplicitOutputArtifact(
        AndroidRuleClasses.ANDROID_RESOURCES_CLASS_JAR);

    JavaCompilationArtifacts.Builder javaArtifactsBuilder = new JavaCompilationArtifacts.Builder();
    JavaTargetAttributes.Builder javacAttributes = new JavaTargetAttributes.Builder(javaSemantics)
        .addSourceJar(resourcesJar);
    JavaCompilationHelper javacHelper = new JavaCompilationHelper(
        ruleContext, javaSemantics, getJavacOpts(), javacAttributes);
    // Only build the class jar if it's not already generated internally by resource processing.
    if (resourceApk.getResourceJavaClassJar() == null) {
      if (useRClassGenerator) {
        new RClassGeneratorActionBuilder(ruleContext)
            .targetAaptVersion(AndroidAaptVersion.chooseTargetAaptVersion(ruleContext))
            .withPrimary(resourceApk.getPrimaryResource())
            .withDependencies(resourceApk.getResourceDependencies())
            .setClassJarOut(resourceClassJar)
            .build();
      } else {
        Artifact outputDepsProto =
            javacHelper.createOutputDepsProtoArtifact(resourceClassJar, javaArtifactsBuilder);
        javacHelper.createCompileActionWithInstrumentation(
            resourceClassJar,
            null /* manifestProtoOutput */,
            null /* genSourceJar */,
            outputDepsProto,
            javaArtifactsBuilder);
      }
    } else {
      // Otherwise, it should have been the AndroidRuleClasses.ANDROID_RESOURCES_CLASS_JAR.
      Preconditions.checkArgument(
          resourceApk.getResourceJavaClassJar().equals(
              ruleContext.getImplicitOutputArtifact(
                  AndroidRuleClasses.ANDROID_RESOURCES_CLASS_JAR)));
    }
    javacHelper.createSourceJarAction(resourceSourceJar, null);
  }

  private void createJarJarActions(
      JavaTargetAttributes.Builder attributes,
      NestedSetBuilder<Artifact> jarsProducedForRuntime,
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
      ruleContext.registerAction(
          FileWriteAction.create(ruleContext, jarJarRuleFile, jarJarRule, false));

      FilesToRunProvider jarjar =
          ruleContext.getExecutablePrerequisite("$jarjar_bin", Mode.HOST);

      ruleContext.registerAction(
          new SpawnAction.Builder()
              .useDefaultShellEnvironment()
              .setExecutable(jarjar)
              .setProgressMessage("Repackaging jar")
              .setMnemonic("AndroidRepackageJar")
              .addInput(jarJarRuleFile)
              .addInput(binaryResourcesJar)
              .addOutput(resourcesJar)
              .addCommandLine(
                  CustomCommandLine.builder()
                      .add("process")
                      .addExecPath(jarJarRuleFile)
                      .addExecPath(binaryResourcesJar)
                      .addExecPath(resourcesJar)
                      .build())
              .build(ruleContext));
      jarsProducedForRuntime.add(resourcesJar);
    }
  }

  private static Artifact createResourceJarArtifact(RuleContext ruleContext,
      ResourceContainer container, String fileNameSuffix) {

    String artifactName = container.getLabel().getName() + fileNameSuffix;

    // Since the Java sources are generated by combining all resources with the
    // ones included in the binary, the path of the artifact has to be unique
    // per binary and per library (not only per library).
    Artifact artifact = ruleContext.getUniqueDirectoryArtifact("resource_jars",
        container.getLabel().getPackageIdentifier().getSourceRoot().getRelative(artifactName),
        ruleContext.getBinOrGenfilesDirectory());
    return artifact;
  }

  public JavaTargetAttributes init(
      JavaSemantics javaSemantics,
      AndroidSemantics androidSemantics,
      ResourceApk resourceApk,
      boolean addCoverageSupport,
      boolean collectJavaCompilationArgs,
      boolean isBinary,
      NestedSet<Artifact> excludedRuntimeArtifacts)
      throws InterruptedException, RuleErrorException {

    classJar = ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_LIBRARY_CLASS_JAR);
    idlHelper = new AndroidIdlHelper(ruleContext, classJar);


    ImmutableList<Artifact> bootclasspath;
    if (getAndroidConfig(ruleContext).desugarJava8()) {
      bootclasspath = ImmutableList.<Artifact>builder()
          .addAll(ruleContext.getPrerequisite("$desugar_java8_extra_bootclasspath", Mode.HOST)
              .getProvider(FileProvider.class)
              .getFilesToBuild())
          .add(AndroidSdkProvider.fromRuleContext(ruleContext).getAndroidJar())
          .build();
    } else {
      bootclasspath =
          ImmutableList.of(AndroidSdkProvider.fromRuleContext(ruleContext).getAndroidJar());
    }
    Iterable<String> javacopts = androidSemantics.getJavacArguments(ruleContext);
    if (DataBinding.isEnabled(ruleContext)) {
      javacopts = Iterables.concat(javacopts, DataBinding.getJavacopts(ruleContext, isBinary));
    }
    JavaTargetAttributes.Builder attributes =
        javaCommon
            .initCommon(idlHelper.getIdlGeneratedJavaSources(), javacopts)
            .setBootClassPath(bootclasspath);
    if (DataBinding.isEnabled(ruleContext)) {
      DataBinding.addAnnotationProcessor(ruleContext, attributes);
    }

    if (excludedRuntimeArtifacts != null) {
      attributes.addExcludedArtifacts(excludedRuntimeArtifacts);
    }

    JavaCompilationArtifacts.Builder artifactsBuilder = new JavaCompilationArtifacts.Builder();
    NestedSetBuilder<Artifact> jarsProducedForRuntime = NestedSetBuilder.<Artifact>stableOrder();
    NestedSetBuilder<Artifact> filesBuilder = NestedSetBuilder.<Artifact>stableOrder();

    Artifact resourcesJar = resourceApk.getResourceJavaSrcJar();
    if (resourcesJar != null) {
      filesBuilder.add(resourcesJar);
      // Use a fast-path R class generator for android_binary with local resources, where there is
      // a bottleneck. For legacy resources, the srcjar and R class compiler don't match up
      // (the legacy srcjar requires the createJarJar step below).
      boolean useRClassGenerator = isBinary && !resourceApk.isLegacy();
      compileResources(javaSemantics, resourceApk, resourcesJar, artifactsBuilder, attributes,
          filesBuilder, useRClassGenerator);

      // Combined resource constants needs to come even before our own classes that may contain
      // local resource constants.
      artifactsBuilder.addRuntimeJar(resourceClassJar);
      jarsProducedForRuntime.add(resourceClassJar);

      if (resourceApk.isLegacy()) {
        // Repackages the R.java for each dependency package and places the resultant jars before
        // the dependency libraries to ensure that the generated resource ids are correct.
        createJarJarActions(attributes, jarsProducedForRuntime,
            resourceApk.getResourceDependencies().getResourceContainers(),
            resourceApk.getPrimaryResource().getJavaPackage(), resourceClassJar);
      }
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

    initJava(
        javaSemantics,
        helper,
        artifactsBuilder,
        collectJavaCompilationArgs,
        filesBuilder,
        isBinary);
    if (ruleContext.hasErrors()) {
      return null;
    }
    if (generatedExtensionRegistryProvider != null) {
      jarsProducedForRuntime.add(generatedExtensionRegistryProvider.getClassJar());
    }
    this.jarsProducedForRuntime = jarsProducedForRuntime.add(classJar).build();
    return helper.getAttributes();
  }

  private JavaCompilationHelper initAttributes(
      JavaTargetAttributes.Builder attributes, JavaSemantics semantics) {
    boolean useDataBinding = DataBinding.isEnabled(ruleContext);
    JavaCompilationHelper helper = new JavaCompilationHelper(ruleContext, semantics,
        javaCommon.getJavacOpts(), attributes,
        useDataBinding ? DataBinding.processDeps(ruleContext) : ImmutableList.<Artifact>of(),
        // We have to disable strict deps checking with data binding because data binding propagates
        // layout XML up the dependency chain. Say a library's XML references a Java class,
        // e.g.: "<variable type="some.package.SomeClass" />". Data binding's annotation processor
        // triggers a compile against SomeClass. Because data binding reprocesses bindings
        // each step up the dependency chain (via merged resources), that means this compile also
        // happens at the top-level binary. Since SomeClass.java is declared in the library, this
        // creates a strict deps violation.
        //
        // This weakening of strict deps is unfortunate and deserves to be fixed. Once data
        // binding integrates with aapt2 this problem should naturally go away (since reprocessing
        // will no longer happen).
        /*disableStrictDeps=*/useDataBinding);

    helper.addLibrariesToAttributes(javaCommon.targetsTreatedAsDeps(ClasspathType.COMPILE_ONLY));
    attributes.setRuleKind(ruleContext.getRule().getRuleClass());
    attributes.setTargetLabel(ruleContext.getLabel());

    JavaCommon.validateConstraint(ruleContext, "android",
        javaCommon.targetsTreatedAsDeps(ClasspathType.BOTH));
    ruleContext.checkSrcsSamePackage(true);
    return helper;
  }

  private void initJava(
      JavaSemantics javaSemantics,
      JavaCompilationHelper helper,
      JavaCompilationArtifacts.Builder javaArtifactsBuilder,
      boolean collectJavaCompilationArgs,
      NestedSetBuilder<Artifact> filesBuilder,
      boolean isBinary)
      throws InterruptedException {
    JavaTargetAttributes attributes = helper.getAttributes();
    if (ruleContext.hasErrors()) {
      // Avoid leaving filesToBuild set to null, otherwise we'll get a NullPointerException masking
      // the real error.
      filesToBuild = filesBuilder.build();
      return;
    }

    Artifact jar = null;
    if (attributes.hasSources() || attributes.hasResources()) {
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
    javaSourceJarsProviderBuilder
        .addSourceJar(srcJar)
        .addAllTransitiveSourceJars(javaCommon.collectTransitiveSourceJars(srcJar));
    helper.createSourceJarAction(srcJar, genSourceJar);

    outputDepsProto = helper.createOutputDepsProtoArtifact(classJar, javaArtifactsBuilder);
    helper.createCompileActionWithInstrumentation(classJar, manifestProtoOutput, genSourceJar,
        outputDepsProto, javaArtifactsBuilder);

    if (isBinary) {
      generatedExtensionRegistryProvider =
          javaSemantics.createGeneratedExtensionRegistry(
              ruleContext,
              javaCommon,
              filesBuilder,
              javaArtifactsBuilder,
              javaRuleOutputJarsProviderBuilder,
              javaSourceJarsProviderBuilder);
    }

    filesToBuild = filesBuilder.build();

    if ((attributes.hasSources()) && jar != null) {
      iJar = helper.createCompileTimeJarAction(jar, javaArtifactsBuilder);
    }

    JavaCompilationArtifacts javaArtifacts = javaArtifactsBuilder.build();
    compileTimeDependencyArtifacts =
        javaCommon.collectCompileTimeDependencyArtifacts(
            javaArtifacts.getCompileTimeDependencyArtifact());
    javaCommon.setJavaCompilationArtifacts(javaArtifacts);

    javaCommon.setClassPathFragment(
        new ClasspathConfiguredFragment(
            javaCommon.getJavaCompilationArtifacts(),
            attributes,
            asNeverLink,
            helper.getBootclasspathOrDefault()));

    transitiveNeverlinkLibraries = collectTransitiveNeverlinkLibraries(
        ruleContext,
        javaCommon.getDependencies(),
        javaCommon.getJavaCompilationArtifacts().getRuntimeJars());
    if (collectJavaCompilationArgs) {
      boolean hasSources = attributes.hasSources();
      this.javaCompilationArgs =
          collectJavaCompilationArgs(exportDeps, asNeverLink, hasSources);
      this.recursiveJavaCompilationArgs = collectJavaCompilationArgs(
          true, asNeverLink, /* hasSources */ true);
    }
  }

  public RuleConfiguredTargetBuilder addTransitiveInfoProviders(
      RuleConfiguredTargetBuilder builder,
      AndroidSemantics androidSemantics,
      Artifact aar,
      ResourceApk resourceApk,
      Artifact zipAlignedApk,
      Iterable<Artifact> apksUnderTest,
      NativeLibs nativeLibs,
      boolean isResourcesOnly) {

    idlHelper.addTransitiveInfoProviders(builder, classJar, manifestProtoOutput);

    if (generatedExtensionRegistryProvider != null) {
      builder.add(GeneratedExtensionRegistryProvider.class, generatedExtensionRegistryProvider);
    }
    OutputJar resourceJar = null;
    if (resourceClassJar != null && resourceSourceJar != null) {
      resourceJar = new OutputJar(resourceClassJar, null, ImmutableList.of(resourceSourceJar));
      javaRuleOutputJarsProviderBuilder.addOutputJar(resourceJar);
    }

    JavaRuleOutputJarsProvider ruleOutputJarsProvider =
        javaRuleOutputJarsProviderBuilder
            .addOutputJar(classJar, iJar, ImmutableList.of(srcJar))
            .setJdeps(outputDepsProto)
            .build();
    JavaSourceJarsProvider sourceJarsProvider = javaSourceJarsProviderBuilder.build();
    JavaCompilationArgsProvider compilationArgsProvider =
        JavaCompilationArgsProvider.create(
            javaCompilationArgs,
            recursiveJavaCompilationArgs,
            compileTimeDependencyArtifacts,
            NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER));
    javaCommon.addTransitiveInfoProviders(builder, filesToBuild, classJar, ANDROID_COLLECTION_SPEC);

    JavaGenJarsProvider javaGenJarsProvider =
        javaCommon.createJavaGenJarsProvider(genClassJar, genSourceJar);
    javaCommon.addJavaGenJarsProvider(builder, javaGenJarsProvider);

    DataBinding.maybeAddProvider(builder, ruleContext);
    JavaInfo javaInfo = JavaInfo.Builder.create()
        .addProvider(JavaCompilationArgsProvider.class, compilationArgsProvider)
        .addProvider(JavaRuleOutputJarsProvider.class, ruleOutputJarsProvider)
        .addProvider(JavaSourceJarsProvider.class, sourceJarsProvider)
        .addProvider(JavaGenJarsProvider.class, javaGenJarsProvider)
        .build();

    return builder
        .setFilesToBuild(filesToBuild)
        .addSkylarkTransitiveInfo(
            JavaSkylarkApiProvider.NAME, JavaSkylarkApiProvider.fromRuleContext())
        .addNativeDeclaredProvider(javaInfo)
        .addProvider(
            JavaRuntimeJarProvider.class,
            new JavaRuntimeJarProvider(javaCommon.getJavaCompilationArtifacts().getRuntimeJars()))
        .addProvider(RunfilesProvider.class, RunfilesProvider.simple(getRunfiles()))
        .addProvider(
            AndroidResourcesProvider.class,
            resourceApk.toResourceProvider(ruleContext.getLabel(), isResourcesOnly))
        .addProvider(
            AndroidIdeInfoProvider.class,
            createAndroidIdeInfoProvider(
                ruleContext,
                androidSemantics,
                idlHelper,
                resourceJar,
                aar,
                resourceApk,
                zipAlignedApk,
                apksUnderTest,
                nativeLibs))
        .addSkylarkTransitiveInfo(AndroidSkylarkApiProvider.NAME, new AndroidSkylarkApiProvider())
        .addOutputGroup(
            OutputGroupProvider.HIDDEN_TOP_LEVEL, collectHiddenTopLevelArtifacts(ruleContext))
        .addOutputGroup(
            JavaSemantics.SOURCE_JARS_OUTPUT_GROUP, sourceJarsProvider.getTransitiveSourceJars());
  }

  private Runfiles getRunfiles() {
    // TODO(bazel-team): why return any Runfiles in the neverlink case?
    if (asNeverLink) {
      return new Runfiles.Builder(
          ruleContext.getWorkspaceName(), ruleContext.getConfiguration().legacyExternalRunfiles())
          .addRunfiles(ruleContext, RunfilesProvider.DEFAULT_RUNFILES)
          .build();
    }
    return JavaCommon.getRunfiles(
        ruleContext, javaCommon.getJavaSemantics(), javaCommon.getJavaCompilationArtifacts(),
        asNeverLink);
  }

  public static PathFragment getAssetDir(RuleContext ruleContext) {
    return PathFragment.create(ruleContext.attributes().get(
        ResourceType.ASSETS.getAttribute() + "_dir",
        Type.STRING));
  }

  public static AndroidResourcesProvider getAndroidResources(RuleContext ruleContext) {
    if (!ruleContext.attributes().has("resources", BuildType.LABEL)) {
      return null;
    }
    TransitiveInfoCollection prerequisite = ruleContext.getPrerequisite("resources", Mode.TARGET);
    if (prerequisite == null) {
      return null;
    }

    AndroidResourcesProvider provider = prerequisite.getProvider(AndroidResourcesProvider.class);

    if (!provider.getIsResourcesOnly()) {
      ruleContext.attributeError(
          "resources",
          "android_library target "
              + prerequisite.getLabel()
              + " cannot be used in the 'resources' attribute as it specifies information (probably"
              + " 'srcs' or 'deps') not directly related to android_resources. Consider moving this"
              + " target from 'resources' to 'deps'.");
      return null;
    }

    ruleContext.ruleWarning(
        "The use of the android_resources rule and the resources attribute is deprecated. "
            + "Please use the resource_files, assets, and manifest attributes of android_library.");
    return provider;
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
    boolean exportDeps = !hasSrcs
        && ruleContext.getFragment(AndroidConfiguration.class).allowSrcsLessAndroidLibraryDeps();
    return javaCommon.collectJavaCompilationArgs(recursive, isNeverLink, exportDeps);
  }

  public ImmutableList<String> getJavacOpts() {
    return javaCommon.getJavacOpts();
  }

  public ImmutableList<Artifact> getRuntimeJars() {
    return javaCommon.getJavaCompilationArtifacts().getRuntimeJars();
  }

  public Artifact getResourceClassJar() {
    return resourceClassJar;
  }

  /**
   * Returns Jars produced by this rule that may go into the runtime classpath.  By contrast
   * {@link #getRuntimeJars()} returns the complete runtime classpath needed by this rule, including
   * dependencies.
   */
  public NestedSet<Artifact> getJarsProducedForRuntime() {
    return jarsProducedForRuntime;
  }

  public Artifact getInstrumentedJar() {
    return javaCommon.getJavaCompilationArtifacts().getInstrumentedJar();
  }

  public NestedSet<Artifact> getTransitiveNeverLinkLibraries() {
    return transitiveNeverlinkLibraries;
  }

  public boolean isNeverLink() {
    return asNeverLink;
  }

  public CcLinkParamsStore getCcLinkParamsStore() {
    return getCcLinkParamsStore(
        javaCommon.targetsTreatedAsDeps(ClasspathType.BOTH), ImmutableList.<String>of());
  }

  public static CcLinkParamsStore getCcLinkParamsStore(
      final Iterable<? extends TransitiveInfoCollection> deps, final Collection<String> linkOpts) {
    return new CcLinkParamsStore() {
      @Override
      protected void collect(
          CcLinkParams.Builder builder, boolean linkingStatically, boolean linkShared) {
        builder.addTransitiveTargets(
            deps,
            // Link in Java-specific C++ code in the transitive closure
            JavaCcLinkParamsProvider.TO_LINK_PARAMS,
            // Link in Android-specific C++ code (e.g., android_libraries) in the transitive closure
            AndroidCcLinkParamsProvider.TO_LINK_PARAMS,
            // Link in non-language-specific C++ code in the transitive closure
            CcLinkParamsInfo.TO_LINK_PARAMS);
        builder.addLinkOpts(linkOpts);
      }
    };
  }

  /**
   * Returns {@link AndroidConfiguration} in given context.
   */
  static AndroidConfiguration getAndroidConfig(RuleContext context) {
    return context.getConfiguration().getFragment(AndroidConfiguration.class);
  }

  private NestedSet<Artifact> collectHiddenTopLevelArtifacts(RuleContext ruleContext) {
    NestedSetBuilder<Artifact> builder = NestedSetBuilder.stableOrder();
    for (OutputGroupProvider provider :
        getTransitivePrerequisites(
            ruleContext, Mode.TARGET, OutputGroupProvider.SKYLARK_CONSTRUCTOR)) {
      builder.addTransitive(provider.getOutputGroup(OutputGroupProvider.HIDDEN_TOP_LEVEL));
    }
    return builder.build();
  }

  /**
   * Returns a {@link JavaCommon} instance with Android data binding support.
   *
   * <p>Binaries need both compile-time and runtime support, while libraries only need compile-time
   * support.
   *
   * <p>No rule needs <i>any</i> support if data binding is disabled.
   */
  static JavaCommon createJavaCommonWithAndroidDataBinding(RuleContext ruleContext,
      JavaSemantics semantics, boolean isLibrary) {
    boolean useDataBinding = DataBinding.isEnabled(ruleContext);

    ImmutableList<Artifact> srcs =
        ruleContext.getPrerequisiteArtifacts("srcs", RuleConfiguredTarget.Mode.TARGET).list();
    if (useDataBinding) {
      // Add this rule's annotation processor input. If the rule doesn't have direct resources,
      // there's no direct data binding info, so there's strictly no need for annotation processing.
      // But it's still important to process the deps' .bin files so any Java class references get
      // re-referenced so they don't get filtered out of the compilation classpath by JavaBuilder
      // (which filters out classpath .jars that "aren't used": see --reduce_classpath). If data
      // binding didn't reprocess a library's data binding expressions redundantly up the dependency
      // chain (meaning each depender processes them again as if they were its own), this problem
      // wouldn't happen.
      Artifact annotationFile = DataBinding.createAnnotationFile(ruleContext);
      if (annotationFile != null) {
        srcs = ImmutableList.<Artifact>builder().addAll(srcs).add(annotationFile).build();
      }
    }

    ImmutableList<TransitiveInfoCollection> compileDeps;
    ImmutableList<TransitiveInfoCollection> runtimeDeps;
    ImmutableList<TransitiveInfoCollection> bothDeps;

    if (isLibrary) {
      compileDeps = JavaCommon.defaultDeps(ruleContext, semantics, ClasspathType.COMPILE_ONLY);
      compileDeps = AndroidIdlHelper.maybeAddSupportLibs(ruleContext, compileDeps);
      runtimeDeps = JavaCommon.defaultDeps(ruleContext, semantics, ClasspathType.RUNTIME_ONLY);
      bothDeps = JavaCommon.defaultDeps(ruleContext, semantics, ClasspathType.BOTH);
    } else {
      // Binary:
      compileDeps = ImmutableList.copyOf(
          ruleContext.getPrerequisites("deps", RuleConfiguredTarget.Mode.TARGET));
      runtimeDeps = compileDeps;
      bothDeps = compileDeps;
    }

    return new JavaCommon(ruleContext, semantics, srcs, compileDeps, runtimeDeps, bothDeps);
  }

  /**
   * Gets the transitive support APKs required by this rule through the {@code support_apks}
   * attribute.
   */
  static NestedSet<Artifact> getSupportApks(RuleContext ruleContext) {
    NestedSetBuilder<Artifact> supportApks = NestedSetBuilder.stableOrder();
    for (TransitiveInfoCollection dep : ruleContext.getPrerequisites("support_apks", Mode.TARGET)) {
      ApkProvider apkProvider = dep.getProvider(ApkProvider.class);
      FileProvider fileProvider = dep.getProvider(FileProvider.class);
      // If ApkProvider is present, do not check FileProvider for .apk files. For example,
      // android_binary creates a FileProvider containing both the signed and unsigned APKs.
      if (apkProvider != null) {
        supportApks.add(apkProvider.getApk());
      } else if (fileProvider != null) {
        // The rule definition should enforce that only .apk files are allowed, however, it can't
        // hurt to double check.
        supportApks.addAll(FileType.filter(fileProvider.getFilesToBuild(), AndroidRuleClasses.APK));
      }
    }
    return supportApks.build();
  }

  public static void validateResourcesAttribute(RuleContext ruleContext) throws RuleErrorException {
    if (ruleContext.attributes().isAttributeValueExplicitlySpecified("resources")
        && !ruleContext.getFragment(AndroidConfiguration.class).allowResourcesAttr()
        && !Whitelist.isAvailable(ruleContext, RESOURCES_WHITELIST_NAME)) {
      ruleContext.throwWithAttributeError(
          "resources",
          "The resources attribute has been removed. Please use resource_files instead.");
    }
  }
}
