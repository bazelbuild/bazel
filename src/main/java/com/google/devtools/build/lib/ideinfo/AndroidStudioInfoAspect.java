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

package com.google.devtools.build.lib.ideinfo;

import static com.google.common.collect.Iterables.transform;
import static com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition.HOST;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;

import com.google.common.base.Function;
import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.io.ByteSource;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.analysis.AnalysisUtils;
import com.google.devtools.build.lib.analysis.ConfiguredAspect;
import com.google.devtools.build.lib.analysis.ConfiguredAspect.Builder;
import com.google.devtools.build.lib.analysis.ConfiguredAspectFactory;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.actions.BinaryFileWriteAction;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.ideinfo.androidstudio.AndroidStudioIdeInfo.AndroidRuleIdeInfo;
import com.google.devtools.build.lib.ideinfo.androidstudio.AndroidStudioIdeInfo.ArtifactLocation;
import com.google.devtools.build.lib.ideinfo.androidstudio.AndroidStudioIdeInfo.CRuleIdeInfo;
import com.google.devtools.build.lib.ideinfo.androidstudio.AndroidStudioIdeInfo.CToolchainIdeInfo;
import com.google.devtools.build.lib.ideinfo.androidstudio.AndroidStudioIdeInfo.JavaRuleIdeInfo;
import com.google.devtools.build.lib.ideinfo.androidstudio.AndroidStudioIdeInfo.LibraryArtifact;
import com.google.devtools.build.lib.ideinfo.androidstudio.AndroidStudioIdeInfo.RuleIdeInfo;
import com.google.devtools.build.lib.ideinfo.androidstudio.AndroidStudioIdeInfo.RuleIdeInfo.Kind;
import com.google.devtools.build.lib.ideinfo.androidstudio.AndroidStudioIdeInfo.TestInfo;
import com.google.devtools.build.lib.packages.AspectDefinition;
import com.google.devtools.build.lib.packages.AspectParameters;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.NativeAspectClass;
import com.google.devtools.build.lib.packages.NonconfigurableAttributeMapper;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.rules.android.AndroidIdeInfoProvider;
import com.google.devtools.build.lib.rules.android.AndroidIdeInfoProvider.SourceDirectory;
import com.google.devtools.build.lib.rules.android.AndroidSdkProvider;
import com.google.devtools.build.lib.rules.cpp.CcToolchainProvider;
import com.google.devtools.build.lib.rules.cpp.CppCompilationContext;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration;
import com.google.devtools.build.lib.rules.java.JavaExportsProvider;
import com.google.devtools.build.lib.rules.java.JavaGenJarsProvider;
import com.google.devtools.build.lib.rules.java.JavaRuleOutputJarsProvider;
import com.google.devtools.build.lib.rules.java.JavaRuleOutputJarsProvider.OutputJar;
import com.google.devtools.build.lib.rules.java.JavaSourceInfoProvider;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.protobuf.MessageLite;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.Collection;
import java.util.List;

import javax.annotation.Nullable;

/**
 * Generates ide-build information for Android Studio.
 */
public class AndroidStudioInfoAspect extends NativeAspectClass implements ConfiguredAspectFactory {
  public static final String NAME = "AndroidStudioInfoAspect";

  // Output groups.

  public static final String IDE_INFO = "ide-info";
  public static final String IDE_INFO_TEXT = "ide-info-text";
  public static final String IDE_RESOLVE = "ide-resolve";

  private final String toolsRepository;
  private final AndroidStudioInfoSemantics androidStudioInfoSemantics;
  private final ImmutableList<PrerequisiteAttr> prerequisiteAttrs;

  public AndroidStudioInfoAspect(
      String toolsRepository,
      AndroidStudioInfoSemantics androidStudioInfoSemantics) {
    this.toolsRepository = toolsRepository;
    this.androidStudioInfoSemantics = androidStudioInfoSemantics;
    this.prerequisiteAttrs = buildPrerequisiteAttrs();
  }

  @Override
  public String getName() {
    return NAME;
  }

  /**
   * Attribute to propagate dependencies along.
   */
  public static class PrerequisiteAttr {
    public final String name;
    public final Type<?> type;
    public PrerequisiteAttr(String name, Type<?> type) {
      this.name = name;
      this.type = type;
    }
  }

  private ImmutableList<PrerequisiteAttr> buildPrerequisiteAttrs() {
    ImmutableList.Builder<PrerequisiteAttr> builder = ImmutableList.builder();
    builder.add(new PrerequisiteAttr("deps", BuildType.LABEL_LIST));
    builder.add(new PrerequisiteAttr("exports", BuildType.LABEL_LIST));
    // From android_test
    builder.add(new PrerequisiteAttr("binary_under_test", BuildType.LABEL));
    // from cc_* rules
    builder.add(new PrerequisiteAttr(":cc_toolchain", BuildType.LABEL));

    androidStudioInfoSemantics.augmentPrerequisiteAttrs(builder);

    return builder.build();
  }

  // File suffixes.
  public static final String ASWB_BUILD_SUFFIX = ".aswb-build";
  public static final String ASWB_BUILD_TEXT_SUFFIX = ".aswb-build.txt";
  public static final Function<Label, String> LABEL_TO_STRING = new Function<Label, String>() {
    @Nullable
    @Override
    public String apply(Label label) {
      return label.toString();
    }
  };

  @Override
  public AspectDefinition getDefinition(AspectParameters aspectParameters) {
    AspectDefinition.Builder builder = new AspectDefinition.Builder(NAME)
        .attributeAspect("runtime_deps", this)
        .attributeAspect("resources", this)
        .add(attr("$packageParser", LABEL).cfg(HOST).exec()
            .value(Label.parseAbsoluteUnchecked(
                toolsRepository + "//tools/android:PackageParser")));

    for (PrerequisiteAttr prerequisiteAttr : prerequisiteAttrs) {
      builder.attributeAspect(prerequisiteAttr.name, this);
    }

    return builder.build();
  }

  @Override
  public ConfiguredAspect create(
      ConfiguredTarget base, RuleContext ruleContext, AspectParameters parameters) {
    ConfiguredAspect.Builder builder = new Builder(NAME, ruleContext);

    AndroidStudioInfoFilesProvider.Builder providerBuilder =
        new AndroidStudioInfoFilesProvider.Builder();

    RuleIdeInfo.Kind ruleKind = getRuleKind(ruleContext.getRule(), base);

    DependenciesResult dependenciesResult = processDependencies(
        base, ruleContext, providerBuilder);

    AndroidStudioInfoFilesProvider provider = createIdeBuildArtifact(
        base,
        ruleContext,
        ruleKind,
        dependenciesResult,
        providerBuilder);

    builder
        .addOutputGroup(IDE_INFO, provider.getIdeInfoFiles())
        .addOutputGroup(IDE_INFO_TEXT, provider.getIdeInfoTextFiles())
        .addOutputGroup(IDE_RESOLVE, provider.getIdeResolveFiles())
        .addProvider(
            AndroidStudioInfoFilesProvider.class,
            provider);

    return builder.build();
  }

  private static class DependenciesResult {
    private DependenciesResult(Iterable<Label> deps,
        Iterable<Label> runtimeDeps, @Nullable Label resources) {
      this.deps = deps;
      this.runtimeDeps = runtimeDeps;
      this.resources = resources;
    }
    final Iterable<Label> deps;
    final Iterable<Label> runtimeDeps;
    @Nullable final Label resources;
  }

  private DependenciesResult processDependencies(
      ConfiguredTarget base, RuleContext ruleContext,
      AndroidStudioInfoFilesProvider.Builder providerBuilder) {

    // Calculate direct dependencies
    ImmutableList.Builder<TransitiveInfoCollection> directDepsBuilder = ImmutableList.builder();
    for (PrerequisiteAttr prerequisiteAttr : prerequisiteAttrs) {
      if (ruleContext.attributes().has(prerequisiteAttr.name, prerequisiteAttr.type)) {
        directDepsBuilder.addAll(ruleContext.getPrerequisites(prerequisiteAttr.name, Mode.TARGET));
      }
    }
    List<TransitiveInfoCollection> directDeps = directDepsBuilder.build();

    // Add exports from direct dependencies
    NestedSetBuilder<Label> dependenciesBuilder = NestedSetBuilder.stableOrder();
    for (AndroidStudioInfoFilesProvider depProvider :
        AnalysisUtils.getProviders(directDeps, AndroidStudioInfoFilesProvider.class)) {
      dependenciesBuilder.addTransitive(depProvider.getExportedDeps());
    }
    for (TransitiveInfoCollection dep : directDeps) {
      dependenciesBuilder.add(dep.getLabel());
    }
    NestedSet<Label> dependencies = dependenciesBuilder.build();

    // Propagate my own exports
    JavaExportsProvider javaExportsProvider = base.getProvider(JavaExportsProvider.class);
    if (javaExportsProvider != null) {
      providerBuilder.exportedDepsBuilder()
          .addTransitive(javaExportsProvider.getTransitiveExports());
    }
    // android_library without sources exports all its deps
    if (ruleContext.getRule().getRuleClass().equals("android_library")) {
      JavaSourceInfoProvider sourceInfoProvider = base.getProvider(JavaSourceInfoProvider.class);
      boolean hasSources = sourceInfoProvider != null
          && !sourceInfoProvider.getSourceFiles().isEmpty();
      if (!hasSources) {
        for (TransitiveInfoCollection dep : directDeps) {
          providerBuilder.exportedDepsBuilder().add(dep.getLabel());
        }
      }
    }

    // runtime_deps
    List<? extends TransitiveInfoCollection> runtimeDeps = ImmutableList.of();
    NestedSetBuilder<Label> runtimeDepsBuilder = NestedSetBuilder.stableOrder();
    if (ruleContext.attributes().has("runtime_deps", BuildType.LABEL_LIST)) {
      runtimeDeps = ruleContext.getPrerequisites("runtime_deps", Mode.TARGET);
      for (TransitiveInfoCollection dep : runtimeDeps) {
        runtimeDepsBuilder.add(dep.getLabel());
      }
    }

    // resources
    @Nullable TransitiveInfoCollection resources =
        ruleContext.attributes().has("resources", BuildType.LABEL)
            ? ruleContext.getPrerequisite("resources", Mode.TARGET)
            : null;

    // Propagate providers from all prerequisites (deps + runtime_deps)
    ImmutableList.Builder<TransitiveInfoCollection> prerequisitesBuilder = ImmutableList.builder();
    prerequisitesBuilder.addAll(directDeps);
    prerequisitesBuilder.addAll(runtimeDeps);
    if (resources != null) {
      prerequisitesBuilder.add(resources);
    }

    List<TransitiveInfoCollection> prerequisites = prerequisitesBuilder.build();

    for (AndroidStudioInfoFilesProvider depProvider :
        AnalysisUtils.getProviders(prerequisites, AndroidStudioInfoFilesProvider.class)) {
      providerBuilder.ideInfoFilesBuilder().addTransitive(depProvider.getIdeInfoFiles());
      providerBuilder.ideInfoTextFilesBuilder().addTransitive(depProvider.getIdeInfoTextFiles());
      providerBuilder.ideResolveFilesBuilder().addTransitive(depProvider.getIdeResolveFiles());
    }


    return new DependenciesResult(
        dependencies,
        runtimeDepsBuilder.build(),
        resources != null ? resources.getLabel() : null);
  }

  private AndroidStudioInfoFilesProvider createIdeBuildArtifact(
      ConfiguredTarget base,
      RuleContext ruleContext,
      Kind ruleKind,
      DependenciesResult dependenciesResult,
      AndroidStudioInfoFilesProvider.Builder providerBuilder) {

    Artifact ideInfoFile = derivedArtifact(base, ruleContext, ASWB_BUILD_SUFFIX);
    Artifact ideInfoTextFile = derivedArtifact(base, ruleContext, ASWB_BUILD_TEXT_SUFFIX);
    providerBuilder.ideInfoFilesBuilder().add(ideInfoFile);
    providerBuilder.ideInfoTextFilesBuilder().add(ideInfoTextFile);
    NestedSetBuilder<Artifact> ideResolveArtifacts = providerBuilder.ideResolveFilesBuilder();

    RuleIdeInfo.Builder outputBuilder = RuleIdeInfo.newBuilder();

    outputBuilder.setLabel(base.getLabel().toString());

    outputBuilder.setBuildFile(
        ruleContext
            .getRule()
            .getPackage()
            .getBuildFile()
            .getPath()
            .toString());

    outputBuilder.setBuildFileArtifactLocation(
        makeArtifactLocation(ruleContext.getRule().getPackage()));

    if (ruleKind != Kind.UNRECOGNIZED) {
      outputBuilder.setKind(ruleKind);
    }
    outputBuilder.setKindString(ruleContext.getRule().getRuleClass());

    // Java rules
    JavaRuleOutputJarsProvider outputJarsProvider =
        base.getProvider(JavaRuleOutputJarsProvider.class);
    if (outputJarsProvider != null) {
      Artifact packageManifest = createPackageManifest(base, ruleContext);
      if (packageManifest != null) {
        providerBuilder.ideInfoFilesBuilder().add(packageManifest);
        ruleContext.registerAction(
            makePackageManifestAction(ruleContext, packageManifest, getJavaSources(ruleContext))
        );
      }

      JavaRuleIdeInfo javaRuleIdeInfo = makeJavaRuleIdeInfo(
          base, ruleContext, outputJarsProvider, ideResolveArtifacts, packageManifest);
      outputBuilder.setJavaRuleIdeInfo(javaRuleIdeInfo);
    }

    // C rules
    if (isCppRule(base)) {
      CppCompilationContext cppCompilationContext = base.getProvider(CppCompilationContext.class);
      if (cppCompilationContext != null) {
        CRuleIdeInfo cRuleIdeInfo =
            makeCRuleIdeInfo(base, ruleContext, cppCompilationContext, ideResolveArtifacts);
        outputBuilder.setCRuleIdeInfo(cRuleIdeInfo);
      }
    }

    // CCToolchain rule
    CcToolchainProvider ccToolchainProvider = base.getProvider(CcToolchainProvider.class);
    if (ccToolchainProvider != null) {
      CppConfiguration cppConfiguration = ccToolchainProvider.getCppConfiguration();
      if (cppConfiguration != null) {
        CToolchainIdeInfo cToolchainIdeInfo = makeCToolchainIdeInfo(ruleContext, cppConfiguration);
        if (cToolchainIdeInfo != null) {
          outputBuilder.setCToolchainIdeInfo(cToolchainIdeInfo);
        }
      }
    }

    // Android rules
    AndroidIdeInfoProvider androidIdeInfoProvider = base.getProvider(AndroidIdeInfoProvider.class);
    if (androidIdeInfoProvider != null) {
      outputBuilder.setAndroidRuleIdeInfo(makeAndroidRuleIdeInfo(
          androidIdeInfoProvider, dependenciesResult, ideResolveArtifacts));
    }

    // Test rules
    if (TargetUtils.isTestRule(base.getTarget())) {
      TestInfo.Builder builder = TestInfo.newBuilder();
      String attr = NonconfigurableAttributeMapper.of(base.getTarget().getAssociatedRule())
          .get("size", Type.STRING);
      if (attr != null) {
        builder.setSize(attr);
      }
      outputBuilder.setTestInfo(builder);
    }

    androidStudioInfoSemantics.augmentRuleInfo(
        outputBuilder, base, ruleContext, ideResolveArtifacts);

    AndroidStudioInfoFilesProvider provider = providerBuilder.build();

    outputBuilder.addAllDependencies(transform(dependenciesResult.deps, LABEL_TO_STRING));
    outputBuilder.addAllRuntimeDeps(transform(dependenciesResult.runtimeDeps, LABEL_TO_STRING));
    outputBuilder.addAllTags(base.getTarget().getAssociatedRule().getRuleTags());

    final RuleIdeInfo ruleIdeInfo = outputBuilder.build();

    ruleContext.registerAction(
        makeProtoWriteAction(ruleContext.getActionOwner(), ruleIdeInfo, ideInfoFile));
    ruleContext.registerAction(
        makeProtoTextWriteAction(ruleContext.getActionOwner(), ruleIdeInfo, ideInfoTextFile));

    return provider;
  }

  private boolean isCppRule(ConfiguredTarget base) {
    String ruleClass = base.getTarget().getAssociatedRule().getRuleClass();
    switch (ruleClass) {
      case "cc_library":
      case "cc_binary":
      case "cc_test":
      case "cc_inc_library:":
        return true;
      default:
        // Fall through
    }
    return androidStudioInfoSemantics.checkForAdditionalCppRules(ruleClass);
  }

  @Nullable private static Artifact createPackageManifest(ConfiguredTarget base,
      RuleContext ruleContext) {
    Collection<Artifact> sourceFiles = getJavaSources(ruleContext);
    if (sourceFiles.isEmpty()) {
      return null;
    }
    return derivedArtifact(base, ruleContext, ".manifest");
  }

  private static Action[] makePackageManifestAction(
      RuleContext ruleContext,
      Artifact packageManifest,
      Collection<Artifact> sourceFiles) {

    return new SpawnAction.Builder()
        .addInputs(sourceFiles)
        .addOutput(packageManifest)
        .setExecutable(ruleContext.getExecutablePrerequisite("$packageParser", Mode.HOST))
        .setCommandLine(CustomCommandLine.builder()
            .addExecPath("--output_manifest", packageManifest)
            .addJoinStrings("--sources", ":", toSerializedArtifactLocations(sourceFiles))
            .build())
        .useParameterFile(ParameterFileType.SHELL_QUOTED)
        .setProgressMessage("Parsing java package strings for " + ruleContext.getRule())
        .setMnemonic("JavaPackageManifest")
        .build(ruleContext);
  }

  private static Iterable<String> toSerializedArtifactLocations(Iterable<Artifact> artifacts) {
    return Iterables.transform(
        Iterables.filter(artifacts, Artifact.MIDDLEMAN_FILTER),
        PACKAGE_PARSER_SERIALIZER);
  }

  private static final Function<Artifact, String> PACKAGE_PARSER_SERIALIZER =
      new Function<Artifact, String>() {
        @Override
        public String apply(Artifact artifact) {
          ArtifactLocation location = makeArtifactLocation(artifact);
          return Joiner.on(",").join(
              location.getRootExecutionPathFragment(),
              location.getRelativePath(),
              location.getRootPath()
          );
        }
      };

  private static Artifact derivedArtifact(ConfiguredTarget base, RuleContext ruleContext,
      String suffix) {
    BuildConfiguration configuration = ruleContext.getConfiguration();
    assert configuration != null;
    Root genfilesDirectory = configuration.getGenfilesDirectory();

    PathFragment derivedFilePath =
        getOutputFilePath(base, ruleContext, suffix);

    return ruleContext.getAnalysisEnvironment().getDerivedArtifact(
        derivedFilePath, genfilesDirectory);
  }

  private static AndroidRuleIdeInfo makeAndroidRuleIdeInfo(
      AndroidIdeInfoProvider androidIdeInfoProvider,
      DependenciesResult dependenciesResult,
      NestedSetBuilder<Artifact> ideResolveArtifacts) {
    AndroidRuleIdeInfo.Builder builder = AndroidRuleIdeInfo.newBuilder();
    if (androidIdeInfoProvider.getSignedApk() != null) {
      builder.setApk(makeArtifactLocation(androidIdeInfoProvider.getSignedApk()));
    }

    Artifact manifest = androidIdeInfoProvider.getManifest();
    if (manifest != null) {
      builder.setManifest(makeArtifactLocation(manifest));
      addResolveArtifact(ideResolveArtifacts, manifest);
    }

    for (Artifact artifact : androidIdeInfoProvider.getApksUnderTest()) {
      builder.addDependencyApk(makeArtifactLocation(artifact));
    }
    for (SourceDirectory resourceDir : androidIdeInfoProvider.getResourceDirs()) {
      ArtifactLocation artifactLocation = makeArtifactLocation(resourceDir);
      builder.addResources(artifactLocation);
    }

    if (androidIdeInfoProvider.getJavaPackage() != null) {
      builder.setJavaPackage(androidIdeInfoProvider.getJavaPackage());
    }

    boolean hasIdlSources = !androidIdeInfoProvider.getIdlSrcs().isEmpty();
    builder.setHasIdlSources(hasIdlSources);
    if (hasIdlSources) {
      LibraryArtifact idlLibraryArtifact = makeLibraryArtifact(ideResolveArtifacts,
          androidIdeInfoProvider.getIdlClassJar(), null, androidIdeInfoProvider.getIdlSourceJar());
      if (idlLibraryArtifact != null) {
        builder.setIdlJar(idlLibraryArtifact);
      }
    }

    builder.setGenerateResourceClass(androidIdeInfoProvider.definesAndroidResources());

    if (dependenciesResult.resources != null) {
      builder.setLegacyResources(dependenciesResult.resources.toString());
    }

    return builder.build();
  }

  private static BinaryFileWriteAction makeProtoWriteAction(
      ActionOwner actionOwner, final MessageLite message, Artifact artifact) {
    return new BinaryFileWriteAction(
        actionOwner,
        artifact,
        new ByteSource() {
          @Override
          public InputStream openStream() throws IOException {
            return message.toByteString().newInput();
          }
        },
        /*makeExecutable =*/ false);
  }

  private static BinaryFileWriteAction makeProtoTextWriteAction(
      ActionOwner actionOwner, final MessageLite message, Artifact artifact) {
    return new BinaryFileWriteAction(
        actionOwner,
        artifact,
        new ByteSource() {
          @Override
          public InputStream openStream() throws IOException {
            return new ByteArrayInputStream(message.toString().getBytes(StandardCharsets.UTF_8));
          }
        },
        /*makeExecutable =*/ false);
  }

  public static ArtifactLocation makeArtifactLocation(Artifact artifact) {
    return makeArtifactLocation(artifact.getRoot(), artifact.getRootRelativePath());
  }

  private static ArtifactLocation makeArtifactLocation(Package pkg) {
    Root root = Root.asSourceRoot(pkg.getSourceRoot());
    PathFragment relativePath = pkg.getBuildFile().getPath().relativeTo(root.getPath());
    return makeArtifactLocation(root, relativePath);
  }

  private static ArtifactLocation makeArtifactLocation(Root root, PathFragment relativePath) {
    return ArtifactLocation.newBuilder()
        .setRootPath(root.getPath().toString())
        .setRootExecutionPathFragment(root.getExecPath().toString())
        .setRelativePath(relativePath.toString())
        .setIsSource(root.isSourceRoot())
        .build();
  }

  private static ArtifactLocation makeArtifactLocation(SourceDirectory resourceDir) {
    return ArtifactLocation.newBuilder()
        .setRootPath(resourceDir.getRootPath().toString())
        .setRootExecutionPathFragment(resourceDir.getRootExecutionPathFragment().toString())
        .setRelativePath(resourceDir.getRelativePath().toString())
        .setIsSource(resourceDir.isSource())
        .build();
  }

  private JavaRuleIdeInfo makeJavaRuleIdeInfo(
      ConfiguredTarget base,
      RuleContext ruleContext,
      JavaRuleOutputJarsProvider outputJarsProvider,
      NestedSetBuilder<Artifact> ideResolveArtifacts,
      @Nullable Artifact packageManifest) {
    JavaRuleIdeInfo.Builder builder = JavaRuleIdeInfo.newBuilder();
    collectJarsFromOutputJarsProvider(builder, ideResolveArtifacts, outputJarsProvider);

    Artifact jdeps = outputJarsProvider.getJdeps();
    if (jdeps != null) {
      builder.setJdeps(makeArtifactLocation(jdeps));
    }

    JavaGenJarsProvider genJarsProvider =
        base.getProvider(JavaGenJarsProvider.class);
    if (genJarsProvider != null) {
      collectGenJars(builder, ideResolveArtifacts, genJarsProvider);
    }

    Collection<Artifact> sourceFiles = getSources(ruleContext);

    for (Artifact sourceFile : sourceFiles) {
      builder.addSources(makeArtifactLocation(sourceFile));
    }

    if (packageManifest != null) {
      builder.setPackageManifest(makeArtifactLocation(packageManifest));
    }

    return builder.build();
  }

  private CRuleIdeInfo makeCRuleIdeInfo(
      ConfiguredTarget base,
      RuleContext ruleContext,
      CppCompilationContext cppCompilationContext,
      NestedSetBuilder<Artifact> ideResolveArtifacts) {
    CRuleIdeInfo.Builder builder = CRuleIdeInfo.newBuilder();

    Collection<Artifact> sourceFiles = getSources(ruleContext);
    for (Artifact sourceFile : sourceFiles) {
      builder.addSource(makeArtifactLocation(sourceFile));
    }

    builder.addAllRuleInclude(getIncludes(ruleContext));
    builder.addAllRuleDefine(getDefines(ruleContext));
    builder.addAllRuleCopt(getCopts(ruleContext));

    // Get information about from the transitive closure
    ImmutableList<PathFragment> transitiveIncludeDirectories =
        cppCompilationContext.getIncludeDirs();
    for (PathFragment pathFragment : transitiveIncludeDirectories) {
      builder.addTransitiveIncludeDirectory(pathFragment.getSafePathString());
    }
    ImmutableList<PathFragment> transitiveQuoteIncludeDirectories =
        cppCompilationContext.getQuoteIncludeDirs();
    for (PathFragment pathFragment : transitiveQuoteIncludeDirectories) {
      builder.addTransitiveQuoteIncludeDirectory(pathFragment.getSafePathString());
    }
    ImmutableList<String> transitiveDefines = cppCompilationContext.getDefines();
    for (String transitiveDefine : transitiveDefines) {
      builder.addTransitiveDefine(transitiveDefine);
    }
    ImmutableList<PathFragment> transitiveSystemIncludeDirectories =
        cppCompilationContext.getSystemIncludeDirs();
    for (PathFragment pathFragment : transitiveSystemIncludeDirectories) {
      builder.addTransitiveSystemIncludeDirectory(pathFragment.getSafePathString());
    }

    androidStudioInfoSemantics.augmentCppRuleInfo(
        builder, base, ruleContext, cppCompilationContext, ideResolveArtifacts);

    return builder.build();
  }

  private static CToolchainIdeInfo makeCToolchainIdeInfo(
      RuleContext ruleContext, CppConfiguration cppConfiguration) {
    CToolchainIdeInfo.Builder builder = CToolchainIdeInfo.newBuilder();
    ImmutableSet<String> features = ruleContext.getFeatures();
    builder.setTargetName(cppConfiguration.getTargetGnuSystemName());

    builder.addAllBaseCompilerOption(cppConfiguration.getCompilerOptions(features));
    builder.addAllCOption(cppConfiguration.getCOptions());
    builder.addAllCppOption(cppConfiguration.getCxxOptions(features));
    builder.addAllLinkOption(cppConfiguration.getLinkOptions());

    // This includes options such as system includes from toolchains.
    builder.addAllUnfilteredCompilerOption(
        cppConfiguration.getUnfilteredCompilerOptions(features));

    builder.setPreprocessorExecutable(
        cppConfiguration.getCpreprocessorExecutable().getSafePathString());
    builder.setCppExecutable(cppConfiguration.getCppExecutable().getSafePathString());

    List<PathFragment> builtInIncludeDirectories = cppConfiguration
        .getBuiltInIncludeDirectories();
    for (PathFragment builtInIncludeDirectory : builtInIncludeDirectories) {
      builder.addBuiltInIncludeDirectory(builtInIncludeDirectory.getSafePathString());
    }
    return builder.build();
  }

  private static void collectJarsFromOutputJarsProvider(
      JavaRuleIdeInfo.Builder builder,
      NestedSetBuilder<Artifact> ideResolveArtifacts,
      JavaRuleOutputJarsProvider outputJarsProvider) {
    for (OutputJar outputJar : outputJarsProvider.getOutputJars()) {
      LibraryArtifact libraryArtifact = makeLibraryArtifact(ideResolveArtifacts,
          outputJar.getClassJar(), outputJar.getIJar(), outputJar.getSrcJar());

      if (libraryArtifact != null) {
        builder.addJars(libraryArtifact);
      }
    }
  }

  @Nullable
  private static LibraryArtifact makeLibraryArtifact(
      NestedSetBuilder<Artifact> ideResolveArtifacts,
      @Nullable Artifact classJar,
      @Nullable Artifact iJar,
      @Nullable Artifact sourceJar
      ) {
    // We don't want to add anything that doesn't have a class jar
    if (classJar == null) {
      return null;
    }
    LibraryArtifact.Builder jarsBuilder = LibraryArtifact.newBuilder();
    jarsBuilder.setJar(makeArtifactLocation(classJar));
    addResolveArtifact(ideResolveArtifacts, classJar);

    if (iJar != null) {
      jarsBuilder.setInterfaceJar(makeArtifactLocation(iJar));
      addResolveArtifact(ideResolveArtifacts, iJar);
    }
    if (sourceJar != null) {
      jarsBuilder.setSourceJar(makeArtifactLocation(sourceJar));
      addResolveArtifact(ideResolveArtifacts, sourceJar);
    }

    return jarsBuilder.build();
  }

  private static void collectGenJars(
      JavaRuleIdeInfo.Builder builder,
      NestedSetBuilder<Artifact> ideResolveArtifacts,
      JavaGenJarsProvider genJarsProvider) {
    LibraryArtifact.Builder genjarsBuilder = LibraryArtifact.newBuilder();

    if (genJarsProvider.usesAnnotationProcessing()) {
      Artifact genClassJar = genJarsProvider.getGenClassJar();
      if (genClassJar != null) {
        genjarsBuilder.setJar(makeArtifactLocation(genClassJar));
        addResolveArtifact(ideResolveArtifacts, genClassJar);
      }
      Artifact gensrcJar = genJarsProvider.getGenSourceJar();
      if (gensrcJar != null) {
        genjarsBuilder.setSourceJar(makeArtifactLocation(gensrcJar));
        addResolveArtifact(ideResolveArtifacts, gensrcJar);
      }
      if (genjarsBuilder.hasJar()) {
        builder.addGeneratedJars(genjarsBuilder.build());
      }
    }
  }

  private static Collection<Artifact> getJavaSources(RuleContext ruleContext) {
    Collection<Artifact> srcs = getSources(ruleContext);
    List<Artifact> javaSrcs = Lists.newArrayList();
    for (Artifact src : srcs) {
      if (src.isSourceArtifact() && src.getRootRelativePathString().endsWith(".java")) {
        javaSrcs.add(src);
      }
    }
    return javaSrcs;
  }

  private static Collection<Artifact> getSources(RuleContext ruleContext) {
    return getTargetListAttribute(ruleContext, "srcs");
  }

  private static Collection<String> getIncludes(RuleContext ruleContext) {
    return getStringListAttribute(ruleContext, "includes");
  }

  private static Collection<String> getDefines(RuleContext ruleContext) {
    return getStringListAttribute(ruleContext, "defines");
  }

  private static Collection<String> getCopts(RuleContext ruleContext) {
    return getStringListAttribute(ruleContext, "copts");
  }

  private static Collection<Artifact> getTargetListAttribute(RuleContext ruleContext,
      String attributeName) {
    return (ruleContext.attributes().has(attributeName, BuildType.LABEL_LIST)
        && ruleContext.getAttributeMode(attributeName) == Mode.TARGET)
        ? ruleContext.getPrerequisiteArtifacts(attributeName, Mode.TARGET).list()
        : ImmutableList.<Artifact>of();
  }

  private static Collection<String> getStringListAttribute(RuleContext ruleContext,
      String attributeName) {
    return ruleContext.attributes().has(attributeName, Type.STRING_LIST)
        ? ruleContext.attributes().get(attributeName, Type.STRING_LIST)
        : ImmutableList.<String>of();
  }

  private static PathFragment getOutputFilePath(ConfiguredTarget base, RuleContext ruleContext,
      String suffix) {
    PathFragment packagePathFragment =
        ruleContext.getLabel().getPackageIdentifier().getPathFragment();
    String name = base.getLabel().getName();
    return new PathFragment(packagePathFragment, new PathFragment(name + suffix));
  }


  private static void addResolveArtifact(NestedSetBuilder<Artifact> ideResolveArtifacts,
      Artifact artifact) {
    if (!artifact.isSourceArtifact()) {
      ideResolveArtifacts.add(artifact);
    }
  }

  @Deprecated
  private RuleIdeInfo.Kind getRuleKind(Rule rule, ConfiguredTarget base) {
    switch (rule.getRuleClassObject().getName()) {
      case "java_library":
        return Kind.JAVA_LIBRARY;
      case "java_import":
        return Kind.JAVA_IMPORT;
      case "java_test":
        return Kind.JAVA_TEST;
      case "java_binary":
        return Kind.JAVA_BINARY;
      case "android_library":
        return Kind.ANDROID_LIBRARY;
      case "android_binary":
        return Kind.ANDROID_BINARY;
      case "android_test":
        return Kind.ANDROID_TEST;
      case "android_robolectric_test":
        return Kind.ANDROID_ROBOELECTRIC_TEST;
      case "proto_library":
        return Kind.PROTO_LIBRARY;
      case "java_plugin":
        return Kind.JAVA_PLUGIN;
      case "android_resources":
        return Kind.ANDROID_RESOURCES;
      case "cc_library":
        return Kind.CC_LIBRARY;
      case "cc_binary":
        return Kind.CC_BINARY;
      case "cc_test":
        return Kind.CC_TEST;
      case "cc_inc_library":
        return Kind.CC_INC_LIBRARY;
      case "cc_toolchain":
        return Kind.CC_TOOLCHAIN;
      case "java_wrap_cc":
        return Kind.JAVA_WRAP_CC;
      default:
      {
        if (base.getProvider(AndroidSdkProvider.class) != null) {
          return RuleIdeInfo.Kind.ANDROID_SDK;
        } else {
          return RuleIdeInfo.Kind.UNRECOGNIZED;
        }
      }
    }
  }
}
