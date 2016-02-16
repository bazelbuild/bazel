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
import com.google.devtools.build.lib.Constants;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.analysis.AnalysisUtils;
import com.google.devtools.build.lib.analysis.ConfiguredAspect;
import com.google.devtools.build.lib.analysis.ConfiguredAspect.Builder;
import com.google.devtools.build.lib.analysis.ConfiguredNativeAspectFactory;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.actions.BinaryFileWriteAction;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.ideinfo.androidstudio.AndroidStudioIdeInfo.AndroidRuleIdeInfo;
import com.google.devtools.build.lib.ideinfo.androidstudio.AndroidStudioIdeInfo.AndroidSdkRuleInfo;
import com.google.devtools.build.lib.ideinfo.androidstudio.AndroidStudioIdeInfo.ArtifactLocation;
import com.google.devtools.build.lib.ideinfo.androidstudio.AndroidStudioIdeInfo.JavaRuleIdeInfo;
import com.google.devtools.build.lib.ideinfo.androidstudio.AndroidStudioIdeInfo.LibraryArtifact;
import com.google.devtools.build.lib.ideinfo.androidstudio.AndroidStudioIdeInfo.RuleIdeInfo;
import com.google.devtools.build.lib.ideinfo.androidstudio.AndroidStudioIdeInfo.RuleIdeInfo.Kind;
import com.google.devtools.build.lib.packages.AspectDefinition;
import com.google.devtools.build.lib.packages.AspectParameters;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.rules.android.AndroidCommon;
import com.google.devtools.build.lib.rules.android.AndroidIdeInfoProvider;
import com.google.devtools.build.lib.rules.android.AndroidIdeInfoProvider.SourceDirectory;
import com.google.devtools.build.lib.rules.android.AndroidSdkProvider;
import com.google.devtools.build.lib.rules.android.LocalResourceContainer;
import com.google.devtools.build.lib.rules.java.JavaExportsProvider;
import com.google.devtools.build.lib.rules.java.JavaGenJarsProvider;
import com.google.devtools.build.lib.rules.java.JavaRuleOutputJarsProvider;
import com.google.devtools.build.lib.rules.java.JavaRuleOutputJarsProvider.OutputJar;
import com.google.devtools.build.lib.rules.java.JavaSourceInfoProvider;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.protobuf.MessageLite;

import java.io.IOException;
import java.io.InputStream;
import java.util.Collection;
import java.util.List;
import java.util.Set;

import javax.annotation.Nullable;

/**
 * Generates ide-build information for Android Studio.
 */
public class AndroidStudioInfoAspect implements ConfiguredNativeAspectFactory {
  public static final String NAME = "AndroidStudioInfoAspect";

  // Output groups.

  public static final String IDE_INFO = "ide-info";
  public static final String IDE_INFO_TEXT = "ide-info-text";
  public static final String IDE_RESOLVE = "ide-resolve";

  private static class PrerequisiteAttr {
    public final String name;
    public final Type<?> type;
    public PrerequisiteAttr(String name, Type<?> type) {
      this.name = name;
      this.type = type;
    }
  }
  public static final PrerequisiteAttr[] PREREQUISITE_ATTRS = {
      new PrerequisiteAttr("deps", BuildType.LABEL_LIST),
      new PrerequisiteAttr("exports", BuildType.LABEL_LIST),
      new PrerequisiteAttr("$robolectric", BuildType.LABEL_LIST), // From android_robolectric_test
      new PrerequisiteAttr("$junit", BuildType.LABEL), // From android_robolectric_test
      new PrerequisiteAttr("binary_under_test", BuildType.LABEL), // From android_test
      new PrerequisiteAttr("java_lib", BuildType.LABEL), // From proto_library
      new PrerequisiteAttr("$proto1_java_lib", BuildType.LABEL), // From proto_library
  };

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

  /** White-list for rules potentially having .java srcs */
  private static final Set<Kind> JAVA_SRC_RULES = ImmutableSet.of(
      Kind.JAVA_LIBRARY,
      Kind.JAVA_TEST,
      Kind.JAVA_BINARY,
      Kind.ANDROID_LIBRARY,
      Kind.ANDROID_BINARY,
      Kind.ANDROID_TEST,
      Kind.ANDROID_ROBOELECTRIC_TEST,
      Kind.JAVA_PLUGIN);

  @Override
  public AspectDefinition getDefinition(AspectParameters aspectParameters) {
    AspectDefinition.Builder builder = new AspectDefinition.Builder(NAME)
        .attributeAspect("runtime_deps", AndroidStudioInfoAspect.class)
        .add(attr("$packageParser", LABEL).cfg(HOST).exec()
            .value(Label.parseAbsoluteUnchecked(
                Constants.TOOLS_REPOSITORY + "//tools/android:PackageParser")));

    for (PrerequisiteAttr prerequisiteAttr : PREREQUISITE_ATTRS) {
      builder.attributeAspect(prerequisiteAttr.name, AndroidStudioInfoAspect.class);
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
        base, ruleContext, providerBuilder, ruleKind);

    AndroidStudioInfoFilesProvider provider;
    if (ruleKind != RuleIdeInfo.Kind.UNRECOGNIZED) {
      provider =
          createIdeBuildArtifact(
              base,
              ruleContext,
              ruleKind,
              dependenciesResult.deps,
              dependenciesResult.runtimeDeps,
              providerBuilder);
    } else {
      provider = providerBuilder.build();
    }

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
        Iterable<Label> runtimeDeps) {
      this.deps = deps;
      this.runtimeDeps = runtimeDeps;
    }
    final Iterable<Label> deps;
    final Iterable<Label> runtimeDeps;
  }

  private DependenciesResult processDependencies(
      ConfiguredTarget base, RuleContext ruleContext,
      AndroidStudioInfoFilesProvider.Builder providerBuilder, RuleIdeInfo.Kind ruleKind) {

    // Calculate direct dependencies
    ImmutableList.Builder<TransitiveInfoCollection> directDepsBuilder = ImmutableList.builder();
    for (PrerequisiteAttr prerequisiteAttr : PREREQUISITE_ATTRS) {
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
    if (ruleKind == Kind.ANDROID_LIBRARY) {
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

    // Propagate providers from all prerequisites (deps + runtime_deps)
    ImmutableList.Builder<TransitiveInfoCollection> prerequisitesBuilder = ImmutableList.builder();
    prerequisitesBuilder.addAll(directDeps);
    prerequisitesBuilder.addAll(runtimeDeps);

    List<TransitiveInfoCollection> prerequisites = prerequisitesBuilder.build();

    for (AndroidStudioInfoFilesProvider depProvider :
        AnalysisUtils.getProviders(prerequisites, AndroidStudioInfoFilesProvider.class)) {
      providerBuilder.ideInfoFilesBuilder().addTransitive(depProvider.getIdeInfoFiles());
      providerBuilder.ideInfoTextFilesBuilder().addTransitive(depProvider.getIdeInfoTextFiles());
      providerBuilder.ideResolveFilesBuilder().addTransitive(depProvider.getIdeResolveFiles());
    }

    return new DependenciesResult(dependencies, runtimeDepsBuilder.build());
  }

  private static AndroidSdkRuleInfo makeAndroidSdkRuleInfo(AndroidSdkProvider provider) {
    AndroidSdkRuleInfo.Builder sdkInfoBuilder = AndroidSdkRuleInfo.newBuilder();

    Path androidSdkDirectory = provider.getAndroidJar().getPath().getParentDirectory();
    sdkInfoBuilder.setAndroidSdkPath(androidSdkDirectory.toString());

    return sdkInfoBuilder.build();
  }

  private AndroidStudioInfoFilesProvider createIdeBuildArtifact(
      ConfiguredTarget base,
      RuleContext ruleContext,
      Kind ruleKind,
      Iterable<Label> directDependencies,
      Iterable<Label> runtimeDeps,
      AndroidStudioInfoFilesProvider.Builder providerBuilder) {

    Artifact ideInfoFile = derivedArtifact(base, ruleContext, ASWB_BUILD_SUFFIX);
    Artifact ideInfoTextFile = derivedArtifact(base, ruleContext, ASWB_BUILD_TEXT_SUFFIX);
    Artifact packageManifest = createPackageManifest(base, ruleContext, ruleKind);
    providerBuilder.ideInfoFilesBuilder().add(ideInfoFile);
    providerBuilder.ideInfoTextFilesBuilder().add(ideInfoTextFile);
    if (packageManifest != null) {
      providerBuilder.ideInfoFilesBuilder().add(packageManifest);
    }
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

    outputBuilder.setKind(ruleKind);

    if (ruleKind == Kind.JAVA_LIBRARY
        || ruleKind == Kind.JAVA_IMPORT
        || ruleKind == Kind.JAVA_TEST
        || ruleKind == Kind.JAVA_BINARY
        || ruleKind == Kind.ANDROID_LIBRARY
        || ruleKind == Kind.ANDROID_BINARY
        || ruleKind == Kind.ANDROID_TEST
        || ruleKind == Kind.ANDROID_ROBOELECTRIC_TEST
        || ruleKind == Kind.PROTO_LIBRARY
        || ruleKind == Kind.JAVA_PLUGIN) {
      JavaRuleIdeInfo javaRuleIdeInfo = makeJavaRuleIdeInfo(
          base, ruleContext, ideResolveArtifacts, packageManifest);
      outputBuilder.setJavaRuleIdeInfo(javaRuleIdeInfo);
    }
    if (ruleKind == Kind.ANDROID_LIBRARY
        || ruleKind == Kind.ANDROID_BINARY
        || ruleKind == Kind.ANDROID_TEST) {
      outputBuilder.setAndroidRuleIdeInfo(
          makeAndroidRuleIdeInfo(ruleContext, base, ideResolveArtifacts));
    }
    if (ruleKind == Kind.ANDROID_SDK) {
      outputBuilder.setAndroidSdkRuleInfo(
          makeAndroidSdkRuleInfo(base.getProvider(AndroidSdkProvider.class)));
    }

    AndroidStudioInfoFilesProvider provider = providerBuilder.build();

    outputBuilder.addAllDependencies(transform(directDependencies, LABEL_TO_STRING));
    outputBuilder.addAllRuntimeDeps(transform(runtimeDeps, LABEL_TO_STRING));
    outputBuilder.addAllTags(base.getTarget().getAssociatedRule().getRuleTags());

    final RuleIdeInfo ruleIdeInfo = outputBuilder.build();
    ruleContext.registerAction(
        makeProtoWriteAction(ruleContext.getActionOwner(), ruleIdeInfo, ideInfoFile));
    ruleContext.registerAction(
        makeProtoTextWriteAction(ruleContext.getActionOwner(), ruleIdeInfo, ideInfoTextFile));
    if (packageManifest != null) {
      ruleContext.registerAction(
          makePackageManifestAction(ruleContext, packageManifest, getJavaSources(ruleContext))
      );
    }

    return provider;
  }

  @Nullable private static Artifact createPackageManifest(ConfiguredTarget base,
      RuleContext ruleContext, Kind ruleKind) {
    if (!JAVA_SRC_RULES.contains(ruleKind)) {
      return null;
    }
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
      RuleContext ruleContext,
      ConfiguredTarget base,
      NestedSetBuilder<Artifact> ideResolveArtifacts) {
    AndroidRuleIdeInfo.Builder builder = AndroidRuleIdeInfo.newBuilder();
    AndroidIdeInfoProvider provider = base.getProvider(AndroidIdeInfoProvider.class);
    assert provider != null;
    if (provider.getSignedApk() != null) {
      builder.setApk(makeArtifactLocation(provider.getSignedApk()));
    }

    Artifact manifest = provider.getManifest();
    if (manifest != null) {
      builder.setManifest(makeArtifactLocation(manifest));
      addResolveArtifact(ideResolveArtifacts, manifest);
    }

    for (Artifact artifact : provider.getApksUnderTest()) {
      builder.addDependencyApk(makeArtifactLocation(artifact));
    }
    for (SourceDirectory resourceDir : provider.getResourceDirs()) {
      ArtifactLocation artifactLocation = makeArtifactLocation(resourceDir);
      builder.addResources(artifactLocation);
    }

    builder.setJavaPackage(AndroidCommon.getJavaPackage(ruleContext));

    boolean hasIdlSources = !provider.getIdlSrcs().isEmpty();
    builder.setHasIdlSources(hasIdlSources);
    if (hasIdlSources) {
      LibraryArtifact.Builder jarBuilder = LibraryArtifact.newBuilder();
      Artifact idlClassJar = provider.getIdlClassJar();
      if (idlClassJar != null) {
        jarBuilder.setJar(makeArtifactLocation(idlClassJar));
        addResolveArtifact(ideResolveArtifacts, idlClassJar);
      }
      Artifact idlSourceJar = provider.getIdlSourceJar();
      if (idlSourceJar != null) {
        jarBuilder.setSourceJar(makeArtifactLocation(idlSourceJar));
        addResolveArtifact(ideResolveArtifacts, idlSourceJar);
      }
      if (idlClassJar != null) {
        builder.setIdlJar(jarBuilder.build());
      }
    }

    builder.setGenerateResourceClass(
        LocalResourceContainer.definesAndroidResources(ruleContext.attributes()));

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

  private static FileWriteAction makeProtoTextWriteAction(
      ActionOwner actionOwner, final MessageLite message, Artifact artifact) {
    return new FileWriteAction(
        actionOwner,
        artifact,
        message.toString(),
        /*makeExecutable =*/ false);
  }

  private static ArtifactLocation makeArtifactLocation(Artifact artifact) {
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

  private static JavaRuleIdeInfo makeJavaRuleIdeInfo(
      ConfiguredTarget base,
      RuleContext ruleContext,
      NestedSetBuilder<Artifact> ideResolveArtifacts,
      @Nullable Artifact packageManifest) {
    JavaRuleIdeInfo.Builder builder = JavaRuleIdeInfo.newBuilder();
    JavaRuleOutputJarsProvider outputJarsProvider =
        base.getProvider(JavaRuleOutputJarsProvider.class);
    if (outputJarsProvider != null) {
      // java_library
      collectJarsFromOutputJarsProvider(builder, ideResolveArtifacts, outputJarsProvider);

      Artifact jdeps = outputJarsProvider.getJdeps();
      if (jdeps != null) {
        builder.setJdeps(makeArtifactLocation(jdeps));
      }
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

  private static void collectJarsFromOutputJarsProvider(
      JavaRuleIdeInfo.Builder builder,
      NestedSetBuilder<Artifact> ideResolveArtifacts,
      JavaRuleOutputJarsProvider outputJarsProvider) {
    LibraryArtifact.Builder jarsBuilder = LibraryArtifact.newBuilder();
    for (OutputJar outputJar : outputJarsProvider.getOutputJars()) {
      Artifact classJar = outputJar.getClassJar();
      if (classJar != null) {
        jarsBuilder.setJar(makeArtifactLocation(classJar));
        addResolveArtifact(ideResolveArtifacts, classJar);
      }
      Artifact iJar = outputJar.getIJar();
      if (iJar != null) {
        jarsBuilder.setInterfaceJar(makeArtifactLocation(iJar));
        addResolveArtifact(ideResolveArtifacts, iJar);
      }
      Artifact srcJar = outputJar.getSrcJar();
      if (srcJar != null) {
        jarsBuilder.setSourceJar(makeArtifactLocation(srcJar));
        addResolveArtifact(ideResolveArtifacts, srcJar);
      }

      // We don't want to add anything that doesn't have a class jar
      if (classJar != null) {
        builder.addJars(jarsBuilder.build());
      }
    }
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
      if (src.getRootRelativePathString().endsWith(".java")) {
        javaSrcs.add(src);
      }
    }
    return javaSrcs;
  }

  private static Collection<Artifact> getSources(RuleContext ruleContext) {
    return ruleContext.attributes().has("srcs", BuildType.LABEL_LIST)
        ? ruleContext.getPrerequisiteArtifacts("srcs", Mode.TARGET).list()
        : ImmutableList.<Artifact>of();
  }

  private static PathFragment getOutputFilePath(ConfiguredTarget base, RuleContext ruleContext,
      String suffix) {
    PathFragment packagePathFragment =
        ruleContext.getLabel().getPackageIdentifier().getPathFragment();
    String name = base.getLabel().getName();
    return new PathFragment(packagePathFragment, new PathFragment(name + suffix));
  }

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

  private static void addResolveArtifact(NestedSetBuilder<Artifact> ideResolveArtifacts,
      Artifact artifact) {
    if (!artifact.isSourceArtifact()) {
      ideResolveArtifacts.add(artifact);
    }
  }
}
