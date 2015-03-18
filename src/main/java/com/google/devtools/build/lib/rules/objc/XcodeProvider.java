// Copyright 2014 Google Inc. All rights reserved.
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

package com.google.devtools.build.lib.rules.objc;

import static com.google.devtools.build.lib.rules.objc.ObjcProvider.BUNDLE_IMPORT_DIR;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.DEFINE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.FORCE_LOAD_FOR_XCODEGEN;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.FRAMEWORK_DIR;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.GENERAL_RESOURCE_FILE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.IMPORTED_LIBRARY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.SDK_DYLIB;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.SDK_FRAMEWORK;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.WEAK_SDK_FRAMEWORK;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.XCASSETS_DIR;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.XCDATAMODEL;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Function;
import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.rules.objc.ObjcProvider.Flag;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.xcode.util.Interspersing;
import com.google.devtools.build.xcode.xcodegen.proto.XcodeGenProtos.DependencyControl;
import com.google.devtools.build.xcode.xcodegen.proto.XcodeGenProtos.TargetControl;
import com.google.devtools.build.xcode.xcodegen.proto.XcodeGenProtos.XcodeprojBuildSetting;

import java.util.Arrays;
import java.util.EnumSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * Provider which provides transitive dependency information that is specific to Xcodegen. In
 * particular, it provides a sequence of targets which can be used to create a self-contained
 * {@code .xcodeproj} file.
 */
@Immutable
public final class XcodeProvider implements TransitiveInfoProvider {
  /**
   * A builder for instances of {@link XcodeProvider}.
   */
  public static final class Builder {
    private Label label;
    private final NestedSetBuilder<String> userHeaderSearchPaths = NestedSetBuilder.stableOrder();
    private final NestedSetBuilder<String> headerSearchPaths = NestedSetBuilder.stableOrder();
    private Optional<InfoplistMerging> infoplistMerging = Optional.absent();
    private final NestedSetBuilder<XcodeProvider> dependencies = NestedSetBuilder.stableOrder();
    private final ImmutableList.Builder<XcodeprojBuildSetting> xcodeprojBuildSettings =
        new ImmutableList.Builder<>();
    private final ImmutableList.Builder<String> copts = new ImmutableList.Builder<>();
    private final ImmutableList.Builder<String> compilationModeCopts =
        new ImmutableList.Builder<>();
    private XcodeProductType productType;
    private final ImmutableList.Builder<Artifact> headers = new ImmutableList.Builder<>();
    private Optional<CompilationArtifacts> compilationArtifacts = Optional.absent();
    private ObjcProvider objcProvider;
    private Optional<XcodeProvider> testHost = Optional.absent();
    private final NestedSetBuilder<Artifact> inputsToXcodegen = NestedSetBuilder.stableOrder();
    private final NestedSetBuilder<Artifact> additionalSources = NestedSetBuilder.stableOrder();
    private final ImmutableList.Builder<XcodeProvider> extensions = new ImmutableList.Builder<>();
    private String architecture;

    /**
     * Sets the label of the build target which corresponds to this Xcode target.
     */
    public Builder setLabel(Label label) {
      this.label = label;
      return this;
    }

    /**
     * Adds user header search paths for this target.
     */
    public Builder addUserHeaderSearchPaths(Iterable<PathFragment> userHeaderSearchPaths) {
      this.userHeaderSearchPaths.addAll(rootEach("$(WORKSPACE_ROOT)", userHeaderSearchPaths));
      return this;
    }

    /**
     * Adds header search paths for this target. Each path is interpreted relative to the given
     * root, such as {@code "$(WORKSPACE_ROOT)"}.
     */
    public Builder addHeaderSearchPaths(String root, Iterable<PathFragment> paths) {
      this.headerSearchPaths.addAll(rootEach(root, paths));
      return this;
    }

    /**
     * Sets the Info.plist merging information. Used for applications. May be
     * absent for other bundles.
     */
    public Builder setInfoplistMerging(InfoplistMerging infoplistMerging) {
      this.infoplistMerging = Optional.of(infoplistMerging);
      return this;
    }

    /**
     * Adds items in the {@link NestedSet}s of the given target to the corresponding sets in this
     * builder. This is useful if the given target is a dependency or like a dependency
     * (e.g. a test host). The given provider is not registered as a dependency with this provider.
     */
    private void addTransitiveSets(XcodeProvider dependencyish) {
      inputsToXcodegen.addTransitive(dependencyish.inputsToXcodegen);
      userHeaderSearchPaths.addTransitive(dependencyish.userHeaderSearchPaths);
      headerSearchPaths.addTransitive(dependencyish.headerSearchPaths);
    }

    /**
     * Adds {@link XcodeProvider}s corresponding to direct dependencies of this target which should
     * be added in the {@code .xcodeproj} file.
     */
    public Builder addDependencies(
        Iterable<XcodeProvider> dependencies, ObjcConfiguration configuration) {
      String architecture = getDependencyArchitecture(configuration);
      for (XcodeProvider dependency : dependencies) {
        // TODO(bazel-team): This is messy. Maybe we should make XcodeProvider be able to specify
        // how to depend on it rather than require this method to choose based on the dependency's
        // type.
        if (dependency.productType == XcodeProductType.EXTENSION) {
          this.extensions.add(dependency);
          this.inputsToXcodegen.addTransitive(dependency.inputsToXcodegen);
        } else if (dependency.architecture.equals(architecture)) {
          this.dependencies.add(dependency);
          this.dependencies.addTransitive(dependency.dependencies);
          this.addTransitiveSets(dependency);
        }
      }
      return this;
    }

    /**
     * Returns the architecture for which we keep dependencies.
     *
     * <p>When building with multiple architectures we want to avoid duplicating the same target for
     * each architecture. Instead we pick one architecture for which to keep all dependencies and
     * discard any others.
     */
    private String getDependencyArchitecture(ObjcConfiguration configuration) {
      List<String> iosMultiCpus = configuration.getIosMultiCpus();
      String architecture = configuration.getIosCpu();
      if (!iosMultiCpus.isEmpty()) {
        architecture = iosMultiCpus.get(0);
      }
      return architecture;
    }

    /**
     * Adds additional build settings of this target.
     */
    public Builder addXcodeprojBuildSettings(
        Iterable<XcodeprojBuildSetting> xcodeprojBuildSettings) {
      this.xcodeprojBuildSettings.addAll(xcodeprojBuildSettings);
      return this;
    }

    /**
     * Sets the copts to use when compiling the Xcode target.
     */
    public Builder addCopts(Iterable<String> copts) {
      this.copts.addAll(copts);
      return this;
    }

    /**
     * Sets the copts derived from compilation mode to use when compiling the Xcode target. These
     * will be included before the DEFINE options.
     */
    public Builder addCompilationModeCopts(Iterable<String> copts) {
      this.compilationModeCopts.addAll(copts);
      return this;
    }

    /**
     * Sets the product type for the PBXTarget in the .xcodeproj file.
     */
    public Builder setProductType(XcodeProductType productType) {
      this.productType = productType;
      return this;
    }

    /**
     * Adds to the header files of this target. It needs not to include the header files of
     * dependencies.
     */
    public Builder addHeaders(Iterable<Artifact> headers) {
      this.headers.addAll(headers);
      return this;
    }

    /**
     * The compilation artifacts for this target.
     */
    public Builder setCompilationArtifacts(CompilationArtifacts compilationArtifacts) {
      this.compilationArtifacts = Optional.of(compilationArtifacts);
      return this;
    }

    /**
     * Any additional sources not included in {@link #setCompilationArtifacts}.
     */
    public Builder addAdditionalSources(Artifact... artifacts) {
      this.additionalSources.addAll(Arrays.asList(artifacts));
      return this;
    }

    /**
     * Sets the {@link ObjcProvider} corresponding to this target.
     */
    public Builder setObjcProvider(ObjcProvider objcProvider) {
      this.objcProvider = objcProvider;
      return this;
    }

    /**
     * Sets the test host. This is used for xctest targets.
     */
    public Builder setTestHost(XcodeProvider testHost) {
      Preconditions.checkState(!this.testHost.isPresent());
      this.testHost = Optional.of(testHost);
      this.addTransitiveSets(testHost);
      return this;
    }

    /**
     * Adds inputs that are passed to Xcodegen when generating the project file.
     */
    public Builder addInputsToXcodegen(Iterable<Artifact> inputsToXcodegen) {
      this.inputsToXcodegen.addAll(inputsToXcodegen);
      return this;
    }

    /**
     * Sets the CPU architecture this xcode target was constructed for, derived from
     * {@link ObjcConfiguration#getIosCpu()}.
     */
    public Builder setArchitecture(String architecture) {
      this.architecture = architecture;
      return this;
    }

    public XcodeProvider build() {
      Preconditions.checkState(
          !testHost.isPresent() || (productType == XcodeProductType.UNIT_TEST),
          "%s product types cannot have a test host (test host: %s).", productType, testHost);
      return new XcodeProvider(this);
    }
  }

  /**
   * A collection of top-level targets that can be used to create a complete project.
   */
  public static final class Project {
    private final NestedSet<Artifact> inputsToXcodegen;
    private final ImmutableList<XcodeProvider> topLevelTargets;

    private Project(
        NestedSet<Artifact> inputsToXcodegen, ImmutableList<XcodeProvider> topLevelTargets) {
      this.inputsToXcodegen = inputsToXcodegen;
      this.topLevelTargets = topLevelTargets;
    }

    public static Project fromTopLevelTarget(XcodeProvider topLevelTarget) {
      return fromTopLevelTargets(ImmutableList.of(topLevelTarget));
    }

    public static Project fromTopLevelTargets(Iterable<XcodeProvider> topLevelTargets) {
      NestedSetBuilder<Artifact> inputsToXcodegen = NestedSetBuilder.stableOrder();
      for (XcodeProvider target : topLevelTargets) {
        inputsToXcodegen.addTransitive(target.inputsToXcodegen);
      }
      return new Project(inputsToXcodegen.build(), ImmutableList.copyOf(topLevelTargets));
    }

    /**
     * Returns artifacts that are passed to the Xcodegen action when generating a project file that
     * contains all of the given targets.
     */
    public NestedSet<Artifact> getInputsToXcodegen() {
      return inputsToXcodegen;
    }

    /**
     * Returns all the target controls that must be added to the xcodegen control. No other target
     * controls are needed to generate a functional project file. This method creates a new list
     * whenever it is called.
     */
    public ImmutableList<TargetControl> targets() {
      // Collect all the dependencies of all the providers, filtering out duplicates.
      Set<XcodeProvider> providerSet = new LinkedHashSet<>();
      for (XcodeProvider target : topLevelTargets) {
        target.collectProviders(providerSet);
      }

      ImmutableList.Builder<TargetControl> controls = new ImmutableList.Builder<>();
      for (XcodeProvider provider : providerSet) {
        controls.add(provider.targetControl());
      }
      return controls.build();
    }
  }

  private final Label label;
  private final NestedSet<String> userHeaderSearchPaths;
  private final NestedSet<String> headerSearchPaths;
  private final Optional<InfoplistMerging> infoplistMerging;
  private final NestedSet<XcodeProvider> dependencies;
  private final ImmutableList<XcodeprojBuildSetting> xcodeprojBuildSettings;
  private final ImmutableList<String> copts;
  private final ImmutableList<String> compilationModeCopts;
  private final XcodeProductType productType;
  private final ImmutableList<Artifact> headers;
  private final Optional<CompilationArtifacts> compilationArtifacts;
  private final ObjcProvider objcProvider;
  private final Optional<XcodeProvider> testHost;
  private final NestedSet<Artifact> inputsToXcodegen;
  private final NestedSet<Artifact> additionalSources;
  private final ImmutableList<XcodeProvider> extensions;
  private final String architecture;

  private XcodeProvider(Builder builder) {
    this.label = Preconditions.checkNotNull(builder.label);
    this.userHeaderSearchPaths = builder.userHeaderSearchPaths.build();
    this.headerSearchPaths = builder.headerSearchPaths.build();
    this.infoplistMerging = builder.infoplistMerging;
    this.dependencies = builder.dependencies.build();
    this.xcodeprojBuildSettings = builder.xcodeprojBuildSettings.build();
    this.copts = builder.copts.build();
    this.compilationModeCopts = builder.compilationModeCopts.build();
    this.productType = Preconditions.checkNotNull(builder.productType);
    this.headers = builder.headers.build();
    this.compilationArtifacts = builder.compilationArtifacts;
    this.objcProvider = Preconditions.checkNotNull(builder.objcProvider);
    this.testHost = Preconditions.checkNotNull(builder.testHost);
    this.inputsToXcodegen = builder.inputsToXcodegen.build();
    this.additionalSources = builder.additionalSources.build();
    this.extensions = builder.extensions.build();
    this.architecture = Preconditions.checkNotNull(builder.architecture);
  }

  /**
   * Creates a builder whose values are all initialized to this provider.
   */
  public Builder toBuilder() {
    Builder builder = new Builder();
    builder.label = label;
    builder.userHeaderSearchPaths.addAll(userHeaderSearchPaths);
    builder.headerSearchPaths.addTransitive(headerSearchPaths);
    builder.infoplistMerging = infoplistMerging;
    builder.dependencies.addTransitive(dependencies);
    builder.xcodeprojBuildSettings.addAll(xcodeprojBuildSettings);
    builder.copts.addAll(copts);
    builder.productType = productType;
    builder.headers.addAll(headers);
    builder.compilationArtifacts = compilationArtifacts;
    builder.objcProvider = objcProvider;
    builder.testHost = testHost;
    builder.inputsToXcodegen.addTransitive(inputsToXcodegen);
    builder.extensions.addAll(extensions);
    return builder;
  }

  private void collectProviders(Set<XcodeProvider> allProviders) {
    if (allProviders.add(this)) {
      for (XcodeProvider dependency : dependencies) {
        dependency.collectProviders(allProviders);
      }
      for (XcodeProvider justTestHost : testHost.asSet()) {
        justTestHost.collectProviders(allProviders);
      }
      for (XcodeProvider extension : extensions) {
        extension.collectProviders(allProviders);
      }
    }
  }

  private static final EnumSet<XcodeProductType> CAN_LINK_PRODUCT_TYPES = EnumSet.of(
      XcodeProductType.APPLICATION, XcodeProductType.BUNDLE, XcodeProductType.UNIT_TEST,
      XcodeProductType.EXTENSION);

  private TargetControl targetControl() {
    String buildFilePath = label.getPackageFragment().getSafePathString() + "/BUILD";
    // TODO(bazel-team): Add provisioning profile information when Xcodegen supports it.
    TargetControl.Builder targetControl = TargetControl.newBuilder()
        .setName(label.getName())
        .setLabel(label.toString())
        .setProductType(productType.getIdentifier())
        .addAllImportedLibrary(Artifact.toExecPaths(objcProvider.get(IMPORTED_LIBRARY)))
        .addAllUserHeaderSearchPath(userHeaderSearchPaths)
        .addAllHeaderSearchPath(headerSearchPaths)
        .addAllSupportFile(Artifact.toExecPaths(headers))
        .addAllCopt(compilationModeCopts)
        .addAllCopt(Interspersing.prependEach("-D", objcProvider.get(DEFINE)))
        .addAllCopt(copts)
        .addAllLinkopt(
            Interspersing.beforeEach("-force_load", objcProvider.get(FORCE_LOAD_FOR_XCODEGEN)))
        .addAllLinkopt(IosSdkCommands.DEFAULT_LINKER_FLAGS)
        .addAllLinkopt(Interspersing.beforeEach(
            "-weak_framework", SdkFramework.names(objcProvider.get(WEAK_SDK_FRAMEWORK))))
        .addAllBuildSetting(xcodeprojBuildSettings)
        .addAllBuildSetting(IosSdkCommands.defaultWarningsForXcode())
        .addAllSdkFramework(SdkFramework.names(objcProvider.get(SDK_FRAMEWORK)))
        .addAllFramework(PathFragment.safePathStrings(objcProvider.get(FRAMEWORK_DIR)))
        .addAllXcassetsDir(PathFragment.safePathStrings(objcProvider.get(XCASSETS_DIR)))
        .addAllXcdatamodel(PathFragment.safePathStrings(
            Xcdatamodel.xcdatamodelDirs(objcProvider.get(XCDATAMODEL))))
        .addAllBundleImport(PathFragment.safePathStrings(objcProvider.get(BUNDLE_IMPORT_DIR)))
        .addAllSdkDylib(objcProvider.get(SDK_DYLIB))
        .addAllGeneralResourceFile(Artifact.toExecPaths(objcProvider.get(GENERAL_RESOURCE_FILE)))
        .addSupportFile(buildFilePath);

    if (CAN_LINK_PRODUCT_TYPES.contains(productType)) {
      for (XcodeProvider dependency : dependencies) {
        // Only add a library target to a binary's dependencies if it has source files to compile.
        // Xcode cannot build targets without a source file in the PBXSourceFilesBuildPhase, so if
        // such a target is present in the control file, it is only to get Xcodegen to put headers
        // and resources not used by the final binary in the Project Navigator.
        //
        // The exceptions to this rule are objc_bundle_library and ios_extension targets. Bundles
        // are generally used for resources and can lack a PBXSourceFilesBuildPhase in the project
        // file and still be considered valid by Xcode.
        //
        // ios_extension targets are an exception because they have no CompilationArtifact object
        // but do have a dummy source file to make Xcode happy.
        boolean hasSources = dependency.compilationArtifacts.isPresent()
            && dependency.compilationArtifacts.get().getArchive().isPresent();
        if (hasSources || (dependency.productType == XcodeProductType.BUNDLE)) {
            targetControl.addDependency(DependencyControl.newBuilder()
                .setTargetLabel(dependency.label.toString())
                .build());
        }
      }
    }
    for (XcodeProvider justTestHost : testHost.asSet()) {
      targetControl.addDependency(DependencyControl.newBuilder()
          .setTargetLabel(justTestHost.label.toString())
          .setTestHost(true)
          .build());
    }
    for (XcodeProvider extension : extensions) {
      targetControl.addDependency(DependencyControl.newBuilder()
          .setTargetLabel(extension.label.toString())
          .build());
    }

    for (InfoplistMerging merging : infoplistMerging.asSet()) {
      for (Artifact infoplist : merging.getPlistWithEverything().asSet()) {
        targetControl.setInfoplist(infoplist.getExecPathString());
      }
    }
    for (CompilationArtifacts artifacts : compilationArtifacts.asSet()) {
      targetControl
          .addAllSourceFile(Artifact.toExecPaths(artifacts.getSrcs()))
          .addAllNonArcSourceFile(Artifact.toExecPaths(artifacts.getNonArcSrcs()));

      for (Artifact pchFile : artifacts.getPchFile().asSet()) {
        targetControl
            .setPchPath(pchFile.getExecPathString())
            .addSupportFile(pchFile.getExecPathString());
      }
    }

    for (Artifact artifact : additionalSources) {
      targetControl.addSourceFile(artifact.getExecPathString());
    }

    if (objcProvider.is(Flag.USES_CPP)) {
      targetControl.addSdkDylib("libc++");
    }

    return targetControl.build();
  }

  /**
   * Prepends the given path to each path in {@code paths}. Empty paths are
   * transformed to the value of {@code variable} rather than {@code variable + "/."}
   */
  @VisibleForTesting
  static Iterable<String> rootEach(final String prefix, Iterable<PathFragment> paths) {
    Preconditions.checkArgument(prefix.startsWith("$"),
        "prefix should start with a build setting variable like '$(NAME)': %s", prefix);
    Preconditions.checkArgument(!prefix.endsWith("/"),
        "prefix should not end with '/': %s", prefix);
    return Iterables.transform(paths, new Function<PathFragment, String>() {
      @Override
      public String apply(PathFragment input) {
        if (input.getSafePathString().equals(".")) {
          return prefix;
        } else {
          return prefix + "/" + input.getSafePathString();
        }
      }
    });
  }
}
