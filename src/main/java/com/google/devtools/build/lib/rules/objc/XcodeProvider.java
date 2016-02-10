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

package com.google.devtools.build.lib.rules.objc;

import static com.google.devtools.build.lib.rules.objc.ObjcProvider.BUNDLE_IMPORT_DIR;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.DEFINE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.FORCE_LOAD_FOR_XCODEGEN;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.FRAMEWORK_DIR;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.GENERAL_RESOURCE_DIR;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.GENERAL_RESOURCE_FILE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.IMPORTED_LIBRARY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.SDK_DYLIB;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.SDK_FRAMEWORK;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.WEAK_SDK_FRAMEWORK;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.XCASSETS_DIR;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.XCDATAMODEL;
import static com.google.devtools.build.lib.rules.objc.XcodeProductType.LIBRARY_STATIC;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Function;
import com.google.common.base.Joiner;
import com.google.common.base.Optional;
import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.rules.apple.AppleToolchain;
import com.google.devtools.build.lib.rules.objc.ObjcProvider.Flag;
import com.google.devtools.build.lib.rules.objc.ReleaseBundlingSupport.SplitArchTransition.ConfigurationDistinguisher;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.xcode.xcodegen.proto.XcodeGenProtos.DependencyControl;
import com.google.devtools.build.xcode.xcodegen.proto.XcodeGenProtos.TargetControl;
import com.google.devtools.build.xcode.xcodegen.proto.XcodeGenProtos.XcodeprojBuildSetting;

import java.util.Arrays;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Provider which provides transitive dependency information that is specific to Xcodegen. In
 * particular, it provides a sequence of targets which can be used to create a self-contained
 * {@code .xcodeproj} file.
 */
@Immutable
public final class XcodeProvider implements TransitiveInfoProvider {
  private static final String COMPANION_LIB_TARGET_LABEL_SUFFIX = "_static_lib";

  /**
   * A builder for instances of {@link XcodeProvider}.
   */
  public static final class Builder {
    private Label label;
    private final NestedSetBuilder<String> propagatedUserHeaderSearchPaths =
        NestedSetBuilder.linkOrder();
    private final NestedSetBuilder<String> nonPropagatedUserHeaderSearchPaths =
        NestedSetBuilder.linkOrder();
    private final NestedSetBuilder<String> propagatedHeaderSearchPaths =
        NestedSetBuilder.linkOrder();
    private final NestedSetBuilder<String> nonPropagatedHeaderSearchPaths =
        NestedSetBuilder.linkOrder();
    private Optional<Artifact> bundleInfoplist = Optional.absent();
    // Dependencies must be in link order because XCode observes the dependency ordering for
    // binary linking.
    private final NestedSetBuilder<XcodeProvider> propagatedDependencies =
        NestedSetBuilder.linkOrder();
    private final NestedSetBuilder<XcodeProvider> nonPropagatedDependencies =
        NestedSetBuilder.linkOrder();
    private final ImmutableList.Builder<XcodeprojBuildSetting> xcodeprojBuildSettings =
        new ImmutableList.Builder<>();
    private final ImmutableList.Builder<XcodeprojBuildSetting>
        companionTargetXcodeprojBuildSettings = new ImmutableList.Builder<>();
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
    private boolean generateCompanionLibTarget = false;
    private ConfigurationDistinguisher configurationDistinguisher;

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
      this.propagatedUserHeaderSearchPaths
          .addAll(rootEach("$(WORKSPACE_ROOT)", userHeaderSearchPaths));
      return this;
    }

    /**
     * Adds header search paths for this target. Each path is interpreted relative to the given
     * root, such as {@code "$(WORKSPACE_ROOT)"}.
     */
    public Builder addHeaderSearchPaths(String root, Iterable<PathFragment> paths) {
      this.propagatedHeaderSearchPaths.addAll(rootEach(root, paths));
      return this;
    }

    /**
     * Adds non-propagated header search paths for this target. Each relative path is interpreted
     * relative to the given root, such as {@code "$(WORKSPACE_ROOT)"}.
     */
    public Builder addNonPropagatedHeaderSearchPaths(String root, Iterable<PathFragment> paths) {
      this.nonPropagatedHeaderSearchPaths.addAll(rootEach(root, paths));
      return this;
    }

    /**
     * Sets the Info.plist for the bundle represented by this provider.
     */
    public Builder setBundleInfoplist(Artifact bundleInfoplist) {
      this.bundleInfoplist = Optional.of(bundleInfoplist);
      return this;
    }

    /**
     * Adds items in the {@link NestedSet}s of the given target to the corresponding sets in this
     * builder. This is useful if the given target is a dependency or like a dependency
     * (e.g. a test host). The given provider is not registered as a dependency with this provider.
     */
    private void addTransitiveSets(XcodeProvider dependencyish) {
      additionalSources.addTransitive(dependencyish.additionalSources);
      inputsToXcodegen.addTransitive(dependencyish.inputsToXcodegen);
      propagatedUserHeaderSearchPaths.addTransitive(dependencyish.propagatedUserHeaderSearchPaths);
      propagatedHeaderSearchPaths.addTransitive(dependencyish.propagatedHeaderSearchPaths);
    }

   /**
     * Adds {@link XcodeProvider}s corresponding to direct dependencies of this target which should
     * be added in the {@code .xcodeproj} file and propagated up the dependency chain.
     */
    public Builder addPropagatedDependencies(Iterable<XcodeProvider> dependencies) {
      return addDependencies(dependencies, /*doPropagate=*/true);
    }

   /**
     * Adds {@link XcodeProvider}s corresponding to direct dependencies of this target which should
     * be added in the {@code .xcodeproj} file and not propagated up the dependency chain.
     */
    public Builder addNonPropagatedDependencies(Iterable<XcodeProvider> dependencies) {
      return addDependencies(dependencies, /*doPropagate=*/false);
    }

    private Builder addDependencies(Iterable<XcodeProvider> dependencies, boolean doPropagate) {
      for (XcodeProvider dependency : dependencies) {
        // TODO(bazel-team): This is messy. Maybe we should make XcodeProvider be able to specify
        // how to depend on it rather than require this method to choose based on the dependency's
        // type.
        if (dependency.productType == XcodeProductType.EXTENSION) {
          this.extensions.add(dependency);
          this.inputsToXcodegen.addTransitive(dependency.inputsToXcodegen);
          this.additionalSources.addTransitive(dependency.additionalSources);
        } else {
          if (doPropagate) {
            this.propagatedDependencies.add(dependency);
            this.propagatedDependencies.addTransitive(dependency.propagatedDependencies);
            this.addTransitiveSets(dependency);
          } else {
            this.nonPropagatedDependencies.add(dependency);
            this.nonPropagatedDependencies.addTransitive(dependency.propagatedDependencies);
            this.nonPropagatedUserHeaderSearchPaths
                .addTransitive(dependency.propagatedUserHeaderSearchPaths);
            this.nonPropagatedHeaderSearchPaths
                .addTransitive(dependency.propagatedHeaderSearchPaths);
            this.inputsToXcodegen.addTransitive(dependency.inputsToXcodegen);
          }
        }
      }
      return this;
    }

    /**
     * Adds additional build settings of this target and its companion library target, if it exists.
     */
    public Builder addXcodeprojBuildSettings(
        Iterable<XcodeprojBuildSetting> xcodeprojBuildSettings) {
      this.xcodeprojBuildSettings.addAll(xcodeprojBuildSettings);
      this.companionTargetXcodeprojBuildSettings.addAll(xcodeprojBuildSettings);
      return this;
    }

    /**
     * Adds additional build settings of this target without adding them to the companion lib
     * target, if it exists.
     */
    public Builder addMainTargetXcodeprojBuildSettings(
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

    /**
     * Generates an extra LIBRARY_STATIC Xcode target with the same compilation artifacts. Dependent
     * Xcode targets will pick this companion library target as its dependency, rather than the
     * main Xcode target of this provider.
     */
    // TODO(bazel-team): Remove this when the binary rule types and bundling rule types are merged.
    public Builder generateCompanionLibTarget() {
      this.generateCompanionLibTarget = true;
      return this;
    }

    /**
     * Sets the distinguisher that will cause this xcode provider to discard any dependencies from
     * sources that are tagged with a different distinguisher.
     */
    public Builder setConfigurationDistinguisher(ConfigurationDistinguisher distinguisher) {
      this.configurationDistinguisher = distinguisher;
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
    private final NestedSet<Artifact> additionalSources;
    private final ImmutableList<XcodeProvider> topLevelTargets;

    private Project(
        NestedSet<Artifact> inputsToXcodegen, NestedSet<Artifact> additionalSources,
        ImmutableList<XcodeProvider> topLevelTargets) {
      this.inputsToXcodegen = inputsToXcodegen;
      this.additionalSources = additionalSources;
      this.topLevelTargets = topLevelTargets;
    }

    public static Project fromTopLevelTarget(XcodeProvider topLevelTarget) {
      return fromTopLevelTargets(ImmutableList.of(topLevelTarget));
    }

    public static Project fromTopLevelTargets(Iterable<XcodeProvider> topLevelTargets) {
      NestedSetBuilder<Artifact> inputsToXcodegen = NestedSetBuilder.stableOrder();
      NestedSetBuilder<Artifact> additionalSources = NestedSetBuilder.stableOrder();
      for (XcodeProvider target : topLevelTargets) {
        inputsToXcodegen.addTransitive(target.inputsToXcodegen);
        additionalSources.addTransitive(target.additionalSources);
      }
      return new Project(inputsToXcodegen.build(), additionalSources.build(),
          ImmutableList.copyOf(topLevelTargets));
    }

    /**
     * Returns artifacts that are passed to the Xcodegen action when generating a project file that
     * contains all of the given targets.
     */
    public NestedSet<Artifact> getInputsToXcodegen() {
      return inputsToXcodegen;
    }
    
    /**
     * Returns artifacts that are additional sources for the Xcodegen action.
     */
    public NestedSet<Artifact> getAdditionalSources() {
      return additionalSources;
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
      Map<Label, XcodeProvider> labelToProvider = new HashMap<>();
      for (XcodeProvider provider : providerSet) {
        XcodeProvider oldProvider = labelToProvider.put(provider.label, provider);
        if (oldProvider != null) {
          if (!oldProvider.architecture.equals(provider.architecture)
              || oldProvider.configurationDistinguisher != provider.configurationDistinguisher) {
            // Do not include duplicate dependencies whose architecture or configuration
            // distinguisher does not match this project's. This check avoids having multiple
            // conflicting Xcode targets for the same BUILD target that are only distinguished by
            // these fields (which Xcode does not care about).
            continue;
          }

          throw new IllegalStateException("Depending on multiple versions of the same xcode target "
              + "is not allowed but occurred for: " + provider.label);
        }
        controls.addAll(provider.targetControls());
      }
      return controls.build();
    }
  }

  private final Label label;
  private final NestedSet<String> propagatedUserHeaderSearchPaths;
  private final NestedSet<String> nonPropagatedUserHeaderSearchPaths;
  private final NestedSet<String> propagatedHeaderSearchPaths;
  private final NestedSet<String> nonPropagatedHeaderSearchPaths;
  private final Optional<Artifact> bundleInfoplist;
  private final NestedSet<XcodeProvider> propagatedDependencies;
  private final NestedSet<XcodeProvider> nonPropagatedDependencies;
  private final ImmutableList<XcodeprojBuildSetting> xcodeprojBuildSettings;
  private final ImmutableList<XcodeprojBuildSetting> companionTargetXcodeprojBuildSettings;
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
  private final boolean generateCompanionLibTarget;
  private final ConfigurationDistinguisher configurationDistinguisher;

  private XcodeProvider(Builder builder) {
    this.label = Preconditions.checkNotNull(builder.label);
    this.propagatedUserHeaderSearchPaths = builder.propagatedUserHeaderSearchPaths.build();
    this.nonPropagatedUserHeaderSearchPaths = builder.nonPropagatedUserHeaderSearchPaths.build();
    this.propagatedHeaderSearchPaths = builder.propagatedHeaderSearchPaths.build();
    this.nonPropagatedHeaderSearchPaths = builder.nonPropagatedHeaderSearchPaths.build();
    this.bundleInfoplist = builder.bundleInfoplist;
    this.propagatedDependencies = builder.propagatedDependencies.build();
    this.nonPropagatedDependencies = builder.nonPropagatedDependencies.build();
    this.xcodeprojBuildSettings = builder.xcodeprojBuildSettings.build();
    this.companionTargetXcodeprojBuildSettings =
        builder.companionTargetXcodeprojBuildSettings.build();
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
    this.generateCompanionLibTarget = builder.generateCompanionLibTarget;
    this.configurationDistinguisher =
        Preconditions.checkNotNull(builder.configurationDistinguisher);
  }

  private void collectProviders(Set<XcodeProvider> allProviders) {
    if (allProviders.add(this)) {
      for (XcodeProvider dependency : Iterables.concat(propagatedDependencies,
          nonPropagatedDependencies)) {
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

  @VisibleForTesting
  static final EnumSet<XcodeProductType> CAN_LINK_PRODUCT_TYPES = EnumSet.of(
      XcodeProductType.APPLICATION, XcodeProductType.BUNDLE, XcodeProductType.UNIT_TEST,
      XcodeProductType.EXTENSION, XcodeProductType.FRAMEWORK);

  /**
   * Returns the name of the Xcode target that corresponds to a build target with the given name.
   * This changes the label to make it a legal Xcode target name (which means removing slashes and
   * the colon). It also makes the label more readable in the Xcode UI by putting the target name
   * first and the package elements in reverse. This means the "important" part is visible even if
   * the project navigator is too narrow to show the entire name.
   */
  static String xcodeTargetName(Label label) {
    return xcodeTargetName(label, /*labelSuffix=*/"");
  }

  /**
   * Returns the name of the companion Xcode library target that corresponds to a build target with
   * the given name. See {@link XcodeSupport#generateCompanionLibXcodeTarget} for the rationale of
   * the companion library target and {@link #xcodeTargetName(Label)} for naming details.
   */
  static String xcodeCompanionLibTargetName(Label label) {
    return xcodeTargetName(label, COMPANION_LIB_TARGET_LABEL_SUFFIX);
  }

  private static String xcodeTargetName(Label label, String labelSuffix) {
    String pathFromWorkspaceRoot = label + labelSuffix;
    if (label.getPackageIdentifier().getRepository().isDefault()) {
      pathFromWorkspaceRoot = pathFromWorkspaceRoot.replace("//", "")
          .replace(':', '/');
    } else {
      pathFromWorkspaceRoot = pathFromWorkspaceRoot.replace("//", "_")
          .replace(':', '/').replace("@", "external_");
    }
    List<String> components = Splitter.on('/').splitToList(pathFromWorkspaceRoot);
    return Joiner.on('_').join(Lists.reverse(components));
  }

  /**
   * Returns the name of the xcode target in this provider to be referenced as a dep for dependents.
   */
  private String dependencyXcodeTargetName() {
    return generateCompanionLibTarget ? xcodeCompanionLibTargetName(label) : xcodeTargetName(label);
  }

  private Iterable<TargetControl> targetControls() {
    TargetControl mainTargetControl = targetControl();
    if (generateCompanionLibTarget) {
      return ImmutableList.of(mainTargetControl, companionLibTargetControl(mainTargetControl));
    } else {
      return ImmutableList.of(mainTargetControl);
    }
  }

  private TargetControl targetControl() {
    String buildFilePath = label.getPackageFragment().getSafePathString() + "/BUILD";
    NestedSet<String> userHeaderSearchPaths = NestedSetBuilder.<String>linkOrder()
        .addTransitive(propagatedUserHeaderSearchPaths)
        .addTransitive(nonPropagatedUserHeaderSearchPaths)
        .build();
    NestedSet<String> headerSearchPaths = NestedSetBuilder.<String>linkOrder()
        .addTransitive(propagatedHeaderSearchPaths)
        .addTransitive(nonPropagatedHeaderSearchPaths)
        .build();

    // TODO(bazel-team): Add provisioning profile information when Xcodegen supports it.
    TargetControl.Builder targetControl =
        TargetControl.newBuilder()
            .setName(label.getName())
            .setLabel(xcodeTargetName(label))
            .setProductType(productType.getIdentifier())
            .addSupportFile(buildFilePath)
            .addAllImportedLibrary(Artifact.toExecPaths(objcProvider.get(IMPORTED_LIBRARY)))
            .addAllUserHeaderSearchPath(userHeaderSearchPaths)
            .addAllHeaderSearchPath(headerSearchPaths)
            .addAllSupportFile(Artifact.toExecPaths(headers))
            .addAllCopt(compilationModeCopts)
            .addAllCopt(CompilationSupport.DEFAULT_COMPILER_FLAGS)
            .addAllCopt(Interspersing.prependEach("-D", objcProvider.get(DEFINE)))
            .addAllCopt(copts)
            .addAllLinkopt(
                Interspersing.beforeEach("-force_load", objcProvider.get(FORCE_LOAD_FOR_XCODEGEN)))
            .addAllLinkopt(CompilationSupport.DEFAULT_LINKER_FLAGS)
            .addAllLinkopt(
                Interspersing.beforeEach(
                    "-weak_framework", SdkFramework.names(objcProvider.get(WEAK_SDK_FRAMEWORK))))
            .addAllBuildSetting(xcodeprojBuildSettings)
            .addAllBuildSetting(AppleToolchain.defaultWarningsForXcode())
            .addAllSdkFramework(SdkFramework.names(objcProvider.get(SDK_FRAMEWORK)))
            .addAllFramework(PathFragment.safePathStrings(objcProvider.get(FRAMEWORK_DIR)))
            .addAllXcassetsDir(PathFragment.safePathStrings(objcProvider.get(XCASSETS_DIR)))
            .addAllXcdatamodel(PathFragment.safePathStrings(
                Xcdatamodels.datamodelDirs(objcProvider.get(XCDATAMODEL))))
            .addAllBundleImport(PathFragment.safePathStrings(objcProvider.get(BUNDLE_IMPORT_DIR)))
            .addAllSdkDylib(objcProvider.get(SDK_DYLIB))
            .addAllGeneralResourceFile(
                Artifact.toExecPaths(objcProvider.get(GENERAL_RESOURCE_FILE)))
            .addAllGeneralResourceFile(
                PathFragment.safePathStrings(objcProvider.get(GENERAL_RESOURCE_DIR)));

    if (CAN_LINK_PRODUCT_TYPES.contains(productType)) {
      // For builds with --ios_multi_cpus set, we may have several copies of some XCodeProviders
      // in the dependencies (one per cpu architecture). We deduplicate the corresponding
      // xcode target names with a LinkedHashSet before adding to the TargetControl.
      Set<DependencyControl> dependencySet = new LinkedHashSet<>();
      for (XcodeProvider dependency : propagatedDependencies) {
        // Only add a library target to a binary's dependencies if it has source files to compile
        // and it is not from the "non_propagated_deps" attribute. Xcode cannot build targets
        // without a source file in the PBXSourceFilesBuildPhase, so if such a target is present in
        // the control file, it is only to get Xcodegen to put headers and resources not used by the
        // final binary in the Project Navigator.
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
          String dependencyXcodeTargetName = dependency.dependencyXcodeTargetName();
          dependencySet.add(DependencyControl.newBuilder()
                .setTargetLabel(dependencyXcodeTargetName)
                .build());
        }
      }

      for (DependencyControl dependencyControl : dependencySet) {
        targetControl.addDependency(dependencyControl);
      }
    }
    for (XcodeProvider justTestHost : testHost.asSet()) {
      targetControl.addDependency(DependencyControl.newBuilder()
          .setTargetLabel(xcodeTargetName(justTestHost.label))
          .setTestHost(true)
          .build());
    }
    for (XcodeProvider extension : extensions) {
      targetControl.addDependency(DependencyControl.newBuilder()
          .setTargetLabel(xcodeTargetName(extension.label))
          .build());
    }

    if (bundleInfoplist.isPresent()) {
      targetControl.setInfoplist(bundleInfoplist.get().getExecPathString());
    }
    for (CompilationArtifacts artifacts : compilationArtifacts.asSet()) {
      targetControl
          .addAllSourceFile(Artifact.toExecPaths(artifacts.getSrcs()))
          .addAllSupportFile(Artifact.toExecPaths(artifacts.getAdditionalHdrs()))
          .addAllSupportFile(Artifact.toExecPaths(artifacts.getPrivateHdrs()))
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

  private TargetControl companionLibTargetControl(TargetControl mainTargetControl) {
    return TargetControl.newBuilder()
        .mergeFrom(mainTargetControl)
        .setName(label.getName() + COMPANION_LIB_TARGET_LABEL_SUFFIX)
        .setLabel(xcodeCompanionLibTargetName(label))
        .setProductType(LIBRARY_STATIC.getIdentifier())
        .clearInfoplist()
        .clearDependency()
        .clearBuildSetting()
        .addAllBuildSetting(companionTargetXcodeprojBuildSettings)
        .addAllBuildSetting(AppleToolchain.defaultWarningsForXcode())
        .build();
  }

  /**
   * Prepends the given path to each path in {@code paths}. Empty paths are
   * transformed to the value of {@code variable} rather than {@code variable + "/."}. Absolute
   * paths are returned without modifications.
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
        } else if (input.isAbsolute()) {
          return input.getSafePathString();
        } else {
          return prefix + "/" + input.getSafePathString();
        }
      }
    });
  }
}
