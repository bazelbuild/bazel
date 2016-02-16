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

import static com.google.devtools.build.lib.rules.objc.ObjcProvider.ASSET_CATALOG;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.BREAKPAD_FILE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.BUNDLE_FILE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.BUNDLE_IMPORT_DIR;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.CC_LIBRARY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.DEFINE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.FLAG;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.FORCE_LOAD_FOR_XCODEGEN;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.FORCE_LOAD_LIBRARY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.FRAMEWORK_DIR;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.FRAMEWORK_FILE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.Flag.USES_CPP;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.Flag.USES_SWIFT;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.GENERAL_RESOURCE_DIR;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.GENERAL_RESOURCE_FILE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.HEADER;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.IMPORTED_LIBRARY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.INCLUDE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.INCLUDE_SYSTEM;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.J2OBJC_LIBRARY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.LIBRARY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.LINKED_BINARY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.LINKOPT;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.MODULE_MAP;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.SDK_DYLIB;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.SDK_FRAMEWORK;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.SOURCE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.STORYBOARD;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.STRINGS;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.TOP_LEVEL_MODULE_MAP;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.WEAK_SDK_FRAMEWORK;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.XCASSETS_DIR;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.XCDATAMODEL;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.XIB;
import static com.google.devtools.build.lib.vfs.PathFragment.TO_PATH_FRAGMENT;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Optional;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.UnmodifiableIterator;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.rules.apple.AppleToolchain;
import com.google.devtools.build.lib.rules.cpp.CcCommon;
import com.google.devtools.build.lib.rules.cpp.CcLinkParams;
import com.google.devtools.build.lib.rules.cpp.CcLinkParamsProvider;
import com.google.devtools.build.lib.rules.cpp.CppCompilationContext;
import com.google.devtools.build.lib.rules.cpp.CppModuleMap;
import com.google.devtools.build.lib.rules.cpp.CppRunfilesProvider;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Contains information common to multiple objc_* rules, and provides a unified API for extracting
 * and accessing it.
 */
// TODO(bazel-team): Decompose and subsume area-specific logic and data into the various *Support
// classes. Make sure to distinguish rule output (providers, runfiles, ...) from intermediate,
// rule-internal information. Any provider created by a rule should not be read, only published.
public final class ObjcCommon {
  /**
   * Provides a way to access attributes that are common to all compilation rules.
   */
  // TODO(bazel-team): Delete and move into support-specific attributes classes once ObjcCommon is
  // gone.
  static final class CompilationAttributes {
    private final RuleContext ruleContext;
    private final ObjcSdkFrameworks.Attributes sdkFrameworkAttributes;

    CompilationAttributes(RuleContext ruleContext) {
      this.ruleContext = Preconditions.checkNotNull(ruleContext);
      this.sdkFrameworkAttributes = new ObjcSdkFrameworks.Attributes(ruleContext);
    }

    ImmutableList<Artifact> hdrs() {
      // Some rules may compile but not have the "hdrs" attribute.
      if (!ruleContext.attributes().has("hdrs", BuildType.LABEL_LIST)) {
        return ImmutableList.of();
      }
      ImmutableList.Builder<Artifact> headers = ImmutableList.builder();
      for (Pair<Artifact, Label> header : CcCommon.getHeaders(ruleContext)) {
        headers.add(header.first);
      }
      return headers.build();
    }

    /**
     * Returns headers that cannot be compiled individually.
     */
    ImmutableList<Artifact> textualHdrs() {
      // Some rules may compile but not have the "textual_hdrs" attribute.
      if (!ruleContext.attributes().has("textual_hdrs", BuildType.LABEL_LIST)) {
        return ImmutableList.of();
      }
      return ruleContext.getPrerequisiteArtifacts("textual_hdrs", Mode.TARGET).list();
    }

    Optional<Artifact> bridgingHeader() {
      Artifact header = ruleContext.getPrerequisiteArtifact("bridging_header", Mode.TARGET);
      return Optional.fromNullable(header);
    }

    Iterable<PathFragment> includes() {
      return Iterables.transform(
          ruleContext.attributes().get("includes", Type.STRING_LIST),
          PathFragment.TO_PATH_FRAGMENT);
    }

    Iterable<PathFragment> sdkIncludes() {
      return Iterables.transform(
          ruleContext.attributes().get("sdk_includes", Type.STRING_LIST),
          PathFragment.TO_PATH_FRAGMENT);
    }

    /**
     * Returns the value of the sdk_frameworks attribute plus frameworks that are included
     * automatically.
     */
    ImmutableSet<SdkFramework> sdkFrameworks() {
      return sdkFrameworkAttributes.sdkFrameworks();
    }

    /**
     * Returns the value of the weak_sdk_frameworks attribute.
     */
    ImmutableSet<SdkFramework> weakSdkFrameworks() {
      return sdkFrameworkAttributes.weakSdkFrameworks();
    }

    /**
     * Returns the value of the sdk_dylibs attribute.
     */
    ImmutableSet<String> sdkDylibs() {
      return sdkFrameworkAttributes.sdkDylibs();
    }

    /**
     * Returns the exec paths of all header search paths that should be added to this target and
     * dependers on this target, obtained from the {@code includes} attribute.
     */
    ImmutableList<PathFragment> headerSearchPaths() {
      ImmutableList.Builder<PathFragment> paths = new ImmutableList.Builder<>();
      PathFragment packageFragment =
          ruleContext.getLabel().getPackageIdentifier().getPathFragment();
      List<PathFragment> rootFragments = ImmutableList.of(
          packageFragment,
          ruleContext.getConfiguration().getGenfilesFragment().getRelative(packageFragment));

      Iterable<PathFragment> relativeIncludes =
          Iterables.filter(includes(), Predicates.not(PathFragment.IS_ABSOLUTE));
      for (PathFragment include : relativeIncludes) {
        for (PathFragment rootFragment : rootFragments) {
          paths.add(rootFragment.getRelative(include).normalize());
        }
      }
      return paths.build();
    }

    /**
     * Returns any values specified in this rule's {@code copts} attribute or an empty list if the
     * attribute does not exist or no values are specified.
     */
    public Iterable<String> copts() {
      if (!ruleContext.attributes().has("copts", Type.STRING_LIST)) {
        return ImmutableList.of();
      }
      return ruleContext.getTokenizedStringListAttr("copts");
    }

    /**
     * Returns any {@code copts} defined on an {@code objc_options} rule that is a dependency of
     * this rule.
     */
    public Iterable<String> optionsCopts() {
      if (!ruleContext.attributes().has("options", BuildType.LABEL)) {
        return ImmutableList.of();
      }
      OptionsProvider optionsProvider =
          ruleContext.getPrerequisite("options", Mode.TARGET, OptionsProvider.class);
      if (optionsProvider == null) {
        return ImmutableList.of();
      }
      return optionsProvider.getCopts();
    }

    /**
     * The clang module maps of direct dependencies of this rule. These are needed to generate
     * this rule's module map.
     */
    public List<CppModuleMap> moduleMapsForDirectDeps() {
      // Make sure all dependencies that have headers are included here. If a module map is missing,
      // its private headers will be treated as public!
      ArrayList<CppModuleMap> moduleMaps = new ArrayList<>();
      collectModuleMapsFromAttributeIfExists(moduleMaps, "deps");
      collectModuleMapsFromAttributeIfExists(moduleMaps, "non_propagated_deps");
      return moduleMaps;
    }

    /**
     * Collects all module maps from the targets in a certain attribute and adds them into
     * {@code moduleMaps}.
     *
     * @param moduleMaps an {@link ArrayList} to collect the module maps into
     * @param attribute the name of a label list attribute to collect module maps from
     */
    private void collectModuleMapsFromAttributeIfExists(
        ArrayList<CppModuleMap> moduleMaps, String attribute) {
      if (ruleContext.attributes().has(attribute, BuildType.LABEL_LIST)) {
        Iterable<ObjcProvider> providers =
            ruleContext.getPrerequisites(attribute, Mode.TARGET, ObjcProvider.class);
        for (ObjcProvider provider : providers) {
          moduleMaps.addAll(provider.get(TOP_LEVEL_MODULE_MAP).toCollection());
        }
      }
    }

    /**
     * Returns whether this target uses language features that require clang modules, such as
     * {@literal @}import.
     */
    public boolean enableModules() {
      return ruleContext.attributes().has("enable_modules", Type.BOOLEAN)
          && ruleContext.attributes().get("enable_modules", Type.BOOLEAN);
    }
  }

  /**
   * Provides a way to access attributes that are common to all resources rules.
   */
  // TODO(bazel-team): Delete and move into support-specific attributes classes once ObjcCommon is
  // gone.
  static final class ResourceAttributes {
    private final RuleContext ruleContext;

    ResourceAttributes(RuleContext ruleContext) {
      this.ruleContext = ruleContext;
    }

    ImmutableList<Artifact> strings() {
      return ruleContext.getPrerequisiteArtifacts("strings", Mode.TARGET).list();
    }

    ImmutableList<Artifact> xibs() {
      return ruleContext.getPrerequisiteArtifacts("xibs", Mode.TARGET).list();
    }

    ImmutableList<Artifact> storyboards() {
      return ruleContext.getPrerequisiteArtifacts("storyboards", Mode.TARGET).list();
    }

    ImmutableList<Artifact> resources() {
      return ruleContext.getPrerequisiteArtifacts("resources", Mode.TARGET).list();
    }

    ImmutableList<Artifact> structuredResources() {
      return ruleContext.getPrerequisiteArtifacts("structured_resources", Mode.TARGET).list();
    }

    ImmutableList<Artifact> datamodels() {
      return ruleContext.getPrerequisiteArtifacts("datamodels", Mode.TARGET).list();
    }

    ImmutableList<Artifact> assetCatalogs() {
      return ruleContext.getPrerequisiteArtifacts("asset_catalogs", Mode.TARGET).list();
    }
  }

  static class Builder {
    private RuleContext context;
    private Optional<CompilationAttributes> compilationAttributes = Optional.absent();
    private Optional<ResourceAttributes> resourceAttributes = Optional.absent();
    private Iterable<SdkFramework> extraSdkFrameworks = ImmutableList.of();
    private Iterable<SdkFramework> extraWeakSdkFrameworks = ImmutableList.of();
    private Iterable<String> extraSdkDylibs = ImmutableList.of();
    private Iterable<Artifact> frameworkImports = ImmutableList.of();
    private Optional<CompilationArtifacts> compilationArtifacts = Optional.absent();
    private Iterable<ObjcProvider> depObjcProviders = ImmutableList.of();
    private Iterable<ObjcProvider> directDepObjcProviders = ImmutableList.of();
    private Iterable<String> defines = ImmutableList.of();
    private Iterable<PathFragment> userHeaderSearchPaths = ImmutableList.of();
    private IntermediateArtifacts intermediateArtifacts;
    private boolean alwayslink;
    private boolean hasModuleMap;
    private Iterable<Artifact> extraImportLibraries = ImmutableList.of();
    private Optional<Artifact> linkedBinary = Optional.absent();
    private Optional<Artifact> breakpadFile = Optional.absent();
    private Iterable<CppCompilationContext> depCcHeaderProviders = ImmutableList.of();
    private Iterable<CcLinkParamsProvider> depCcLinkProviders = ImmutableList.of();

    Builder(RuleContext context) {
      this.context = Preconditions.checkNotNull(context);
    }

    public Builder setCompilationAttributes(CompilationAttributes baseCompilationAttributes) {
      Preconditions.checkState(!this.compilationAttributes.isPresent(),
          "compilationAttributes is already set to: %s", this.compilationAttributes);
      this.compilationAttributes = Optional.of(baseCompilationAttributes);
      return this;
    }

    public Builder setResourceAttributes(ResourceAttributes baseResourceAttributes) {
      Preconditions.checkState(!this.resourceAttributes.isPresent(),
          "resourceAttributes is already set to: %s", this.resourceAttributes);
      this.resourceAttributes = Optional.of(baseResourceAttributes);
      return this;
    }

    Builder addExtraSdkFrameworks(Iterable<SdkFramework> extraSdkFrameworks) {
      this.extraSdkFrameworks = Iterables.concat(this.extraSdkFrameworks, extraSdkFrameworks);
      return this;
    }

    Builder addExtraWeakSdkFrameworks(Iterable<SdkFramework> extraWeakSdkFrameworks) {
      this.extraWeakSdkFrameworks =
          Iterables.concat(this.extraWeakSdkFrameworks, extraWeakSdkFrameworks);
      return this;
    }

    Builder addExtraSdkDylibs(Iterable<String> extraSdkDylibs) {
      this.extraSdkDylibs = Iterables.concat(this.extraSdkDylibs, extraSdkDylibs);
      return this;
    }

    Builder addFrameworkImports(Iterable<Artifact> frameworkImports) {
      this.frameworkImports = Iterables.concat(this.frameworkImports, frameworkImports);
      return this;
    }

    Builder setCompilationArtifacts(CompilationArtifacts compilationArtifacts) {
      Preconditions.checkState(!this.compilationArtifacts.isPresent(),
          "compilationArtifacts is already set to: %s", this.compilationArtifacts);
      this.compilationArtifacts = Optional.of(compilationArtifacts);
      return this;
    }

    /**
     * Add providers which will be exposed both to the declaring rule and to any dependers on the
     * declaring rule.
     */
    Builder addDepObjcProviders(Iterable<ObjcProvider> depObjcProviders) {
      this.depObjcProviders = Iterables.concat(this.depObjcProviders, depObjcProviders);
      return this;
    }

    /**
     * Add providers which will only be used by the declaring rule, and won't be propagated to any
     * dependers on the declaring rule.
     */
    Builder addNonPropagatedDepObjcProviders(Iterable<ObjcProvider> directDepObjcProviders) {
      this.directDepObjcProviders = Iterables.concat(
          this.directDepObjcProviders, directDepObjcProviders);
      return this;
    }

    public Builder addUserHeaderSearchPaths(Iterable<PathFragment> userHeaderSearchPaths) {
      this.userHeaderSearchPaths =
          Iterables.concat(this.userHeaderSearchPaths, userHeaderSearchPaths);
      return this;
    }

    public Builder addDefines(Iterable<String> defines) {
      this.defines = Iterables.concat(this.defines, defines);
      return this;
    }

    Builder setIntermediateArtifacts(IntermediateArtifacts intermediateArtifacts) {
      this.intermediateArtifacts = intermediateArtifacts;
      return this;
    }

    Builder setAlwayslink(boolean alwayslink) {
      this.alwayslink = alwayslink;
      return this;
    }

    /**
     * Specifies that this target has a clang module map. This should be called if this target
     * compiles sources or exposes headers for other targets to use. Note that this does not add
     * the action to generate the module map. It simply indicates that it should be added to the
     * provider.
     */
    Builder setHasModuleMap() {
      this.hasModuleMap = true;
      return this;
    }

    /**
     * Adds additional static libraries to be linked into the final ObjC application bundle.
     */
    Builder addExtraImportLibraries(Iterable<Artifact> extraImportLibraries) {
      this.extraImportLibraries = Iterables.concat(this.extraImportLibraries, extraImportLibraries);
      return this;
    }

    /**
     * Sets a linked binary generated by this rule to be propagated to dependers.
     */
    Builder setLinkedBinary(Artifact linkedBinary) {
      this.linkedBinary = Optional.of(linkedBinary);
      return this;
    }

    /**
     * Sets a breakpad file (used by the breakpad crash reporting system) generated by this rule to
     * be propagated to dependers.
     */
    Builder setBreakpadFile(Artifact breakpadFile) {
      this.breakpadFile = Optional.of(breakpadFile);
      return this;
    }

    /**
     * Sets information from {@code cc_library} dependencies to be used during compilation.
     */
    public Builder addDepCcHeaderProviders(Iterable<CppCompilationContext> depCcHeaderProviders) {
      this.depCcHeaderProviders = Iterables.concat(this.depCcHeaderProviders, depCcHeaderProviders);
      return this;
    }

    /**
     * Sets information from {@code cc_library} dependencies to be used during linking.
     */
    public Builder addDepCcLinkProviders(RuleContext ruleContext) {
      for (TransitiveInfoCollection dep : ruleContext.getPrerequisites("deps", Mode.TARGET)) {
        // Hack to determine if dep is a cc target. Required so objc_library archives packed in
        // CcLinkParamsProvider do not get consumed as cc targets.
        if (dep.getProvider(CppRunfilesProvider.class) != null) {
          CcLinkParamsProvider ccLinkParamsProvider = dep.getProvider(CcLinkParamsProvider.class);
          this.depCcLinkProviders =
              Iterables.concat(
                  this.depCcLinkProviders,
                  ImmutableList.<CcLinkParamsProvider>of(ccLinkParamsProvider));
        }
      }
      return this;
    }

    ObjcCommon build() {
      Iterable<BundleableFile> bundleImports = BundleableFile.bundleImportsFromRule(context);

      ObjcProvider.Builder objcProvider = new ObjcProvider.Builder()
          .addAll(IMPORTED_LIBRARY, extraImportLibraries)
          .addAll(BUNDLE_FILE, bundleImports)
          .addAll(BUNDLE_IMPORT_DIR,
              uniqueContainers(BundleableFile.toArtifacts(bundleImports), BUNDLE_CONTAINER_TYPE))
          .addAll(SDK_FRAMEWORK, extraSdkFrameworks)
          .addAll(WEAK_SDK_FRAMEWORK, extraWeakSdkFrameworks)
          .addAll(SDK_DYLIB, extraSdkDylibs)
          .addAll(FRAMEWORK_FILE, frameworkImports)
          .addAll(FRAMEWORK_DIR, uniqueContainers(frameworkImports, FRAMEWORK_CONTAINER_TYPE))
          .addAll(INCLUDE, userHeaderSearchPaths)
          .addAll(DEFINE, defines)
          .addTransitiveAndPropagate(depObjcProviders)
          .addTransitiveWithoutPropagating(directDepObjcProviders);

      for (CppCompilationContext headerProvider : depCcHeaderProviders) {
        objcProvider.addTransitiveAndPropagate(HEADER, headerProvider.getDeclaredIncludeSrcs());
        objcProvider.addAll(INCLUDE, headerProvider.getIncludeDirs());
        // TODO(bazel-team): This pulls in stl via CppHelper.mergeToolchainDependentContext but
        // probably shouldn't.
        objcProvider.addAll(INCLUDE_SYSTEM, headerProvider.getSystemIncludeDirs());
        objcProvider.addAll(DEFINE, headerProvider.getDefines());
      }
      for (CcLinkParamsProvider linkProvider : depCcLinkProviders) {
        CcLinkParams params = linkProvider.getCcLinkParams(true, false);
        ImmutableList<String> linkOpts = params.flattenedLinkopts();
        ImmutableSet.Builder<SdkFramework> frameworkLinkOpts = new ImmutableSet.Builder<>();
        ImmutableList.Builder<String> nonFrameworkLinkOpts = new ImmutableList.Builder<>();
        // Add any framework flags as frameworks directly, rather than as linkopts.
        for (UnmodifiableIterator<String> iterator = linkOpts.iterator(); iterator.hasNext(); ) {
          String arg = iterator.next();
          if (arg.equals("-framework") && iterator.hasNext()) {
            String framework = iterator.next();
            frameworkLinkOpts.add(new SdkFramework(framework));
          } else {
            nonFrameworkLinkOpts.add(arg);
          }
        }

        objcProvider
            .addAll(SDK_FRAMEWORK, frameworkLinkOpts.build())
            .addAll(LINKOPT, nonFrameworkLinkOpts.build())
            .addTransitiveAndPropagate(CC_LIBRARY, params.getLibraries());
      }

      if (compilationAttributes.isPresent()) {
        CompilationAttributes attributes = compilationAttributes.get();
        Iterable<PathFragment> sdkIncludes = Iterables.transform(
            Interspersing.prependEach(
                AppleToolchain.sdkDir() + "/usr/include/",
                PathFragment.safePathStrings(attributes.sdkIncludes())),
            TO_PATH_FRAGMENT);
        objcProvider
            .addAll(HEADER, attributes.hdrs())
            .addAll(HEADER, attributes.textualHdrs())
            .addAll(INCLUDE, attributes.headerSearchPaths())
            .addAll(INCLUDE, sdkIncludes)
            .addAll(SDK_FRAMEWORK, attributes.sdkFrameworks())
            .addAll(WEAK_SDK_FRAMEWORK, attributes.weakSdkFrameworks())
            .addAll(SDK_DYLIB, attributes.sdkDylibs());
      }

      if (resourceAttributes.isPresent()) {
        ResourceAttributes attributes = resourceAttributes.get();
        objcProvider
            .addAll(GENERAL_RESOURCE_FILE, attributes.storyboards())
            .addAll(GENERAL_RESOURCE_FILE, attributes.resources())
            .addAll(GENERAL_RESOURCE_FILE, attributes.strings())
            .addAll(GENERAL_RESOURCE_FILE, attributes.xibs())
            .addAll(
                GENERAL_RESOURCE_DIR, xcodeStructuredResourceDirs(attributes.structuredResources()))
            .addAll(BUNDLE_FILE, BundleableFile.flattenedRawResourceFiles(attributes.resources()))
            .addAll(
                BUNDLE_FILE,
                BundleableFile.structuredRawResourceFiles(attributes.structuredResources()))
            .addAll(
                XCASSETS_DIR,
                uniqueContainers(attributes.assetCatalogs(), ASSET_CATALOG_CONTAINER_TYPE))
            .addAll(ASSET_CATALOG, attributes.assetCatalogs())
            .addAll(XCDATAMODEL, attributes.datamodels())
            .addAll(XIB, attributes.xibs())
            .addAll(STRINGS, attributes.strings())
            .addAll(STORYBOARD, attributes.storyboards());
      }

      if (ObjcRuleClasses.useLaunchStoryboard(context)) {
        Artifact launchStoryboard =
            context.getPrerequisiteArtifact("launch_storyboard", Mode.TARGET);
        objcProvider.add(GENERAL_RESOURCE_FILE, launchStoryboard);
        if (ObjcRuleClasses.STORYBOARD_TYPE.matches(launchStoryboard.getPath())) {
          objcProvider.add(STORYBOARD, launchStoryboard);
        } else {
          objcProvider.add(XIB, launchStoryboard);
        }
      }

      for (CompilationArtifacts artifacts : compilationArtifacts.asSet()) {
        Iterable<Artifact> allSources =
            Iterables.concat(artifacts.getSrcs(), artifacts.getNonArcSrcs());
        // TODO(bazel-team): Add private headers to the provider when we have module maps to enforce
        // them.
        objcProvider
            .addAll(HEADER, artifacts.getAdditionalHdrs())
            .addAll(LIBRARY, artifacts.getArchive().asSet())
            .addAll(SOURCE, allSources);

        if (artifacts.getArchive().isPresent()
            && J2ObjcLibrary.J2OBJC_SUPPORTED_RULES.contains(context.getRule().getRuleClass())) {
          objcProvider.addAll(J2OBJC_LIBRARY, artifacts.getArchive().asSet());
        }

        boolean usesCpp = false;
        boolean usesSwift = false;
        for (Artifact sourceFile :
            Iterables.concat(artifacts.getSrcs(), artifacts.getNonArcSrcs())) {
          usesCpp = usesCpp || ObjcRuleClasses.CPP_SOURCES.matches(sourceFile.getExecPath());
          usesSwift = usesSwift || ObjcRuleClasses.SWIFT_SOURCES.matches(sourceFile.getExecPath());
        }

        if (usesCpp) {
          objcProvider.add(FLAG, USES_CPP);
        }

        if (usesSwift) {
          objcProvider.add(FLAG, USES_SWIFT);
        }
      }

      if (alwayslink) {
        for (CompilationArtifacts artifacts : compilationArtifacts.asSet()) {
          for (Artifact archive : artifacts.getArchive().asSet()) {
            objcProvider.add(FORCE_LOAD_LIBRARY, archive);
            objcProvider.add(FORCE_LOAD_FOR_XCODEGEN, String.format(
                "$(BUILT_PRODUCTS_DIR)/lib%s.a",
                XcodeProvider.xcodeTargetName(context.getLabel())));
          }
        }
        for (Artifact archive : extraImportLibraries) {
          objcProvider.add(FORCE_LOAD_LIBRARY, archive);
          objcProvider.add(FORCE_LOAD_FOR_XCODEGEN,
              "$(WORKSPACE_ROOT)/" + archive.getExecPath().getSafePathString());
        }
      }

      if (hasModuleMap && ObjcRuleClasses.objcConfiguration(context).moduleMapsEnabled()) {
        CppModuleMap moduleMap = intermediateArtifacts.moduleMap();
        objcProvider.add(MODULE_MAP, moduleMap.getArtifact());
        objcProvider.add(TOP_LEVEL_MODULE_MAP, moduleMap);
      }

      objcProvider.addAll(LINKED_BINARY, linkedBinary.asSet())
          .addAll(BREAKPAD_FILE, breakpadFile.asSet());

      return new ObjcCommon(objcProvider.build(), compilationArtifacts);
    }

  }

  static final FileType BUNDLE_CONTAINER_TYPE = FileType.of(".bundle");

  static final FileType ASSET_CATALOG_CONTAINER_TYPE = FileType.of(".xcassets");

  public static final FileType FRAMEWORK_CONTAINER_TYPE = FileType.of(".framework");
  private final ObjcProvider objcProvider;

  private final Optional<CompilationArtifacts> compilationArtifacts;

  private ObjcCommon(
      ObjcProvider objcProvider,
      Optional<CompilationArtifacts> compilationArtifacts) {
    this.objcProvider = Preconditions.checkNotNull(objcProvider);
    this.compilationArtifacts = Preconditions.checkNotNull(compilationArtifacts);
  }

  public ObjcProvider getObjcProvider() {
    return objcProvider;
  }

  public Optional<CompilationArtifacts> getCompilationArtifacts() {
    return compilationArtifacts;
  }

  /**
   * Returns an {@link Optional} containing the compiled {@code .a} file, or
   * {@link Optional#absent()} if this object contains no {@link CompilationArtifacts} or the
   * compilation information has no sources.
   */
  public Optional<Artifact> getCompiledArchive() {
    if (compilationArtifacts.isPresent()) {
      return compilationArtifacts.get().getArchive();
    }
    return Optional.absent();
  }

  static ImmutableList<PathFragment> userHeaderSearchPaths(BuildConfiguration configuration) {
    return ImmutableList.of(
        new PathFragment("."),
        configuration.getGenfilesFragment());
  }

  /**
   * Returns the first directory in the sequence of parents of the exec path of the given artifact
   * that matches {@code type}. For instance, if {@code type} is FileType.of(".foo") and the exec
   * path of {@code artifact} is {@code a/b/c/bar.foo/d/e}, then the return value is
   * {@code a/b/c/bar.foo}.
   */
  static Optional<PathFragment> nearestContainerMatching(FileType type, Artifact artifact) {
    PathFragment container = artifact.getExecPath();
    do {
      if (type.matches(container)) {
        return Optional.of(container);
      }
      container = container.getParentDirectory();
    } while (container != null);
    return Optional.absent();
  }

  /**
   * Similar to {@link #nearestContainerMatching(FileType, Artifact)}, but tries matching several
   * file types in {@code types}, and returns a path for the first match in the sequence.
   */
  static Optional<PathFragment> nearestContainerMatching(
      Iterable<FileType> types, Artifact artifact) {
    for (FileType type : types) {
      for (PathFragment container : nearestContainerMatching(type, artifact).asSet()) {
        return Optional.of(container);
      }
    }
    return Optional.absent();
  }

  /**
   * Returns all directories matching {@code containerType} that contain the items in
   * {@code artifacts}. This function ignores artifacts that are not in any directory matching
   * {@code containerType}.
   */
  static Iterable<PathFragment> uniqueContainers(
      Iterable<Artifact> artifacts, FileType containerType) {
    ImmutableSet.Builder<PathFragment> containers = new ImmutableSet.Builder<>();
    for (Artifact artifact : artifacts) {
      containers.addAll(ObjcCommon.nearestContainerMatching(containerType, artifact).asSet());
    }
    return containers.build();
  }

  /**
   * Returns the Xcode structured resource directory paths.
   *
   * <p>For a checked-in source artifact "//a/b/res/sub_dir/d" included by objc rule "//a/b:c",
   * "a/b/res" will be returned. For a generated source artifact "res/sub_dir/d" owned by genrule
   * "//a/b:c", "bazel-out/.../genfiles/a/b/res" will be returned.
   *
   * <p>When XCode sees a included resource directory of "a/b/res", the entire directory structure
   * up to "res" will be copied into the app bundle.
   */
  private static Iterable<PathFragment> xcodeStructuredResourceDirs(Iterable<Artifact> artifacts) {
    ImmutableSet.Builder<PathFragment> containers = new ImmutableSet.Builder<>();
    for (Artifact artifact : artifacts) {
      PathFragment ownerRuleDirectory = artifact.getArtifactOwner().getLabel().getPackageFragment();
      String containerName =
          artifact.getRootRelativePath().relativeTo(ownerRuleDirectory).getSegment(0);
      PathFragment rootExecPath = artifact.getRoot().getExecPath();
      containers.add(rootExecPath.getRelative(ownerRuleDirectory.getRelative(containerName)));
    }

    return containers.build();
  }

  /**
   * Similar to {@link #nearestContainerMatching(FileType, Artifact)}, but returns the container
   * closest to the root that matches the given type.
   */
  static Optional<PathFragment> farthestContainerMatching(FileType type, Artifact artifact) {
    PathFragment container = artifact.getExecPath();
    Optional<PathFragment> lastMatch = Optional.absent();
    do {
      if (type.matches(container)) {
        lastMatch = Optional.of(container);
      }
      container = container.getParentDirectory();
    } while (container != null);
    return lastMatch;
  }

  static Iterable<String> notInContainerErrors(
      Iterable<Artifact> artifacts, FileType containerType) {
    return notInContainerErrors(artifacts, ImmutableList.of(containerType));
  }

  @VisibleForTesting
  static final String NOT_IN_CONTAINER_ERROR_FORMAT =
      "File '%s' is not in a directory of one of these type(s): %s";

  static Iterable<String> notInContainerErrors(
      Iterable<Artifact> artifacts, Iterable<FileType> containerTypes) {
    Set<String> errors = new HashSet<>();
    for (Artifact artifact : artifacts) {
      boolean inContainer = nearestContainerMatching(containerTypes, artifact).isPresent();
      if (!inContainer) {
        errors.add(String.format(NOT_IN_CONTAINER_ERROR_FORMAT,
            artifact.getExecPath(), Iterables.toString(containerTypes)));
      }
    }
    return errors;
  }

}
