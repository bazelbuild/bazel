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

import static com.google.devtools.build.lib.rules.objc.ObjcProvider.CC_LIBRARY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.FLAG;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.FORCE_LOAD_LIBRARY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.Flag.USES_CPP;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.HEADER;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.IMPORTED_LIBRARY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.J2OBJC_LIBRARY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.LIBRARY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.LINKED_BINARY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.LINKOPT;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.MODULE_MAP;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.SDK_DYLIB;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.SDK_FRAMEWORK;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.SOURCE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.TOP_LEVEL_MODULE_MAP;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.UMBRELLA_HEADER;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.WEAK_SDK_FRAMEWORK;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Streams;
import com.google.common.collect.UnmodifiableIterator;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.rules.apple.AppleToolchain;
import com.google.devtools.build.lib.rules.cpp.CcCompilationContext;
import com.google.devtools.build.lib.rules.cpp.CcInfo;
import com.google.devtools.build.lib.rules.cpp.CcLinkingContext;
import com.google.devtools.build.lib.rules.cpp.CppModuleMap;
import com.google.devtools.build.lib.rules.cpp.LibraryToLink;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import net.starlark.java.eval.StarlarkSemantics;

/**
 * Contains information common to multiple objc_* rules, and provides a unified API for extracting
 * and accessing it.
 */
// TODO(bazel-team): Decompose and subsume area-specific logic and data into the various *Support
// classes. Make sure to distinguish rule output (providers, runfiles, ...) from intermediate,
// rule-internal information. Any provider created by a rule should not be read, only published.
public final class ObjcCommon {

  /** Filters fileset artifacts out of a group of artifacts. */
  public static ImmutableList<Artifact> filterFileset(Iterable<Artifact> artifacts) {
    ImmutableList.Builder<Artifact> inputs = ImmutableList.<Artifact>builder();
    for (Artifact artifact : artifacts) {
      if (!artifact.isFileset()) {
        inputs.add(artifact);
      }
    }
    return inputs.build();
  }

  /**
   * Indicates the purpose the ObjcCommon is used for.
   *
   * <p>The purpose determines whether ObjcCommon.build() will build an ObjcProvider or an
   * ObjcProvider.Builder. In compile-and-link mode, ObjcCommon.build() will output an
   * ObjcProvider.Builder. The builder is expected to combine with the CcCompilationContext from a
   * compile action, to form a complete ObjcProvider. In link-only mode, ObjcCommon can (and does)
   * output the full ObjcProvider.
   */
  public enum Purpose {
    /** The ObjcCommon will be used for compile and link. */
    COMPILE_AND_LINK,
    /** The ObjcCommon will be used for linking only. */
    LINK_ONLY,
  }

  static class Builder {
    private final boolean compileInfoMigration;
    private final Purpose purpose;
    private final RuleContext context;
    private final StarlarkSemantics semantics;
    private final BuildConfiguration buildConfiguration;
    private Optional<CompilationAttributes> compilationAttributes = Optional.absent();
    private Optional<CompilationArtifacts> compilationArtifacts = Optional.absent();
    private Iterable<ObjcProvider> depObjcProviders = ImmutableList.of();
    private Iterable<ObjcProvider> runtimeDepObjcProviders = ImmutableList.of();
    private Iterable<PathFragment> includes = ImmutableList.of();
    private IntermediateArtifacts intermediateArtifacts;
    private boolean alwayslink;
    private boolean hasModuleMap;
    private Iterable<Artifact> extraImportLibraries = ImmutableList.of();
    private Optional<Artifact> linkedBinary = Optional.absent();
    private Iterable<CcCompilationContext> depCcHeaderProviders = ImmutableList.of();
    private Iterable<CcLinkingContext> depCcLinkProviders = ImmutableList.of();
    private Iterable<CcCompilationContext> depCcDirectProviders = ImmutableList.of();

    /**
     * Builder for {@link ObjcCommon} obtaining both attribute data and configuration data from the
     * given rule context.
     */
    Builder(Purpose purpose, RuleContext context) throws InterruptedException {
      this(purpose, context, context.getConfiguration());
    }

    /**
     * Builder for {@link ObjcCommon} obtaining attribute data from the rule context and
     * configuration data from the given configuration object for use in situations where a single
     * target's outputs are under multiple configurations.
     */
    Builder(Purpose purpose, RuleContext context, BuildConfiguration buildConfiguration)
        throws InterruptedException {
      this.purpose = purpose;
      this.context = Preconditions.checkNotNull(context);
      this.semantics = context.getAnalysisEnvironment().getStarlarkSemantics();
      this.buildConfiguration = Preconditions.checkNotNull(buildConfiguration);

      ObjcConfiguration objcConfiguration = buildConfiguration.getFragment(ObjcConfiguration.class);

      this.compileInfoMigration = objcConfiguration.compileInfoMigration();
    }

    public Builder setCompilationAttributes(CompilationAttributes baseCompilationAttributes) {
      Preconditions.checkState(
          !this.compilationAttributes.isPresent(),
          "compilationAttributes is already set to: %s",
          this.compilationAttributes);
      this.compilationAttributes = Optional.of(baseCompilationAttributes);
      return this;
    }

    Builder setCompilationArtifacts(CompilationArtifacts compilationArtifacts) {
      Preconditions.checkState(
          !this.compilationArtifacts.isPresent(),
          "compilationArtifacts is already set to: %s",
          this.compilationArtifacts);
      this.compilationArtifacts = Optional.of(compilationArtifacts);
      return this;
    }

    private static ImmutableList<CcCompilationContext> getCcCompilationContexts(
        Iterable<CcInfo> ccInfos) {
      return Streams.stream(ccInfos)
          .map(CcInfo::getCcCompilationContext)
          .collect(ImmutableList.toImmutableList());
    }

    Builder addDepCcHeaderProviders(Iterable<CcInfo> cppDeps) {
      this.depCcHeaderProviders =
          Iterables.concat(this.depCcHeaderProviders, getCcCompilationContexts(cppDeps));
      return this;
    }

    Builder addDepCcDirectProviders(Iterable<CcInfo> cppDeps) {
      this.depCcDirectProviders =
          Iterables.concat(this.depCcDirectProviders, getCcCompilationContexts(cppDeps));
      return this;
    }

    private Builder addDepsPreMigration(List<ConfiguredTargetAndData> deps) {
      ImmutableList.Builder<ObjcProvider> propagatedObjcDeps = ImmutableList.builder();
      ImmutableList.Builder<CcInfo> cppDeps = ImmutableList.builder();
      ImmutableList.Builder<CcLinkingContext> cppDepLinkParams = ImmutableList.builder();

      for (ConfiguredTargetAndData dep : deps) {
        ConfiguredTarget depCT = dep.getConfiguredTarget();
        // It is redundant to process both ObjcProvider and CcInfo; doing so causes direct header
        // field to include indirect headers from deps.
        if (depCT.get(ObjcProvider.STARLARK_CONSTRUCTOR) != null) {
          addAnyProviders(propagatedObjcDeps, depCT, ObjcProvider.STARLARK_CONSTRUCTOR);
        } else {
          addAnyProviders(cppDeps, depCT, CcInfo.PROVIDER);
          if (isCcLibrary(dep)) {
            cppDepLinkParams.add(depCT.get(CcInfo.PROVIDER).getCcLinkingContext());
          }
        }
      }
      addDepObjcProviders(propagatedObjcDeps.build());
      ImmutableList<CcInfo> ccInfos = cppDeps.build();
      addDepCcHeaderProviders(ccInfos);
      addDepCcDirectProviders(ccInfos);
      this.depCcLinkProviders = Iterables.concat(this.depCcLinkProviders, cppDepLinkParams.build());

      return this;
    }

    private Builder addDepsPostMigration(List<ConfiguredTargetAndData> deps) {
      ImmutableList.Builder<ObjcProvider> propagatedObjcDeps = ImmutableList.builder();
      ImmutableList.Builder<CcInfo> cppDeps = ImmutableList.builder();
      ImmutableList.Builder<CcInfo> directCppDeps = ImmutableList.builder();
      ImmutableList.Builder<CcLinkingContext> cppDepLinkParams = ImmutableList.builder();

      for (ConfiguredTargetAndData dep : deps) {
        ConfiguredTarget depCT = dep.getConfiguredTarget();
        if (depCT.get(ObjcProvider.STARLARK_CONSTRUCTOR) != null) {
          addAnyProviders(propagatedObjcDeps, depCT, ObjcProvider.STARLARK_CONSTRUCTOR);
        } else {
          // This is the way we inject cc_library attributes into direct fields.
          addAnyProviders(directCppDeps, depCT, CcInfo.PROVIDER);
        }
        addAnyProviders(cppDeps, depCT, CcInfo.PROVIDER);
        if (isCcLibrary(dep)) {
          cppDepLinkParams.add(depCT.get(CcInfo.PROVIDER).getCcLinkingContext());
        }
      }
      addDepObjcProviders(propagatedObjcDeps.build());
      addDepCcHeaderProviders(cppDeps.build());
      addDepCcDirectProviders(directCppDeps.build());
      this.depCcLinkProviders = Iterables.concat(this.depCcLinkProviders, cppDepLinkParams.build());

      return this;
    }

    Builder addDeps(List<ConfiguredTargetAndData> deps) {
      if (compileInfoMigration) {
        return addDepsPostMigration(deps);
      } else {
        return addDepsPreMigration(deps);
      }
    }

    private Builder addRuntimeDepsPreMigration(
        List<? extends TransitiveInfoCollection> runtimeDeps) {
      ImmutableList.Builder<ObjcProvider> propagatedDeps = ImmutableList.builder();

      for (TransitiveInfoCollection dep : runtimeDeps) {
        addAnyProviders(propagatedDeps, dep, ObjcProvider.STARLARK_CONSTRUCTOR);
      }
      this.runtimeDepObjcProviders =
          Iterables.concat(this.runtimeDepObjcProviders, propagatedDeps.build());
      return this;
    }

    private Builder addRuntimeDepsPostMigration(
        List<? extends TransitiveInfoCollection> runtimeDeps) {
      ImmutableList.Builder<ObjcProvider> propagatedObjcDeps = ImmutableList.builder();
      ImmutableList.Builder<CcInfo> cppDeps = ImmutableList.builder();

      for (TransitiveInfoCollection dep : runtimeDeps) {
        addAnyProviders(propagatedObjcDeps, dep, ObjcProvider.STARLARK_CONSTRUCTOR);
        addAnyProviders(cppDeps, dep, CcInfo.PROVIDER);
      }
      this.runtimeDepObjcProviders =
          Iterables.concat(this.runtimeDepObjcProviders, propagatedObjcDeps.build());
      addDepCcHeaderProviders(cppDeps.build());
      return this;
    }

    /**
     * Adds providers for runtime frameworks included in the final app bundle but not linked with
     * at build time.
     */
    Builder addRuntimeDeps(List<? extends TransitiveInfoCollection> runtimeDeps) {
      if (compileInfoMigration) {
        return addRuntimeDepsPostMigration(runtimeDeps);
      } else {
        return addRuntimeDepsPreMigration(runtimeDeps);
      }
    }

    private <T extends Info> ImmutableList.Builder<T> addAnyProviders(
        ImmutableList.Builder<T> listBuilder,
        TransitiveInfoCollection collection,
        BuiltinProvider<T> providerClass) {
      T provider = collection.get(providerClass);
      if (provider != null) {
        listBuilder.add(provider);
      }
      return listBuilder;
    }

    /**
     * Add providers which will be exposed both to the declaring rule and to any dependers on the
     * declaring rule.
     */
    Builder addDepObjcProviders(Iterable<ObjcProvider> depObjcProviders) {
      this.depObjcProviders = Iterables.concat(this.depObjcProviders, depObjcProviders);
      return this;
    }

    /** Adds includes to be passed into compile actions with {@code -I}. */
    public Builder addIncludes(NestedSet<PathFragment> includes) {
      // The includes are copied to a new list in the .build() method, so flattening here should be
      // benign.
      this.includes = Iterables.concat(this.includes, includes.toList());
      return this;
    }

    /** Adds includes to be passed into compile actions with {@code -I}. */
    public Builder addIncludes(Iterable<PathFragment> includes) {
      this.includes = Iterables.concat(this.includes, includes);
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

    ObjcCommon build() {

      ObjcCompilationContext.Builder objcCompilationContextBuilder =
          ObjcCompilationContext.builder(compileInfoMigration);

      ObjcProvider.Builder objcProvider = new ObjcProvider.NativeBuilder(semantics);

      objcProvider
          .addAll(IMPORTED_LIBRARY, extraImportLibraries)
          .addTransitiveAndPropagate(depObjcProviders);

      objcCompilationContextBuilder
          .addIncludes(includes)
          .addDepObjcProviders(depObjcProviders)
          .addDepObjcProviders(runtimeDepObjcProviders)
          // TODO(bazel-team): This pulls in stl via
          // CcCompilationHelper.getStlCcCompilationContext(), but probably shouldn't.
          .addDepCcCompilationContexts(depCcHeaderProviders);

      for (CcCompilationContext headerProvider : depCcDirectProviders) {
        objcProvider.addAllDirect(HEADER, headerProvider.getDeclaredIncludeSrcs().toList());
      }

      for (ObjcProvider provider : runtimeDepObjcProviders) {
        objcProvider.addTransitiveAndPropagate(ObjcProvider.MERGE_ZIP, provider);
      }

      for (CcLinkingContext linkProvider : depCcLinkProviders) {
        ImmutableList<String> linkOpts = linkProvider.getFlattenedUserLinkFlags();
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
            .addTransitiveAndPropagate(
                CC_LIBRARY,
                NestedSetBuilder.<LibraryToLink>linkOrder()
                    .addTransitive(linkProvider.getLibraries())
                    .build());
      }

      if (compilationAttributes.isPresent()) {
        CompilationAttributes attributes = compilationAttributes.get();
        PathFragment usrIncludeDir = PathFragment.create(AppleToolchain.sdkDir() + "/usr/include/");
        Iterable<PathFragment> sdkIncludes =
            Iterables.transform(
                attributes.sdkIncludes().toList(), (p) -> usrIncludeDir.getRelative(p));
        objcProvider
            .addAll(SDK_FRAMEWORK, attributes.sdkFrameworks())
            .addAll(WEAK_SDK_FRAMEWORK, attributes.weakSdkFrameworks())
            .addAll(SDK_DYLIB, attributes.sdkDylibs())
            .addAllDirect(HEADER, attributes.hdrs().toList())
            .addAllDirect(HEADER, attributes.textualHdrs().toList());
        objcCompilationContextBuilder
            .addPublicHeaders(filterFileset(attributes.hdrs().toList()))
            .addPublicTextualHeaders(filterFileset(attributes.textualHdrs().toList()))
            .addDefines(attributes.defines())
            .addIncludes(
                attributes
                    .headerSearchPaths(
                        buildConfiguration.getGenfilesFragment(context.getRepository()))
                    .toList())
            .addIncludes(sdkIncludes);
      }

      for (CompilationArtifacts artifacts : compilationArtifacts.asSet()) {
        Iterable<Artifact> allSources =
            Iterables.concat(
                artifacts.getSrcs(), artifacts.getNonArcSrcs(), artifacts.getPrivateHdrs());
        objcProvider
            .addAll(LIBRARY, artifacts.getArchive().asSet())
            .addAll(SOURCE, allSources)
            .addAllDirect(SOURCE, allSources)
            .addAllDirect(HEADER, filterFileset(artifacts.getAdditionalHdrs().toList()));
        objcCompilationContextBuilder.addPublicHeaders(
            filterFileset(artifacts.getAdditionalHdrs().toList()));
        objcCompilationContextBuilder.addPrivateHeaders(artifacts.getPrivateHdrs());

        if (artifacts.getArchive().isPresent()
            && J2ObjcLibrary.J2OBJC_SUPPORTED_RULES.contains(context.getRule().getRuleClass())) {
          objcProvider.addAll(J2OBJC_LIBRARY, artifacts.getArchive().asSet());
        }

        boolean usesCpp = false;
        for (Artifact sourceFile :
            Iterables.concat(artifacts.getSrcs(), artifacts.getNonArcSrcs())) {
          usesCpp = usesCpp || ObjcRuleClasses.CPP_SOURCES.matches(sourceFile.getExecPath());
        }
        if (usesCpp) {
          objcProvider.add(FLAG, USES_CPP);
        }
      }

      if (alwayslink) {
        for (CompilationArtifacts artifacts : compilationArtifacts.asSet()) {
          for (Artifact archive : artifacts.getArchive().asSet()) {
            objcProvider.add(FORCE_LOAD_LIBRARY, archive);
          }
        }
        for (Artifact archive : extraImportLibraries) {
          objcProvider.add(FORCE_LOAD_LIBRARY, archive);
        }
      }

      if (hasModuleMap) {
        CppModuleMap moduleMap = intermediateArtifacts.moduleMap();
        Optional<Artifact> umbrellaHeader = moduleMap.getUmbrellaHeader();
        if (umbrellaHeader.isPresent()) {
          objcProvider.add(UMBRELLA_HEADER, umbrellaHeader.get());
        }
        objcProvider.add(MODULE_MAP, moduleMap.getArtifact());
        objcProvider.addDirect(MODULE_MAP, moduleMap.getArtifact());
        objcProvider.add(TOP_LEVEL_MODULE_MAP, moduleMap);
      }

      objcProvider.addAll(LINKED_BINARY, linkedBinary.asSet());

      ObjcCompilationContext objcCompilationContext = objcCompilationContextBuilder.build();

      ObjcProvider.NativeBuilder objcProviderNativeBuilder =
          (ObjcProvider.NativeBuilder) objcProvider;
      if (purpose == Purpose.LINK_ONLY) {
        objcProviderNativeBuilder.setCcCompilationContext(
            objcCompilationContext.createCcCompilationContext());
      }
      return new ObjcCommon(
          purpose, objcProviderNativeBuilder, objcCompilationContext, compilationArtifacts);
    }

    private static boolean isCcLibrary(ConfiguredTargetAndData info) {
      try {
        String targetName = info.getTarget().getTargetKind();

        for (String ruleClassName : ObjcRuleClasses.CompilingRule.ALLOWED_CC_DEPS_RULE_CLASSES) {
          if (targetName.equals(ruleClassName + " rule")) {
            return true;
          }
        }
        return false;
      } catch (Exception e) {
        return false;
      }
    }
  }

  private final Purpose purpose;
  private final ObjcProvider.NativeBuilder objcProviderBuilder;
  private final ObjcCompilationContext objcCompilationContext;

  private final Optional<CompilationArtifacts> compilationArtifacts;

  private ObjcCommon(
      Purpose purpose,
      ObjcProvider.NativeBuilder objcProviderBuilder,
      ObjcCompilationContext objcCompilationContext,
      Optional<CompilationArtifacts> compilationArtifacts) {
    this.purpose = purpose;
    this.objcProviderBuilder = Preconditions.checkNotNull(objcProviderBuilder);
    this.objcCompilationContext = Preconditions.checkNotNull(objcCompilationContext);
    this.compilationArtifacts = Preconditions.checkNotNull(compilationArtifacts);
  }

  public Purpose getPurpose() {
    return purpose;
  }

  public ObjcProvider.NativeBuilder getObjcProviderBuilder() {
    return objcProviderBuilder;
  }

  public ObjcCompilationContext getObjcCompilationContext() {
    return objcCompilationContext;
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

  /**
   * Returns effective compilation options that do not arise from the crosstool.
   */
  static Iterable<String> getNonCrosstoolCopts(RuleContext ruleContext) {
    return Iterables.concat(
        ruleContext.getFragment(ObjcConfiguration.class).getCopts(),
        ruleContext.getExpander().withDataLocations().tokenized("copts"));
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

  static Iterable<String> notInContainerErrors(
      Iterable<Artifact> artifacts, Iterable<FileType> containerTypes) {
    Set<String> errors = new HashSet<>();
    for (Artifact artifact : artifacts) {
      boolean inContainer = nearestContainerMatching(containerTypes, artifact).isPresent();
      if (!inContainer) {
        errors.add(
            String.format(
                NOT_IN_CONTAINER_ERROR_FORMAT,
                artifact.getExecPath(),
                Iterables.toString(containerTypes)));
      }
    }
    return errors;
  }

  @VisibleForTesting
  static final String NOT_IN_CONTAINER_ERROR_FORMAT =
      "File '%s' is not in a directory of one of these type(s): %s";
}
