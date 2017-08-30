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

package com.google.devtools.build.lib.rules.objc;

import static com.google.devtools.build.lib.packages.ImplicitOutputsFunction.fromTemplates;
import static com.google.devtools.build.lib.rules.cpp.Link.LINK_LIBRARY_FILETYPES;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.DYNAMIC_FRAMEWORK_DIR;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.FORCE_LOAD_LIBRARY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.FRAMEWORK_SEARCH_PATH_ONLY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.IMPORTED_LIBRARY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.LIBRARY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.SDK_DYLIB;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.SDK_FRAMEWORK;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.STATIC_FRAMEWORK_DIR;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.COMPILABLE_SRCS_TYPE;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.HEADERS;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.NON_ARC_SRCS_TYPE;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.PRECOMPILED_SRCS_TYPE;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.SRCS_TYPE;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.STRIP;
import static java.nio.charset.StandardCharsets.ISO_8859_1;
import static java.util.stream.Collectors.toCollection;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.ListMultimap;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ParameterFile;
import com.google.devtools.build.lib.analysis.AnalysisEnvironment;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.PrerequisiteArtifacts;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoProviderMap;
import com.google.devtools.build.lib.analysis.actions.CommandLine;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.ParameterFileWriteAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesCollector;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesCollector.InstrumentationSpec;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesCollector.LocalMetadataCollector;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction.SafeImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.apple.ApplePlatform;
import com.google.devtools.build.lib.rules.apple.ApplePlatform.PlatformType;
import com.google.devtools.build.lib.rules.apple.AppleToolchain;
import com.google.devtools.build.lib.rules.apple.XcodeConfig;
import com.google.devtools.build.lib.rules.cpp.CcToolchain;
import com.google.devtools.build.lib.rules.cpp.CcToolchainProvider;
import com.google.devtools.build.lib.rules.cpp.CppFileTypes;
import com.google.devtools.build.lib.rules.cpp.CppHelper;
import com.google.devtools.build.lib.rules.cpp.CppModuleMap;
import com.google.devtools.build.lib.rules.cpp.CppModuleMapAction;
import com.google.devtools.build.lib.rules.cpp.FdoSupportProvider;
import com.google.devtools.build.lib.rules.cpp.UmbrellaHeaderAction;
import com.google.devtools.build.lib.rules.objc.ObjcCommandLineOptions.ObjcCrosstoolMode;
import com.google.devtools.build.lib.util.FileTypeSet;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.stream.Stream;
import javax.annotation.Nullable;

/**
 * Support for rules that compile sources. Provides ways to determine files that should be output,
 * registering Xcode settings and generating the various actions that might be needed for
 * compilation.
 *
 * <p>A subclass should express a particular strategy for compile and link action registration.
 * Subclasses should implement the API without adding new visible methods - rule implementations
 * should be able to use a {@link CompilationSupport} instance to compile and link source without
 * knowing the subclass being used.
 *
 * <p>Methods on this class can be called in any order without impacting the result.
 */
public abstract class CompilationSupport {

  @VisibleForTesting
  static final String OBJC_MODULE_CACHE_DIR_NAME = "_objc_module_cache";
  
  @VisibleForTesting
  static final String MODULES_CACHE_PATH_WARNING =
      "setting '-fmodules-cache-path' manually in copts is unsupported";

  @VisibleForTesting
  static final String ABSOLUTE_INCLUDES_PATH_FORMAT =
      "The path '%s' is absolute, but only relative paths are allowed.";

  @VisibleForTesting
  static final ImmutableList<String> LINKER_COVERAGE_FLAGS =
      ImmutableList.of("-ftest-coverage", "-fprofile-arcs");

  @VisibleForTesting
  static final ImmutableList<String> LINKER_LLVM_COVERAGE_FLAGS =
      ImmutableList.of("-fprofile-instr-generate");

  // Flags for clang 6.1(xcode 6.4)
  @VisibleForTesting
  static final ImmutableList<String> CLANG_GCOV_COVERAGE_FLAGS =
      ImmutableList.of("-fprofile-arcs", "-ftest-coverage");

  @VisibleForTesting
  static final ImmutableList<String> CLANG_LLVM_COVERAGE_FLAGS =
      ImmutableList.of("-fprofile-instr-generate", "-fcoverage-mapping");

  // These are added by Xcode when building, because the simulator is built on OSX
  // frameworks so we aim compile to match the OSX objc runtime.
  @VisibleForTesting
  static final ImmutableList<String> SIMULATOR_COMPILE_FLAGS =
      ImmutableList.of(
          "-fexceptions", "-fasm-blocks", "-fobjc-abi-version=2", "-fobjc-legacy-dispatch");

  private static final String FRAMEWORK_SUFFIX = ".framework";

  /** Selects cc libraries that have alwayslink=1. */
  protected static final Predicate<Artifact> ALWAYS_LINKED_CC_LIBRARY =
      input -> LINK_LIBRARY_FILETYPES.matches(input.getFilename());

  /**
   * Returns the location of the xcrunwrapper tool.
   */
  public static final FilesToRunProvider xcrunwrapper(RuleContext ruleContext) {
    return ruleContext.getExecutablePrerequisite("$xcrunwrapper", Mode.HOST);
  }

  /**
   * Returns the location of the libtool tool.
   */
  public static final FilesToRunProvider libtool(RuleContext ruleContext) {
    return ruleContext.getExecutablePrerequisite(ObjcRuleClasses.LIBTOOL_ATTRIBUTE, Mode.HOST);
  }
  
  /**
   * Files which can be instrumented along with the attributes in which they may occur and the
   * attributes along which they are propagated from dependencies (via
   * {@link InstrumentedFilesProvider}).
   */
  private static final InstrumentationSpec INSTRUMENTATION_SPEC =
      new InstrumentationSpec(
              FileTypeSet.of(
                  ObjcRuleClasses.NON_CPP_SOURCES,
                  ObjcRuleClasses.CPP_SOURCES,
                  HEADERS))
          .withSourceAttributes("srcs", "non_arc_srcs", "hdrs")
          .withDependencyAttributes("deps", "data", "binary", "xctest_app");

  /**
   * Defines a library that contains the transitive closure of dependencies.
   */
  public static final SafeImplicitOutputsFunction FULLY_LINKED_LIB =
      fromTemplates("%{name}_fully_linked.a");

  /**
   * Iterable wrapper providing strong type safety for arguments to binary linking.
   */
  static final class ExtraLinkArgs extends IterableWrapper<String> {
    ExtraLinkArgs(String... args) {
      super(args);
    }

    ExtraLinkArgs(Iterable<String> args) {
      super(args);
    }
  }

  /**
   * Iterable wrapper providing strong type safety for extra compile flags.
   */
  static final class ExtraCompileArgs extends IterableWrapper<String> {
    static final ExtraCompileArgs NONE = new ExtraCompileArgs();
    ExtraCompileArgs(String... args) {
      super(args);
    }
  }

  @VisibleForTesting
  static final String FILE_IN_SRCS_AND_HDRS_WARNING_FORMAT =
      "File '%s' is in both srcs and hdrs.";

  @VisibleForTesting
  static final String FILE_IN_SRCS_AND_NON_ARC_SRCS_ERROR_FORMAT =
      "File '%s' is present in both srcs and non_arc_srcs which is forbidden.";

  static final ImmutableList<String> DEFAULT_COMPILER_FLAGS = ImmutableList.of("-DOS_IOS");

  static final ImmutableList<String> DEFAULT_LINKER_FLAGS = ImmutableList.of("-ObjC");

  /**
   * Set of {@link com.google.devtools.build.lib.util.FileType} of source artifacts that are
   * compatible with header thinning.
   */
  protected static final FileTypeSet SOURCES_FOR_HEADER_THINNING =
      FileTypeSet.of(
          CppFileTypes.OBJC_SOURCE,
          CppFileTypes.OBJCPP_SOURCE,
          CppFileTypes.CPP_SOURCE,
          CppFileTypes.C_SOURCE);

  /**
   * Returns information about the given rule's compilation artifacts.
   */
  // TODO(bazel-team): Remove this information from ObjcCommon and move it internal to this class.
  static CompilationArtifacts compilationArtifacts(RuleContext ruleContext) {
    return compilationArtifacts(ruleContext,  ObjcRuleClasses.intermediateArtifacts(ruleContext));
  }

  /**
   * Returns information about the given rule's compilation artifacts. Dependencies specified
   * in the current rule's attributes are obtained via {@code ruleContext}. Output locations
   * are determined using the given {@code intermediateArtifacts} object. The fact that these
   * are distinct objects allows the caller to generate compilation actions pertaining to
   * a configuration separate from the current rule's configuration.
   */
  static CompilationArtifacts compilationArtifacts(RuleContext ruleContext,
      IntermediateArtifacts intermediateArtifacts) {
    PrerequisiteArtifacts srcs = ruleContext.getPrerequisiteArtifacts("srcs", Mode.TARGET)
        .errorsForNonMatching(SRCS_TYPE);
    return new CompilationArtifacts.Builder()
        .addSrcs(srcs.filter(COMPILABLE_SRCS_TYPE).list())
        .addNonArcSrcs(
            ruleContext
                .getPrerequisiteArtifacts("non_arc_srcs", Mode.TARGET)
                .errorsForNonMatching(NON_ARC_SRCS_TYPE)
                .list())
        .addPrivateHdrs(srcs.filter(HEADERS).list())
        .addPrecompiledSrcs(srcs.filter(PRECOMPILED_SRCS_TYPE).list())
        .setIntermediateArtifacts(intermediateArtifacts)
        .build();
  }

  /** Returns a list of framework search path flags for clang actions. */
  static Iterable<String> commonFrameworkFlags(
      ObjcProvider provider, RuleContext ruleContext, ApplePlatform applePlaform) {
    return Interspersing.beforeEach("-F",
        commonFrameworkNames(provider, ruleContext, applePlaform));
  }

  /** Returns a list of frameworks for clang actions. */
  static Iterable<String> commonFrameworkNames(
      ObjcProvider provider, RuleContext ruleContext, ApplePlatform platform) {

    ImmutableList.Builder<String> frameworkNames =
        new ImmutableList.Builder<String>()
            .add(AppleToolchain.sdkFrameworkDir(platform, ruleContext));
    // As of sdk8.1, XCTest is in a base Framework dir.
    if (platform.getType() != PlatformType.WATCHOS) { // WatchOS does not have this directory.
      frameworkNames.add(AppleToolchain.platformDeveloperFrameworkDir(platform));
    }
    return frameworkNames
        // Add custom (non-SDK) framework search paths. For each framework foo/bar.framework,
        // include "foo" as a search path.
        .addAll(PathFragment.safePathStrings(
            uniqueParentDirectories(provider.get(STATIC_FRAMEWORK_DIR))))
        .addAll(PathFragment.safePathStrings(
            uniqueParentDirectories(provider.get(DYNAMIC_FRAMEWORK_DIR))))
        .addAll(
            PathFragment.safePathStrings(
                uniqueParentDirectories(provider.get(FRAMEWORK_SEARCH_PATH_ONLY))))
        .build();
  }

  protected final RuleContext ruleContext;
  protected final BuildConfiguration buildConfiguration;
  protected final ObjcConfiguration objcConfiguration;
  protected final AppleConfiguration appleConfiguration;
  protected final CompilationAttributes attributes;
  protected final IntermediateArtifacts intermediateArtifacts;
  protected final boolean useDeps;
  protected final Map<String, NestedSet<Artifact>> outputGroupCollector;
  protected final CcToolchainProvider toolchain;
  protected final ImmutableList.Builder<TransitiveInfoProviderMap> providerCollector;
  protected final boolean isTestRule;
  protected final boolean usePch;

  /**
   * Creates a new compilation support for the given rule and build configuration.
   *
   * <p>All actions will be created under the given build configuration, which may be different than
   * the current rule context configuration.
   *
   * <p>The compilation and linking flags will be retrieved from the given compilation attributes.
   * The names of the generated artifacts will be retrieved from the given intermediate artifacts.
   *
   * <p>By instantiating multiple compilation supports for the same rule but with intermediate
   * artifacts with different output prefixes, multiple archives can be compiled for the same rule
   * context.
   */
  protected CompilationSupport(
      RuleContext ruleContext,
      BuildConfiguration buildConfiguration,
      IntermediateArtifacts intermediateArtifacts,
      CompilationAttributes compilationAttributes,
      boolean useDeps,
      Map<String, NestedSet<Artifact>> outputGroupCollector,
      CcToolchainProvider toolchain,
      ImmutableList.Builder<TransitiveInfoProviderMap> providerCollector,
      boolean isTestRule,
      boolean usePch) {
    this.ruleContext = ruleContext;
    this.buildConfiguration = buildConfiguration;
    this.objcConfiguration = buildConfiguration.getFragment(ObjcConfiguration.class);
    this.appleConfiguration = buildConfiguration.getFragment(AppleConfiguration.class);
    this.attributes = compilationAttributes;
    this.intermediateArtifacts = intermediateArtifacts;
    this.useDeps = useDeps;
    this.isTestRule = isTestRule;
    this.outputGroupCollector = outputGroupCollector;
    this.providerCollector = providerCollector;
    this.usePch = usePch;
    // TODO(b/62143697): Remove this check once all rules are using the crosstool support.
    if (ruleContext
        .attributes()
        .has(CcToolchain.CC_TOOLCHAIN_DEFAULT_ATTRIBUTE_NAME, BuildType.LABEL)) {
      if (toolchain == null) {
        toolchain =  CppHelper.getToolchainUsingDefaultCcToolchainAttribute(ruleContext);
      }
      this.toolchain = toolchain;
    } else {
      // Since the rule context doesn't have a toolchain at all, ignore any provided override.
      this.toolchain = null;
    }
  }

  /** Builder for {@link CompilationSupport} */
  public static class Builder {
    private RuleContext ruleContext;
    private BuildConfiguration buildConfiguration;
    private IntermediateArtifacts intermediateArtifacts;
    private CompilationAttributes compilationAttributes;
    private boolean useDeps = true;
    private Map<String, NestedSet<Artifact>> outputGroupCollector;
    private ImmutableList.Builder<TransitiveInfoProviderMap> providerCollector =
        ImmutableList.builder();
    private boolean isObjcLibrary = false;
    private CcToolchainProvider toolchain;
    private boolean isTestRule = false;
    private boolean usePch = true;

    /** Sets the {@link RuleContext} for the calling target. */
    public Builder setRuleContext(RuleContext ruleContext) {
      this.ruleContext = ruleContext;
      return this;
    }

    /** Sets the {@link BuildConfiguration} for the calling target. */
    public Builder setConfig(BuildConfiguration buildConfiguration) {
      this.buildConfiguration = buildConfiguration;
      return this;
    }

    /** Sets {@link IntermediateArtifacts} for deriving artifact paths. */
    public Builder setIntermediateArtifacts(IntermediateArtifacts intermediateArtifacts) {
      this.intermediateArtifacts = intermediateArtifacts;
      return this;
    }

    /** Sets {@link CompilationAttributes} for the calling target. */
    public Builder setCompilationAttributes(CompilationAttributes compilationAttributes) {
      this.compilationAttributes = compilationAttributes;
      return this;
    }

    /**
     * Sets that this {@link CompilationSupport} will not take deps into account in determining
     * compilation actions.
     */
    public Builder doNotUseDeps() {
      this.useDeps = false;
      return this;
    }

    /**
     * Sets that this {@link CompilationSupport} will not use the pch from the rule context in
     * determining compilation actions.
     */
    public Builder doNotUsePch() {
      this.usePch = false;
      return this;
    }

    /**
     * Indicates that this CompilationSupport is for use in an objc_library target. This will cause
     * CrosstoolCompilationSupport to be used if --experimental_objc_crosstool=library
     */
    public Builder setIsObjcLibrary() {
      this.isObjcLibrary = true;
      return this;
    }

    /**
     * Indicates that this CompilationSupport is for use in a test rule.
     */
    public Builder setIsTestRule() {
      this.isTestRule = true;
      return this;
    }

    /**
     * Causes the provided map to be updated with output groups produced by compile action
     * registration.
     *
     * <p>This map is intended to be mutated by {@link
     * CompilationSupport#registerCompileAndArchiveActions}. The added output groups should be
     * exported by the calling rule class implementation.
     */
    public Builder setOutputGroupCollector(Map<String, NestedSet<Artifact>> outputGroupCollector) {
      this.outputGroupCollector = outputGroupCollector;
      return this;
    }

    /**
     * Sets {@link CcToolchainProvider} for the calling target.
     *
     * <p>This is needed if it can't correctly be inferred directly from the rule context. Setting
     * to null causes the default to be used as if this was never called.
     */
    public Builder setToolchainProvider(CcToolchainProvider toolchain) {
      this.toolchain = toolchain;
      return this;
    }

    /**
     * Causes the provided list to be updated with providers that should be exported by this target.
     */
    public Builder setProviderCollector(
        ImmutableList.Builder<TransitiveInfoProviderMap> providerCollector) {
      Preconditions.checkNotNull(providerCollector);
      this.providerCollector = providerCollector;
      return this;
    }

    /**
     * Returns a {@link CompilationSupport} instance. This is either a {@link
     * CrosstoolCompilationSupport} or {@link LegacyCompilationSupport} depending on the value of
     * --experimental_objc_crosstool.
     */
    public CompilationSupport build() {
      Preconditions.checkNotNull(ruleContext, "CompilationSupport is missing RuleContext");

      if (buildConfiguration == null) {
        buildConfiguration = ruleContext.getConfiguration();
      }

      if (intermediateArtifacts == null) {
        intermediateArtifacts =
            ObjcRuleClasses.intermediateArtifacts(ruleContext, buildConfiguration);
      }

      if (compilationAttributes == null) {
        compilationAttributes = CompilationAttributes.Builder.fromRuleContext(ruleContext).build();
      }

      if (outputGroupCollector == null) {
        outputGroupCollector = new TreeMap<>();
      }

      ObjcCrosstoolMode objcCrosstoolMode =
          buildConfiguration.getFragment(ObjcConfiguration.class).getObjcCrosstoolMode();
      if (objcCrosstoolMode == ObjcCrosstoolMode.ALL
          || (isObjcLibrary && objcCrosstoolMode == ObjcCrosstoolMode.LIBRARY)) {
        return new CrosstoolCompilationSupport(
            ruleContext,
            buildConfiguration,
            intermediateArtifacts,
            compilationAttributes,
            useDeps,
            outputGroupCollector,
            toolchain,
            providerCollector,
            isTestRule,
            usePch);
      } else {
        return new LegacyCompilationSupport(
            ruleContext,
            buildConfiguration,
            intermediateArtifacts,
            compilationAttributes,
            useDeps,
            outputGroupCollector,
            toolchain,
            isTestRule,
            usePch);
      }
    }
  }

 /**
   * Registers all actions necessary to compile this rule's sources and archive them.
   *
   * @param compilationArtifacts collection of artifacts required for the compilation
   * @param objcProvider provides all compiling and linking information to register these actions
   * @return this compilation support
   * @throws RuleErrorException for invalid crosstool files
   */
  CompilationSupport registerCompileAndArchiveActions(
      CompilationArtifacts compilationArtifacts,
      ObjcProvider objcProvider) throws RuleErrorException, InterruptedException {
    return registerCompileAndArchiveActions(
        compilationArtifacts,
        objcProvider,
        ExtraCompileArgs.NONE,
        ImmutableList.<PathFragment>of(),
        toolchain,
        maybeGetFdoSupport());
  }

  /**
   * Registers all actions necessary to compile this rule's sources and archive them.
   *
   * @param compilationArtifacts collection of artifacts required for the compilation
   * @param objcProvider provides all compiling and linking information to register these actions
   * @param toolchain the toolchain to be used in determining command lines
   * @return this compilation support
   * @throws RuleErrorException for invalid crosstool files
   */
  CompilationSupport registerCompileAndArchiveActions(
      CompilationArtifacts compilationArtifacts,
      ObjcProvider objcProvider,
      CcToolchainProvider toolchain)
      throws RuleErrorException, InterruptedException {
    return registerCompileAndArchiveActions(
        compilationArtifacts,
        objcProvider,
        ExtraCompileArgs.NONE,
        ImmutableList.<PathFragment>of(),
        toolchain,
        maybeGetFdoSupport());
  }

  /**
   * Registers all actions necessary to compile this rule's sources and archive them.
   *
   * @param common common information about this rule and its dependencies
   * @return this compilation support
   * @throws RuleErrorException for invalid crosstool files
   */
  CompilationSupport registerCompileAndArchiveActions(ObjcCommon common)
      throws RuleErrorException, InterruptedException {
    return registerCompileAndArchiveActions(
        common, ExtraCompileArgs.NONE, ImmutableList.<PathFragment>of());
  }

  /**
   * Registers all actions necessary to compile this rule's sources and archive them.
   *
   * @param common common information about this rule and its dependencies
   * @param priorityHeaders priority headers to be included before the dependency headers
   * @return this compilation support
   * @throws RuleErrorException for invalid crosstool files
   */
  CompilationSupport registerCompileAndArchiveActions(
      ObjcCommon common, Iterable<PathFragment> priorityHeaders)
      throws RuleErrorException, InterruptedException {
    return registerCompileAndArchiveActions(common, ExtraCompileArgs.NONE, priorityHeaders);
  }

  /**
   * Registers all actions necessary to compile this rule's sources and archive them.
   *
   * @param common common information about this rule and its dependencies
   * @param extraCompileArgs args to be added to compile actions
   * @return this compilation support
   * @throws RuleErrorException for invalid crosstool files
   */
  CompilationSupport registerCompileAndArchiveActions(
      ObjcCommon common, ExtraCompileArgs extraCompileArgs)
      throws RuleErrorException, InterruptedException {
    return registerCompileAndArchiveActions(
        common, extraCompileArgs, ImmutableList.<PathFragment>of());
  }
 
  /**
   * Registers an action to create an archive artifact by fully (statically) linking all transitive
   * dependencies of this rule.
   *
   * @param objcProvider provides all compiling and linking information to create this artifact
   * @param outputArchive the output artifact for this action
   */
  public CompilationSupport registerFullyLinkAction(
      ObjcProvider objcProvider, Artifact outputArchive) throws InterruptedException {
    return registerFullyLinkAction(
        objcProvider,
        outputArchive,
        toolchain,
        maybeGetFdoSupport());
  }

  /**
   * Registers an action to create an archive artifact by fully (statically) linking all transitive
   * dependencies of this rule.
   *
   * @param objcProvider provides all compiling and linking information to create this artifact
   * @param outputArchive the output artifact for this action
   * @param ccToolchain the cpp toolchain provider, may be null
   * @param fdoSupport the cpp FDO support provider, may be null
   */
  public CompilationSupport registerFullyLinkAction(
      ObjcProvider objcProvider, Artifact outputArchive, @Nullable CcToolchainProvider ccToolchain,
      @Nullable FdoSupportProvider fdoSupport) throws InterruptedException {
    ImmutableList<Artifact> inputArtifacts = ImmutableList.<Artifact>builder()
        .addAll(objcProvider.getObjcLibraries())
        .addAll(objcProvider.get(IMPORTED_LIBRARY))
        .addAll(objcProvider.getCcLibraries()).build();
    return registerFullyLinkAction(
        objcProvider,
        inputArtifacts,
        outputArchive,
        ccToolchain,
        fdoSupport);
  }
  
  /**
   * Registers an action to create an archive artifact by fully (statically) linking all transitive
   * dependencies of this rule *except* for dependencies given in {@code avoidsDeps}.
   *
   * @param objcProvider provides all compiling and linking information to create this artifact
   * @param outputArchive the output artifact for this action
   * @param avoidsDeps list of providers with dependencies that should not be linked into the output
   *     artifact
   */
  public CompilationSupport registerFullyLinkActionWithAvoids(
      ObjcProvider objcProvider, Artifact outputArchive, Iterable<ObjcProvider> avoidsDeps)
      throws InterruptedException {
    ImmutableSet.Builder<Artifact> avoidsDepsArtifacts = ImmutableSet.builder();

    for (ObjcProvider avoidsProvider : avoidsDeps) {
      avoidsDepsArtifacts.addAll(avoidsProvider.getObjcLibraries())
          .addAll(avoidsProvider.get(IMPORTED_LIBRARY))
          .addAll(avoidsProvider.getCcLibraries());
    }
    ImmutableList<Artifact> depsArtifacts = ImmutableList.<Artifact>builder()
        .addAll(objcProvider.getObjcLibraries())
        .addAll(objcProvider.get(IMPORTED_LIBRARY))
        .addAll(objcProvider.getCcLibraries()).build();
    
    Iterable<Artifact> inputArtifacts = Iterables.filter(depsArtifacts,
        Predicates.not(Predicates.in(avoidsDepsArtifacts.build())));
    return registerFullyLinkAction(
        objcProvider,
        inputArtifacts,
        outputArchive,
        toolchain,
        maybeGetFdoSupport());
  }
    
  /**
   * Returns a provider that collects this target's instrumented sources as well as those of its
   * dependencies.
   *
   * @param common common information about this rule and its dependencies
   * @return an instrumented files provider
   */
  public InstrumentedFilesProvider getInstrumentedFilesProvider(ObjcCommon common) {
    ImmutableList.Builder<Artifact> oFiles = ImmutableList.builder();

    if (common.getCompilationArtifacts().isPresent()) {
      CompilationArtifacts artifacts = common.getCompilationArtifacts().get();
      for (Artifact artifact : Iterables.concat(artifacts.getSrcs(), artifacts.getNonArcSrcs())) {
        oFiles.add(intermediateArtifacts.objFile(artifact));
      }
    }

    return InstrumentedFilesCollector.collect(
        ruleContext,
        INSTRUMENTATION_SPEC,
        new ObjcCoverageMetadataCollector(),
        oFiles.build(),
        getGcovForObjectiveCIfNeeded(),
        // The COVERAGE_GCOV_PATH environment variable is added in TestSupport#getExtraProviders()
        NestedSetBuilder.<Pair<String, String>>emptySet(Order.COMPILE_ORDER),
        !isTestRule);
  }

  /**
   * Registers an action that will generate a clang module map for this target, using the hdrs
   * attribute of this rule.
   */
  CompilationSupport registerGenerateModuleMapAction(CompilationArtifacts compilationArtifacts) {
    // TODO(bazel-team): Include textual headers in the module map when Xcode 6 support is
    // dropped.
    // TODO(b/32225593): Include private headers in the module map.
    Iterable<Artifact> publicHeaders = attributes.hdrs();
      publicHeaders = Iterables.concat(publicHeaders, compilationArtifacts.getAdditionalHdrs());
    CppModuleMap moduleMap = intermediateArtifacts.moduleMap();
    registerGenerateModuleMapAction(moduleMap, publicHeaders);

    Optional<Artifact> umbrellaHeader = moduleMap.getUmbrellaHeader();
    if (umbrellaHeader.isPresent()) {
      registerGenerateUmbrellaHeaderAction(umbrellaHeader.get(), publicHeaders);
    }

    return this;
  }

  /**
   * Validates compilation-related attributes on this rule.
   *
   * @return this compilation support
   * @throws RuleErrorException if there are attribute errors
   */
  CompilationSupport validateAttributes() throws RuleErrorException {
    for (PathFragment absoluteInclude :
        Iterables.filter(attributes.includes(), PathFragment::isAbsolute)) {
      ruleContext.attributeError(
          "includes", String.format(ABSOLUTE_INCLUDES_PATH_FORMAT, absoluteInclude));
    }

    if (ruleContext.attributes().has("srcs", BuildType.LABEL_LIST)) {
      ImmutableSet<Artifact> hdrsSet = ImmutableSet.copyOf(attributes.hdrs());
      ImmutableSet<Artifact> srcsSet =
          ImmutableSet.copyOf(ruleContext.getPrerequisiteArtifacts("srcs", Mode.TARGET).list());

      // Check for overlap between srcs and hdrs.
      for (Artifact header : Sets.intersection(hdrsSet, srcsSet)) {
        String path = header.getRootRelativePath().toString();
        ruleContext.attributeWarning(
            "srcs", String.format(FILE_IN_SRCS_AND_HDRS_WARNING_FORMAT, path));
      }

      // Check for overlap between srcs and non_arc_srcs.
      ImmutableSet<Artifact> nonArcSrcsSet =
          ImmutableSet.copyOf(
              ruleContext.getPrerequisiteArtifacts("non_arc_srcs", Mode.TARGET).list());
      for (Artifact conflict : Sets.intersection(nonArcSrcsSet, srcsSet)) {
        String path = conflict.getRootRelativePath().toString();
        ruleContext.attributeError(
            "srcs", String.format(FILE_IN_SRCS_AND_NON_ARC_SRCS_ERROR_FORMAT, path));
      }
    }

    ruleContext.assertNoErrors();
    return this;
  }

  /**
   * Registers all actions necessary to compile this rule's sources and archive them.
   *
   * @param compilationArtifacts collection of artifacts required for the compilation
   * @param objcProvider provides all compiling and linking information to register these actions
   * @param extraCompileArgs args to be added to compile actions
   * @param priorityHeaders priority headers to be included before the dependency headers
   * @param ccToolchain the cpp toolchain provider, may be null
   * @param fdoSupport the cpp FDO support provider, may be null
   * @return this compilation support
   * @throws RuleErrorException for invalid crosstool files
   */
  abstract CompilationSupport registerCompileAndArchiveActions(
      CompilationArtifacts compilationArtifacts, ObjcProvider objcProvider,
      ExtraCompileArgs extraCompileArgs, Iterable<PathFragment> priorityHeaders,
      @Nullable CcToolchainProvider ccToolchain, @Nullable FdoSupportProvider fdoSupport)
      throws RuleErrorException, InterruptedException;

  /**
   * Registers all actions necessary to compile this rule's sources and archive them.
   *
   * @param common common information about this rule and its dependencies
   * @param extraCompileArgs args to be added to compile actions
   * @param priorityHeaders priority headers to be included before the dependency headers
   * @return this compilation support
   * @throws RuleErrorException for invalid crosstool files
   */
  CompilationSupport registerCompileAndArchiveActions(
      ObjcCommon common, ExtraCompileArgs extraCompileArgs, Iterable<PathFragment> priorityHeaders)
      throws RuleErrorException, InterruptedException {
    if (common.getCompilationArtifacts().isPresent()) {
      registerCompileAndArchiveActions(
          common.getCompilationArtifacts().get(),
          common.getObjcProvider(),
          extraCompileArgs,
          priorityHeaders,
          toolchain,
          maybeGetFdoSupport());
    }
    return this;
  }
  /**
   * Registers any actions necessary to link this rule and its dependencies. Manually sets the
   * toolchain.
   *
   * <p>Dsym bundle is generated if {@link ObjcConfiguration#generateDsym()} is set.
   *
   * <p>When Bazel flags {@code --compilation_mode=opt} and {@code --objc_enable_binary_stripping}
   * are specified, additional optimizations will be performed on the linked binary: all-symbol
   * stripping (using {@code /usr/bin/strip}) and dead-code stripping (using linker flags: {@code
   * -dead_strip} and {@code -no_dead_strip_inits_and_terms}).
   *
   * @param objcProvider common information about this rule's attributes and its dependencies
   * @param j2ObjcMappingFileProvider contains mapping files for j2objc transpilation
   * @param j2ObjcEntryClassProvider contains j2objc entry class information for dead code removal
   * @param extraLinkArgs any additional arguments to pass to the linker
   * @param extraLinkInputs any additional input artifacts to pass to the link action
   * @param dsymOutputType the file type of the dSYM bundle to be generated
   * @param toolchain the CROSSTOOL-derived toolchain for use in linking
   * @return this compilation support
   */
  abstract CompilationSupport registerLinkActions(
      ObjcProvider objcProvider,
      J2ObjcMappingFileProvider j2ObjcMappingFileProvider,
      J2ObjcEntryClassProvider j2ObjcEntryClassProvider,
      ExtraLinkArgs extraLinkArgs,
      Iterable<Artifact> extraLinkInputs,
      DsymOutputType dsymOutputType,
      CcToolchainProvider toolchain)
      throws InterruptedException;

  /**
   * Registers any actions necessary to link this rule and its dependencies. Automatically infers
   * the toolchain from the configuration of this CompilationSupport - if a different toolchain is
   * required, use the custom toolchain override.
   *
   * <p>Dsym bundle is generated if {@link ObjcConfiguration#generateDsym()} is set.
   *
   * <p>When Bazel flags {@code --compilation_mode=opt} and {@code --objc_enable_binary_stripping}
   * are specified, additional optimizations will be performed on the linked binary: all-symbol
   * stripping (using {@code /usr/bin/strip}) and dead-code stripping (using linker flags: {@code
   * -dead_strip} and {@code -no_dead_strip_inits_and_terms}).
   *
   * @param objcProvider common information about this rule's attributes and its dependencies
   * @param j2ObjcMappingFileProvider contains mapping files for j2objc transpilation
   * @param j2ObjcEntryClassProvider contains j2objc entry class information for dead code removal
   * @param extraLinkArgs any additional arguments to pass to the linker
   * @param extraLinkInputs any additional input artifacts to pass to the link action
   * @param dsymOutputType the file type of the dSYM bundle to be generated
   * @return this compilation support
   */
  CompilationSupport registerLinkActions(
      ObjcProvider objcProvider,
      J2ObjcMappingFileProvider j2ObjcMappingFileProvider,
      J2ObjcEntryClassProvider j2ObjcEntryClassProvider,
      ExtraLinkArgs extraLinkArgs,
      Iterable<Artifact> extraLinkInputs,
      DsymOutputType dsymOutputType)
      throws InterruptedException {
    return registerLinkActions(
        objcProvider,
        j2ObjcMappingFileProvider,
        j2ObjcEntryClassProvider,
        extraLinkArgs,
        extraLinkInputs,
        dsymOutputType,
        CppHelper.getToolchainUsingDefaultCcToolchainAttribute(ruleContext));
  }

  /**
   * Returns the copts for the compile action in the current rule context (using a combination of
   * the rule's "copts" attribute as well as the current configuration copts).
   */
  protected Iterable<String> getCompileRuleCopts() {
    List<String> copts =
        Stream.concat(objcConfiguration.getCopts().stream(), attributes.copts().stream())
            .collect(toCollection(ArrayList::new));

    for (String copt : copts) {
      if (copt.contains("-fmodules-cache-path")) {
        // Bazel decides on the cache path location.
        ruleContext.ruleWarning(MODULES_CACHE_PATH_WARNING);
      }
    }

    if (attributes.enableModules() && !getCustomModuleMap(ruleContext).isPresent()) {
      copts.add("-fmodules");
    }
    if (copts.contains("-fmodules")) {
      // If modules are enabled, clang caches module information. If unspecified, this is a
      // system-wide cache directory, which is a problem for remote executors which may run
      // multiple actions with different source trees that can't share this cache.
      // We thus set its path to the root of the genfiles directory.
      // Unfortunately, this cache contains non-hermetic information, thus we avoid declaring it as
      // an implicit output (as outputs must be hermetic).
      String cachePath =
          buildConfiguration.getGenfilesFragment() + "/" + OBJC_MODULE_CACHE_DIR_NAME;
      copts.add("-fmodules-cache-path=" + cachePath);
    }
    return copts;
  }

  /**
   * Registers an action that writes given set of object files to the given objList. This objList is
   * suitable to signal symbols to archive in a libtool archiving invocation.
   */
  protected CompilationSupport registerObjFilelistAction(
      Iterable<Artifact> objFiles, Artifact objList) {
    ImmutableSet<Artifact> dedupedObjFiles = ImmutableSet.copyOf(objFiles);
    CustomCommandLine.Builder objFilesToLinkParam = new CustomCommandLine.Builder();
    ImmutableList.Builder<Artifact> treeObjFiles = new ImmutableList.Builder<>();

    for (Artifact objFile : dedupedObjFiles) {
      // If the obj file is a tree artifact, we need to expand it into the contained individual
      // files properly.
      if (objFile.isTreeArtifact()) {
        treeObjFiles.add(objFile);
        objFilesToLinkParam.addExpandedTreeArtifactExecPaths(objFile);
      } else {
        objFilesToLinkParam.addPath(objFile.getExecPath());
      }
    }

    ruleContext.registerAction(
        new ParameterFileWriteAction(
            ruleContext.getActionOwner(),
            treeObjFiles.build(),
            objList,
            objFilesToLinkParam.build(),
            ParameterFile.ParameterFileType.UNQUOTED,
            ISO_8859_1));
    return this;
  }
 
  /**
   * Registers an action to create an archive artifact by fully (statically) linking all transitive
   * dependencies of this rule.
   *
   * @param objcProvider provides all compiling and linking information to create this artifact
   * @param inputArtifacts inputs for this action
   * @param outputArchive the output artifact for this action
   * @param ccToolchain the cpp toolchain provider, may be null
   * @param fdoSupport the cpp FDO support provider, may be null
   * @return this {@link CompilationSupport} instance
   */
  protected abstract CompilationSupport registerFullyLinkAction(
      ObjcProvider objcProvider, Iterable<Artifact> inputArtifacts, Artifact outputArchive,
      @Nullable CcToolchainProvider ccToolchain, @Nullable FdoSupportProvider fdoSupport)
      throws InterruptedException;

  private PathFragment removeSuffix(PathFragment path, String suffix) {
    String name = path.getBaseName();
    Preconditions.checkArgument(
        name.endsWith(suffix), "expected %s to end with %s, but it does not", name, suffix);
    return path.replaceName(name.substring(0, name.length() - suffix.length()));
  }

  protected CompilationSupport registerDsymActions(DsymOutputType dsymOutputType) {
    Artifact tempDsymBundleZip = intermediateArtifacts.tempDsymBundleZip(dsymOutputType);
    Artifact linkedBinary =
        objcConfiguration.shouldStripBinary()
            ? intermediateArtifacts.unstrippedSingleArchitectureBinary()
            : intermediateArtifacts.strippedSingleArchitectureBinary();
    Artifact debugSymbolFile = intermediateArtifacts.dsymSymbol(dsymOutputType);
    Artifact dsymPlist = intermediateArtifacts.dsymPlist(dsymOutputType);

    PathFragment dsymOutputDir = removeSuffix(tempDsymBundleZip.getExecPath(), ".temp.zip");
    PathFragment dsymPlistZipEntry = dsymPlist.getExecPath().relativeTo(dsymOutputDir);
    PathFragment debugSymbolFileZipEntry =
        debugSymbolFile
            .getExecPath()
            .replaceName(linkedBinary.getFilename())
            .relativeTo(dsymOutputDir);

    StringBuilder unzipDsymCommand =
        new StringBuilder()
            .append(
                String.format(
                    "unzip -p %s %s > %s",
                    tempDsymBundleZip.getExecPathString(),
                    dsymPlistZipEntry,
                    dsymPlist.getExecPathString()))
            .append(
                String.format(
                    " && unzip -p %s %s > %s",
                    tempDsymBundleZip.getExecPathString(),
                    debugSymbolFileZipEntry,
                    debugSymbolFile.getExecPathString()));

    ruleContext.registerAction(
        new SpawnAction.Builder()
            .setMnemonic("UnzipDsym")
            .setShellCommand(unzipDsymCommand.toString())
            .addInput(tempDsymBundleZip)
            .addOutput(dsymPlist)
            .addOutput(debugSymbolFile)
            .build(ruleContext));

    return this;
  }

 /**
   * Returns all framework names to pass to the linker using {@code -framework} flags. For a
   * framework in the directory foo/bar.framework, the name is "bar". Each framework is found
   * without using the full path by means of the framework search paths. Search paths are added by
   * {@link#commonLinkAndCompileFlagsForClang(ObjcProvider, ObjcConfiguration, AppleConfiguration)})
   *
   * <p>It's awful that we can't pass the full path to the framework and avoid framework search
   * paths, but this is imposed on us by clang. clang does not support passing the full path to the
   * framework, so Bazel cannot do it either.
   */
  protected Set<String> frameworkNames(ObjcProvider provider) {
    Set<String> names = new LinkedHashSet<>();
    Iterables.addAll(names, SdkFramework.names(provider.get(SDK_FRAMEWORK)));
    for (PathFragment frameworkDir :
        Iterables.concat(provider.get(STATIC_FRAMEWORK_DIR), provider.get(DYNAMIC_FRAMEWORK_DIR))) {
      String segment = frameworkDir.getBaseName();
      Preconditions.checkState(
          segment.endsWith(FRAMEWORK_SUFFIX),
          "expect %s to end with %s, but it does not",
          segment,
          FRAMEWORK_SUFFIX);
      names.add(segment.substring(0, segment.length() - FRAMEWORK_SUFFIX.length()));
    }
    return names;
  }
  
  /**
   * Returns libraries that should be passed to the linker.
   */
  protected ImmutableList<String> libraryNames(ObjcProvider objcProvider) {
    ImmutableList.Builder<String> args = new ImmutableList.Builder<>();
    for (String dylib : objcProvider.get(SDK_DYLIB)) {
      if (dylib.startsWith("lib")) {
        // remove lib prefix if it exists which is standard
        // for libraries (libxml.dylib -> -lxml).
        dylib = dylib.substring(3);
      }
      args.add(dylib);
    }
    return args.build();
  }
  
  /**
   * Returns libraries that should be passed into the linker with {@code -force_load}.
   */
  protected ImmutableSet<Artifact> getForceLoadArtifacts(ObjcProvider objcProvider) {
    ImmutableList<Artifact> ccLibraries = objcProvider.getCcLibraries();
    Iterable<Artifact> ccLibrariesToForceLoad =
        Iterables.filter(ccLibraries, ALWAYS_LINKED_CC_LIBRARY);

    return ImmutableSet.<Artifact>builder()
        .addAll(objcProvider.get(FORCE_LOAD_LIBRARY))
        .addAll(ccLibrariesToForceLoad)
        .build();
  }

  /** Returns pruned J2Objc archives for this target. */
  protected ImmutableList<Artifact> j2objcPrunedLibraries(ObjcProvider objcProvider) {
    ImmutableList.Builder<Artifact> j2objcPrunedLibraryBuilder = ImmutableList.builder();
    for (Artifact j2objcLibrary : objcProvider.get(ObjcProvider.J2OBJC_LIBRARY)) {
      j2objcPrunedLibraryBuilder.add(intermediateArtifacts.j2objcPrunedArchive(j2objcLibrary));
    }
    return j2objcPrunedLibraryBuilder.build();
  }
  
  /**
   * Returns true if this build should strip J2Objc dead code.
   */
  protected boolean stripJ2ObjcDeadCode(J2ObjcEntryClassProvider j2ObjcEntryClassProvider) {
    J2ObjcConfiguration j2objcConfiguration =
        buildConfiguration.getFragment(J2ObjcConfiguration.class);
    // Only perform J2ObjC dead code stripping if flag --j2objc_dead_code_removal is specified and
    // users have specified entry classes.
    return j2objcConfiguration.removeDeadCode()
        && !j2ObjcEntryClassProvider.getEntryClasses().isEmpty();
  }

  /**
   * Registers actions to perform J2Objc dead code removal.
   */
  protected void registerJ2ObjcDeadCodeRemovalActions(
      ObjcProvider objcProvider,
      J2ObjcMappingFileProvider j2ObjcMappingFileProvider,
      J2ObjcEntryClassProvider j2ObjcEntryClassProvider) {
    NestedSet<String> entryClasses = j2ObjcEntryClassProvider.getEntryClasses();
    Artifact pruner = ruleContext.getPrerequisiteArtifact("$j2objc_dead_code_pruner", Mode.HOST);
    NestedSet<Artifact> j2ObjcDependencyMappingFiles =
        j2ObjcMappingFileProvider.getDependencyMappingFiles();
    NestedSet<Artifact> j2ObjcHeaderMappingFiles =
        j2ObjcMappingFileProvider.getHeaderMappingFiles();
    NestedSet<Artifact> j2ObjcArchiveSourceMappingFiles =
        j2ObjcMappingFileProvider.getArchiveSourceMappingFiles();

    for (Artifact j2objcArchive : objcProvider.get(ObjcProvider.J2OBJC_LIBRARY)) {
      PathFragment paramFilePath =
          FileSystemUtils.replaceExtension(
              j2objcArchive.getOwner().toPathFragment(), ".param.j2objc");
      Artifact paramFile =
          ruleContext.getUniqueDirectoryArtifact(
              "_j2objc_pruned", paramFilePath,
              buildConfiguration.getBinDirectory(ruleContext.getRule().getRepository()));
      Artifact prunedJ2ObjcArchive = intermediateArtifacts.j2objcPrunedArchive(j2objcArchive);
      Artifact dummyArchive =
          Iterables.getOnlyElement(
              ruleContext
                  .getPrerequisite("$dummy_lib", Mode.TARGET, ObjcProvider.SKYLARK_CONSTRUCTOR)
                  .get(LIBRARY));

      CustomCommandLine commandLine =
          CustomCommandLine.builder()
              .addExecPath("--input_archive", j2objcArchive)
              .addExecPath("--output_archive", prunedJ2ObjcArchive)
              .addExecPath("--dummy_archive", dummyArchive)
              .addExecPath("--xcrunwrapper", xcrunwrapper(ruleContext).getExecutable())
              .addJoinedExecPaths("--dependency_mapping_files", ",", j2ObjcDependencyMappingFiles)
              .addJoinedExecPaths("--header_mapping_files", ",", j2ObjcHeaderMappingFiles)
              .addJoinedExecPaths(
                  "--archive_source_mapping_files", ",", j2ObjcArchiveSourceMappingFiles)
              .add("--entry_classes")
              .addJoined(",", entryClasses)
              .build();

      ruleContext.registerAction(
          new ParameterFileWriteAction(
              ruleContext.getActionOwner(),
              paramFile,
              commandLine,
              ParameterFile.ParameterFileType.UNQUOTED,
              ISO_8859_1));
      ruleContext.registerAction(
          ObjcRuleClasses.spawnAppleEnvActionBuilder(
                  appleConfiguration, appleConfiguration.getSingleArchPlatform())
              .setMnemonic("DummyPruner")
              .setExecutable(pruner)
              .addInput(dummyArchive)
              .addInput(pruner)
              .addInput(paramFile)
              .addInput(j2objcArchive)
              .addInput(xcrunwrapper(ruleContext).getExecutable())
              .addTransitiveInputs(j2ObjcDependencyMappingFiles)
              .addTransitiveInputs(j2ObjcHeaderMappingFiles)
              .addTransitiveInputs(j2ObjcArchiveSourceMappingFiles)
              .setCommandLine(
                  CustomCommandLine.builder().addFormatted("@%s", paramFile.getExecPath()).build())
              .addOutput(prunedJ2ObjcArchive)
              .build(ruleContext));
    }
  }
  
  /** Returns archives arising from j2objc transpilation after dead code removal. */
  protected Iterable<Artifact> computeAndStripPrunedJ2ObjcArchives(
      J2ObjcEntryClassProvider j2ObjcEntryClassProvider,
      J2ObjcMappingFileProvider j2ObjcMappingFileProvider,
      ObjcProvider objcProvider) {
    Iterable<Artifact> prunedJ2ObjcArchives = ImmutableList.<Artifact>of();
    if (stripJ2ObjcDeadCode(j2ObjcEntryClassProvider)) {
      registerJ2ObjcDeadCodeRemovalActions(
          objcProvider, j2ObjcMappingFileProvider, j2ObjcEntryClassProvider);
      prunedJ2ObjcArchives = j2objcPrunedLibraries(objcProvider);
    }
    return prunedJ2ObjcArchives;
  }

  /**
   * Returns a nested set of Bazel-built ObjC libraries with all unpruned J2ObjC libraries
   * substituted with pruned ones.
   */
  protected ImmutableList<Artifact> substituteJ2ObjcPrunedLibraries(ObjcProvider objcProvider) {
    ImmutableList.Builder<Artifact> libraries = new ImmutableList.Builder<>();

    Set<Artifact> unprunedJ2ObjcLibs = objcProvider.get(ObjcProvider.J2OBJC_LIBRARY).toSet();
    for (Artifact library : objcProvider.getObjcLibraries()) {
      // If we match an unpruned J2ObjC library, add the pruned version of the J2ObjC static library
      // instead.
      if (unprunedJ2ObjcLibs.contains(library)) {
        libraries.add(intermediateArtifacts.j2objcPrunedArchive(library));
      } else {
        libraries.add(library);
      }
    }
    return libraries.build();
  }
  
  /** Returns the artifact that should be the outcome of this build's link action */
  protected Artifact getBinaryToLink() {

    // When compilation_mode=opt and objc_enable_binary_stripping are specified, the unstripped
    // binary containing debug symbols is generated by the linker, which also needs the debug
    // symbols for dead-code removal. The binary is also used to generate dSYM bundle if
    // --apple_generate_dsym is specified. A symbol strip action is later registered to strip
    // the symbol table from the unstripped binary.
    return objcConfiguration.shouldStripBinary()
        ? intermediateArtifacts.unstrippedSingleArchitectureBinary()
        : intermediateArtifacts.strippedSingleArchitectureBinary();    
  }

  private static CommandLine symbolStripCommandLine(
      ImmutableList<String> extraFlags, Artifact unstrippedArtifact, Artifact strippedArtifact) {
    return CustomCommandLine.builder()
        .add(STRIP)
        .addAll(extraFlags)
        .addExecPath("-o", strippedArtifact)
        .addPath(unstrippedArtifact.getExecPath())
        .build();
  }

  /** Signals if stripping should include options for dynamic libraries. */
  protected enum StrippingType {
    DEFAULT, DYNAMIC_LIB
  }

  /**
   * Registers an action that uses the 'strip' tool to perform binary stripping on the given binary
   * subject to the given {@link StrippingType}.
   */
  protected void registerBinaryStripAction(Artifact binaryToLink, StrippingType strippingType) {
    final ImmutableList<String> stripArgs;
    if (isTestRule) {
      // For test targets, only debug symbols are stripped off, since /usr/bin/strip is not able
      // to strip off all symbols in XCTest bundle.
      stripArgs = ImmutableList.of("-S");
    } else if (strippingType == StrippingType.DYNAMIC_LIB) {
      // For dynamic libs must pass "-x" to strip only local symbols.
      stripArgs = ImmutableList.of("-x");
    } else {
      stripArgs = ImmutableList.<String>of();
    }

    Artifact strippedBinary = intermediateArtifacts.strippedSingleArchitectureBinary();

    ruleContext.registerAction(
        ObjcRuleClasses.spawnAppleEnvActionBuilder(
                appleConfiguration, appleConfiguration.getSingleArchPlatform())
            .setMnemonic("ObjcBinarySymbolStrip")
            .setExecutable(xcrunwrapper(ruleContext))
            .setCommandLine(symbolStripCommandLine(stripArgs, binaryToLink, strippedBinary))
            .addOutput(strippedBinary)
            .addInput(binaryToLink)
            .build(ruleContext));
  }

  private NestedSet<Artifact> getGcovForObjectiveCIfNeeded() {
    if (ruleContext.getConfiguration().isCodeCoverageEnabled()
        && ruleContext.attributes().has(IosTest.OBJC_GCOV_ATTR, BuildType.LABEL)) {
      return PrerequisiteArtifacts.nestedSet(ruleContext, IosTest.OBJC_GCOV_ATTR, Mode.HOST);
    } else {
      return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    }
  }

  CompilationSupport registerGenerateUmbrellaHeaderAction(
      Artifact umbrellaHeader, Iterable<Artifact> publicHeaders) {
     ruleContext.registerAction(
        new UmbrellaHeaderAction(
            ruleContext.getActionOwner(),
            umbrellaHeader,
            publicHeaders,
            ImmutableList.<PathFragment>of()));
 
    return this;
  }

  protected Optional<Artifact> getPchFile() {
    if (!usePch) {
      return Optional.absent();
    }
    Artifact pchHdr = null;
    if (ruleContext.attributes().has("pch", BuildType.LABEL)) {
      pchHdr = ruleContext.getPrerequisiteArtifact("pch", Mode.TARGET);
    }
    return Optional.fromNullable(pchHdr);
  }

  /**
   * Registers an action that will generate a clang module map.
   * @param moduleMap the module map to generate
   * @param publicHeaders the headers that should be directly accessible by dependers
   * @return this compilation support
   */
  public CompilationSupport registerGenerateModuleMapAction(
      CppModuleMap moduleMap, Iterable<Artifact> publicHeaders) {
    publicHeaders = Iterables.filter(publicHeaders, CppFileTypes.MODULE_MAP_HEADER);
    ruleContext.registerAction(
        new CppModuleMapAction(
            ruleContext.getActionOwner(),
            moduleMap,
            ImmutableList.<Artifact>of(),
            publicHeaders,
            attributes.moduleMapsForDirectDeps(),
            ImmutableList.<PathFragment>of(),
            /*compiledModule=*/ true,
            /*moduleMapHomeIsCwd=*/ false,
            /*generateSubModules=*/ false,
            /*externDependencies=*/ true));

    return this;
  }
    
  /**
   * Collector that, given a list of output artifacts, finds and registers coverage notes metadata
   * for any compilation action.
   */
  private static class ObjcCoverageMetadataCollector extends LocalMetadataCollector {

    @Override
    public void collectMetadataArtifacts(
        Iterable<Artifact> artifacts,
        AnalysisEnvironment analysisEnvironment,
        NestedSetBuilder<Artifact> metadataFilesBuilder) {
      for (Artifact artifact : artifacts) {
        ActionAnalysisMetadata action = analysisEnvironment.getLocalGeneratingAction(artifact);
        if (action.getMnemonic().equals("ObjcCompile")) {
          addOutputs(metadataFilesBuilder, action, ObjcRuleClasses.COVERAGE_NOTES);
        }
      }
    }
  }

  private static Iterable<PathFragment> uniqueParentDirectories(Iterable<PathFragment> paths) {
    ImmutableSet.Builder<PathFragment> parents = new ImmutableSet.Builder<>();
    for (PathFragment path : paths) {
      parents.add(path.getParentDirectory());
    }
    return parents.build();
  }

  /** Holds information about Objective-C compile actions that require header thinning. */
  protected static final class ObjcHeaderThinningInfo {
    /** Source file for compile action. */
    public final Artifact sourceFile;
    /** headers_list file for compile action. */
    public final Artifact headersListFile;
    /** Command line arguments for compile action execution. */
    public final ImmutableList<String> arguments;

    public ObjcHeaderThinningInfo(
        Artifact sourceFile, Artifact headersListFile, ImmutableList<String> arguments) {
      this.sourceFile = Preconditions.checkNotNull(sourceFile);
      this.headersListFile = Preconditions.checkNotNull(headersListFile);
      this.arguments = Preconditions.checkNotNull(arguments);
    }

    public ObjcHeaderThinningInfo(
        Artifact sourceFile, Artifact headersListFile, Iterable<String> arguments) {
      this(sourceFile, headersListFile, ImmutableList.copyOf(arguments));
    }
  }

  /**
   * Returns true when ObjC header thinning is enabled via configuration and an a valid
   * header_scanner executable target is provided.
   */
  protected boolean isHeaderThinningEnabled() {
    if (objcConfiguration.useExperimentalHeaderThinning()
        && ruleContext.isAttrDefined(ObjcRuleClasses.HEADER_SCANNER_ATTRIBUTE, BuildType.LABEL)) {
      FilesToRunProvider tool = getHeaderThinningToolExecutable();
      // Additional here to ensure that an Executable Artifact exists to disable where the tool
      // is an empty filegroup
      return tool != null && tool.getExecutable() != null;
    }
    return false;
  }

  protected FilesToRunProvider getHeaderThinningToolExecutable() {
    return ruleContext
        .getPrerequisite(ObjcRuleClasses.HEADER_SCANNER_ATTRIBUTE, Mode.HOST)
        .getProvider(FilesToRunProvider.class);
  }

  /**
   * Creates and registers ObjcHeaderScanning {@link SpawnAction}. Groups all the actions by their
   * compilation command line arguments and creates a ObjcHeaderScanning action for each unique one.
   *
   * <p>The number of sources to scan per actions are bounded so that targets with a high number of
   * sources are not penalized. A large number of sources may require a lot of processing
   * particularly when the headers required for different sources vary greatly and the caching
   * mechanism in the tool is largely useless. In these instances these actions would benefit by
   * being distributed so they don't contribute to the critical path. The partition size is
   * configurable so that it can be tuned.
   */
  protected void registerHeaderScanningActions(
      ImmutableList<ObjcHeaderThinningInfo> headerThinningInfo,
      ObjcProvider objcProvider,
      CompilationArtifacts compilationArtifacts) {
    if (headerThinningInfo.isEmpty()) {
      return;
    }

    ListMultimap<ImmutableList<String>, ObjcHeaderThinningInfo>
        objcHeaderThinningInfoByCommandLine = groupActionsByCommandLine(headerThinningInfo);
    // Register a header scanning spawn action for each unique set of command line arguments
    for (ImmutableList<String> args : objcHeaderThinningInfoByCommandLine.keySet()) {
      // As infos is in insertion order we should reliably get the same sublists below
      for (List<ObjcHeaderThinningInfo> partition :
          Lists.partition(
              objcHeaderThinningInfoByCommandLine.get(args),
              objcConfiguration.objcHeaderThinningPartitionSize())) {
        registerHeaderScanningAction(objcProvider, compilationArtifacts, args, partition);
      }
    }
  }

  private void registerHeaderScanningAction(
      ObjcProvider objcProvider,
      CompilationArtifacts compilationArtifacts,
      ImmutableList<String> args,
      List<ObjcHeaderThinningInfo> infos) {
    SpawnAction.Builder builder =
        new SpawnAction.Builder()
            .setMnemonic("ObjcHeaderScanning")
            .setExecutable(getHeaderThinningToolExecutable())
            .addInputs(
                ruleContext
                    .getPrerequisiteArtifacts(ObjcRuleClasses.APPLE_SDK_ATTRIBUTE, Mode.TARGET)
                    .list());
    CustomCommandLine.Builder cmdLine =
        CustomCommandLine.builder()
            .add("--arch", appleConfiguration.getSingleArchitecture().toLowerCase())
            .add("--platform", appleConfiguration.getSingleArchPlatform().getLowerCaseNameInPlist())
            .add(
                "--sdk_version",
                XcodeConfig.getSdkVersionForPlatform(
                        ruleContext, appleConfiguration.getSingleArchPlatform())
                    .toStringWithMinimumComponents(2))
            .add(
                "--xcode_version",
                XcodeConfig.getXcodeVersion(ruleContext).toStringWithMinimumComponents(2))
            .add("--");
    for (ObjcHeaderThinningInfo info : infos) {
      cmdLine.addFormatted(
          "%s:%s", info.sourceFile.getExecPath(), info.headersListFile.getExecPath());
      builder.addInput(info.sourceFile).addOutput(info.headersListFile);
    }
    ruleContext.registerAction(
        builder
            .setCommandLine(cmdLine.add("--").addAll(args).build())
            .addInputs(compilationArtifacts.getPrivateHdrs())
            .addTransitiveInputs(attributes.hdrs())
            .addTransitiveInputs(objcProvider.get(ObjcProvider.HEADER))
            .addInputs(getPchFile().asSet())
            .addTransitiveInputs(objcProvider.get(ObjcProvider.STATIC_FRAMEWORK_FILE))
            .addTransitiveInputs(objcProvider.get(ObjcProvider.DYNAMIC_FRAMEWORK_FILE))
            .build(ruleContext));
  }

  /**
   * Groups {@link ObjcHeaderThinningInfo} objects based on the command line arguments of the
   * ObjcCompile action.
   *
   * <p>Grouping by command line arguments allows {@link
   * #registerHeaderScanningActions(ImmutableList, ObjcProvider, CompilationArtifacts)} to create a
   * {@link SpawnAction} based on the compiler command line flags that may cause a difference in
   * behaviour by the preprocessor. Some of the command line arguments must be filtered out as they
   * change with every source {@link Artifact}; for example the object file (-o) and dotd filenames
   * (-MF). These arguments are known not to change the preprocessor behaviour.
   *
   * @param headerThinningInfos information for compile actions that require header thinning
   * @return values in {@code headerThinningInfos} grouped by compile action command line arguments
   */
  private static ListMultimap<ImmutableList<String>, ObjcHeaderThinningInfo>
      groupActionsByCommandLine(ImmutableList<ObjcHeaderThinningInfo> headerThinningInfos) {
    // Maintain insertion order so that iteration in #registerHeaderScanningActions is deterministic
    ListMultimap<ImmutableList<String>, ObjcHeaderThinningInfo>
        objcHeaderThinningInfoByCommandLine = ArrayListMultimap.create();
    for (ObjcHeaderThinningInfo info : headerThinningInfos) {
      ImmutableList.Builder<String> filteredArgumentsBuilder = ImmutableList.builder();
      List<String> arguments = info.arguments;
      for (int i = 0; i < arguments.size(); ++i) {
        String arg = arguments.get(i);
        if (arg.equals("-MF") || arg.equals("-o") || arg.equals("-c")) {
          ++i;
        } else if (!arg.equals("-MD")) {
          filteredArgumentsBuilder.add(arg);
        }
      }
      objcHeaderThinningInfoByCommandLine.put(filteredArgumentsBuilder.build(), info);
    }
    return objcHeaderThinningInfoByCommandLine;
  }

  @Nullable
  private FdoSupportProvider maybeGetFdoSupport() {
    // TODO(rduan): Remove this check once all rules are using the crosstool support.
    if (ruleContext
        .attributes()
        .has(CcToolchain.CC_TOOLCHAIN_DEFAULT_ATTRIBUTE_NAME, BuildType.LABEL)) {
      return CppHelper.getFdoSupportUsingDefaultCcToolchainAttribute(ruleContext);
    } else {
      return null;
    }
  }

  public static Optional<Artifact> getCustomModuleMap(RuleContext ruleContext) {
    if (ruleContext.attributes().has("module_map", BuildType.LABEL)) {
      return Optional.fromNullable(ruleContext.getPrerequisiteArtifact("module_map", Mode.TARGET));
    }
    return Optional.absent();
  }
}
