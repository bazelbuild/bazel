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
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.FORCE_LOAD_LIBRARY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.FRAMEWORK_DIR;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.FRAMEWORK_SEARCH_PATH_ONLY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.IMPORTED_LIBRARY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.INCLUDE_SYSTEM;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.LIBRARY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.SDK_DYLIB;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.SDK_FRAMEWORK;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.COMPILABLE_SRCS_TYPE;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.HEADERS;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.NON_ARC_SRCS_TYPE;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.PRECOMPILED_SRCS_TYPE;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.SRCS_TYPE;
import static java.nio.charset.StandardCharsets.ISO_8859_1;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
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
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.ParameterFileWriteAction;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction.SafeImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.apple.AppleToolchain;
import com.google.devtools.build.lib.rules.apple.Platform;
import com.google.devtools.build.lib.rules.apple.Platform.PlatformType;
import com.google.devtools.build.lib.rules.cpp.CppModuleMap;
import com.google.devtools.build.lib.rules.cpp.CppModuleMapAction;
import com.google.devtools.build.lib.rules.objc.XcodeProvider.Builder;
import com.google.devtools.build.lib.rules.test.InstrumentedFilesCollector;
import com.google.devtools.build.lib.rules.test.InstrumentedFilesCollector.InstrumentationSpec;
import com.google.devtools.build.lib.rules.test.InstrumentedFilesCollector.LocalMetadataCollector;
import com.google.devtools.build.lib.rules.test.InstrumentedFilesProvider;
import com.google.devtools.build.lib.util.FileTypeSet;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

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
      new Predicate<Artifact>() {
        @Override
        public boolean apply(Artifact input) {
          return LINK_LIBRARY_FILETYPES.matches(input.getFilename());
        }
      };
  
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
  
  private static final Predicate<String> INCLUDE_DIR_OPTION_IN_COPTS =
      new Predicate<String>() {
        @Override
        public boolean apply(String copt) {
          return copt.startsWith("-I") && copt.length() > 2;
        }
      };

  /** Predicate that matches all artifacts that can be used in a Clang module map. */
  private static final Predicate<Artifact> MODULE_MAP_HEADER =
      new Predicate<Artifact>() {
        @Override
        public boolean apply(Artifact artifact) {
          if (artifact.isTreeArtifact()) {
            // Tree artifact is basically a directory, which does not have any information about
            // the contained files and their extensions. Here we assume the passed in tree artifact
            // contains proper header files with .h extension.
            return true;
          } else {
            // The current clang (clang-600.0.57) on Darwin doesn't support 'textual', so we can't
            // have '.inc' files in the module map (since they're implictly textual).
            // TODO(bazel-team): Use HEADERS file type once clang-700 is the base clang we support.
            return artifact.getFilename().endsWith(".h");
          }
        }
      };

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
        .setPchFile(Optional.fromNullable(ruleContext.getPrerequisiteArtifact("pch", Mode.TARGET)))
        .build();
  }

  /** Returns a list of framework search path flags for clang actions. */
  static Iterable<String> commonFrameworkFlags(
      ObjcProvider provider, AppleConfiguration appleConfiguration) {
    return Interspersing.beforeEach("-F", commonFrameworkNames(provider, appleConfiguration));
  }

  /** Returns a list of frameworks for clang actions. */
  static Iterable<String> commonFrameworkNames(
      ObjcProvider provider, AppleConfiguration appleConfiguration) {
    Platform platform = appleConfiguration.getSingleArchPlatform();

    ImmutableList.Builder<String> frameworkNames =
        new ImmutableList.Builder<String>()
            .add(AppleToolchain.sdkFrameworkDir(platform, appleConfiguration));
    if (platform.getType() == PlatformType.IOS) {
      // As of sdk8.1, XCTest is in a base Framework dir
      frameworkNames.add(AppleToolchain.platformDeveloperFrameworkDir(appleConfiguration));
    }
    return frameworkNames
        // Add custom (non-SDK) framework search paths. For each framework foo/bar.framework,
        // include "foo" as a search path.
        .addAll(PathFragment.safePathStrings(uniqueParentDirectories(provider.get(FRAMEWORK_DIR))))
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
   * artifacts with different output prefixes, multiple archives can be compiled for the same
   * rule context.
   */
  public CompilationSupport(
      RuleContext ruleContext,
      BuildConfiguration buildConfiguration,
      IntermediateArtifacts intermediateArtifacts,
      CompilationAttributes compilationAttributes) {
    this.ruleContext = ruleContext;
    this.buildConfiguration = buildConfiguration;
    this.objcConfiguration = buildConfiguration.getFragment(ObjcConfiguration.class);
    this.appleConfiguration = buildConfiguration.getFragment(AppleConfiguration.class);
    this.attributes = compilationAttributes;
    this.intermediateArtifacts = intermediateArtifacts;
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
    ImmutableList<Artifact> inputArtifacts = ImmutableList.<Artifact>builder()
        .addAll(objcProvider.getObjcLibraries())
        .addAll(objcProvider.get(IMPORTED_LIBRARY))
        .addAll(objcProvider.getCcLibraries()).build();
    return registerFullyLinkAction(objcProvider, inputArtifacts, outputArchive);
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
    return registerFullyLinkAction(objcProvider, inputArtifacts, outputArchive);
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
        !TargetUtils.isTestRule(ruleContext.getTarget()));
  }

  /**
   * Registers an action that will generate a clang module map for this target, using the hdrs
   * attribute of this rule.
   */
  CompilationSupport registerGenerateModuleMapAction(
      Optional<CompilationArtifacts> compilationArtifacts) {
    // TODO(bazel-team): Include textual headers in the module map when Xcode 6 support is
    // dropped.
    Iterable<Artifact> publicHeaders = attributes.hdrs();
    Iterable<Artifact> privateHeaders = ImmutableList.of();
    if (compilationArtifacts.isPresent()) {
      CompilationArtifacts artifacts = compilationArtifacts.get();
      publicHeaders = Iterables.concat(publicHeaders, artifacts.getAdditionalHdrs());
      privateHeaders = Iterables.concat(privateHeaders, artifacts.getPrivateHdrs());
    }
    CppModuleMap moduleMap = intermediateArtifacts.moduleMap();
    registerGenerateModuleMapAction(moduleMap, publicHeaders, privateHeaders);

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
        Iterables.filter(attributes.includes(), PathFragment.IS_ABSOLUTE)) {
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
   * Sets compilation-related Xcode project information on the given provider builder.
   *
   * @param common common information about this rule's attributes and its dependencies
   * @return this compilation support
   */
  CompilationSupport addXcodeSettings(Builder xcodeProviderBuilder, ObjcCommon common) {
    for (CompilationArtifacts artifacts : common.getCompilationArtifacts().asSet()) {
      xcodeProviderBuilder.setCompilationArtifacts(artifacts);
    }

    // The include directory options ("-I") are parsed out of copts. The include directories are
    // added as non-propagated header search paths local to the associated Xcode target.
    Iterable<String> copts = Iterables.concat(objcConfiguration.getCopts(), attributes.copts());
    Iterable<String> includeDirOptions = Iterables.filter(copts, INCLUDE_DIR_OPTION_IN_COPTS);
    Iterable<String> coptsWithoutIncludeDirs = Iterables.filter(
        copts, Predicates.not(INCLUDE_DIR_OPTION_IN_COPTS));
    ImmutableList.Builder<PathFragment> nonPropagatedHeaderSearchPaths =
        new ImmutableList.Builder<>();
    for (String includeDirOption : includeDirOptions) {
      nonPropagatedHeaderSearchPaths.add(new PathFragment(includeDirOption.substring(2)));
    }

    // We also need to add the -isystem directories from the CC header providers. ObjCommon
    // adds these to the objcProvider, so let's just get them from there.
    Iterable<PathFragment> includeSystemPaths = common.getObjcProvider().get(INCLUDE_SYSTEM);

    xcodeProviderBuilder
        .addHeaders(attributes.hdrs())
        .addHeaders(attributes.textualHdrs())
        .addUserHeaderSearchPaths(ObjcCommon.userHeaderSearchPaths(buildConfiguration))
        .addHeaderSearchPaths("$(WORKSPACE_ROOT)",
            attributes.headerSearchPaths(buildConfiguration.getGenfilesFragment()))
        .addHeaderSearchPaths("$(WORKSPACE_ROOT)", includeSystemPaths)
        .addHeaderSearchPaths("$(SDKROOT)/usr/include", attributes.sdkIncludes())
        .addNonPropagatedHeaderSearchPaths(
            "$(WORKSPACE_ROOT)", nonPropagatedHeaderSearchPaths.build())
        .addCompilationModeCopts(objcConfiguration.getCoptsForCompilationMode())
        .addCopts(coptsWithoutIncludeDirs);

    return this;
  }
 
  /**
   * Registers all actions necessary to compile this rule's sources and archive them.
   *
   * @param common common information about this rule and its dependencies
   * @param extraCompileArgs args to be added to compile actions
   * @param priorityHeaders priority headers to be included before the dependency headers
   * @return this compilation support
   * @throws RuleErrorException for invalid crosstool files
   */
  abstract CompilationSupport registerCompileAndArchiveActions(
      ObjcCommon common, ExtraCompileArgs extraCompileArgs, Iterable<PathFragment> priorityHeaders)
      throws RuleErrorException, InterruptedException;

  /**
   * Registers any actions necessary to link this rule and its dependencies.
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
  abstract CompilationSupport registerLinkActions(
      ObjcProvider objcProvider,
      J2ObjcMappingFileProvider j2ObjcMappingFileProvider,
      J2ObjcEntryClassProvider j2ObjcEntryClassProvider,
      ExtraLinkArgs extraLinkArgs,
      Iterable<Artifact> extraLinkInputs,
      DsymOutputType dsymOutputType) throws InterruptedException;

  /**
   * Returns the copts for the compile action in the current rule context (using a combination of
   * the rule's "copts" attribute as well as the current configuration copts).
   */
  protected Iterable<String> getCompileRuleCopts() {
    List<String> copts =
        Lists.newArrayList(Iterables.concat(objcConfiguration.getCopts(), attributes.copts()));

    for (String copt : copts) {
      if (copt.contains("-fmodules-cache-path")) {
        // Bazel decides on the cache path location.
        ruleContext.ruleWarning(MODULES_CACHE_PATH_WARNING);
      }
    }

    if (attributes.enableModules()) {
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
   * @return this {@link CompilationSupport} instance
   */
  protected abstract CompilationSupport registerFullyLinkAction(
      ObjcProvider objcProvider, Iterable<Artifact> inputArtifacts, Artifact outputArchive)
      throws InterruptedException;

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
    for (PathFragment frameworkDir : provider.get(FRAMEWORK_DIR)) {
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
              "_j2objc_pruned", paramFilePath, ruleContext.getBinOrGenfilesDirectory());
      Artifact prunedJ2ObjcArchive = intermediateArtifacts.j2objcPrunedArchive(j2objcArchive);
      Artifact dummyArchive =
          Iterables.getOnlyElement(
              ruleContext
                  .getPrerequisite("$dummy_lib", Mode.TARGET, ObjcProvider.class)
                  .get(LIBRARY));

      CustomCommandLine commandLine =
          CustomCommandLine.builder()
              .addExecPath("--input_archive", j2objcArchive)
              .addExecPath("--output_archive", prunedJ2ObjcArchive)
              .addExecPath("--dummy_archive", dummyArchive)
              .addExecPath("--xcrunwrapper", xcrunwrapper(ruleContext).getExecutable())
              .addJoinExecPaths("--dependency_mapping_files", ",", j2ObjcDependencyMappingFiles)
              .addJoinExecPaths("--header_mapping_files", ",", j2ObjcHeaderMappingFiles)
              .addJoinExecPaths(
                  "--archive_source_mapping_files", ",", j2ObjcArchiveSourceMappingFiles)
              .add("--entry_classes")
              .add(Joiner.on(",").join(entryClasses))
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
                  CustomCommandLine.builder().addPaths("@%s", paramFile.getExecPath()).build())
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

  private NestedSet<Artifact> getGcovForObjectiveCIfNeeded() {
    if (ruleContext.getConfiguration().isCodeCoverageEnabled()
        && ruleContext.attributes().has(IosTest.OBJC_GCOV_ATTR, BuildType.LABEL)) {
      return PrerequisiteArtifacts.nestedSet(ruleContext, IosTest.OBJC_GCOV_ATTR, Mode.HOST);
    } else {
      return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    }
  }

  /**
   * Registers an action that will generate a clang module map.
   *
   * @param moduleMap the module map to generate
   * @param publicHeaders the headers that should be directly accessible by dependers
   * @param privateHeaders the headers that should only be directly accessible by this module
   */
  private void registerGenerateModuleMapAction(
      CppModuleMap moduleMap, Iterable<Artifact> publicHeaders, Iterable<Artifact> privateHeaders) {
    publicHeaders = Iterables.filter(publicHeaders, MODULE_MAP_HEADER);
    privateHeaders = Iterables.filter(privateHeaders, MODULE_MAP_HEADER);
    ruleContext.registerAction(
        new CppModuleMapAction(
            ruleContext.getActionOwner(),
            moduleMap,
            privateHeaders,
            publicHeaders,
            attributes.moduleMapsForDirectDeps(),
            ImmutableList.<PathFragment>of(),
            /*compiledModule=*/ true,
            /*moduleMapHomeIsCwd=*/ false,
            /*generateSubModules=*/ false,
            /*externDependencies=*/ true));
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
}
