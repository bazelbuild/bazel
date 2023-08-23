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

import static com.google.common.base.Preconditions.checkState;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.STRIP;
import static java.nio.charset.StandardCharsets.ISO_8859_1;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Optional;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandLine;
import com.google.devtools.build.lib.actions.ParamFileInfo;
import com.google.devtools.build.lib.actions.ParameterFile;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine.VectorArg;
import com.google.devtools.build.lib.analysis.actions.ParameterFileWriteAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.StarlarkInfo;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.apple.XcodeConfigInfo;
import com.google.devtools.build.lib.rules.cpp.CcCommon;
import com.google.devtools.build.lib.rules.cpp.CcCommon.Language;
import com.google.devtools.build.lib.rules.cpp.CcCompilationContext;
import com.google.devtools.build.lib.rules.cpp.CcCompilationOutputs;
import com.google.devtools.build.lib.rules.cpp.CcInfo;
import com.google.devtools.build.lib.rules.cpp.CcLinkingContext;
import com.google.devtools.build.lib.rules.cpp.CcLinkingContext.LinkOptions;
import com.google.devtools.build.lib.rules.cpp.CcLinkingContext.LinkerInput;
import com.google.devtools.build.lib.rules.cpp.CcLinkingHelper;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CcToolchainProvider;
import com.google.devtools.build.lib.rules.cpp.CcToolchainRule;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppHelper;
import com.google.devtools.build.lib.rules.cpp.CppLinkAction;
import com.google.devtools.build.lib.rules.cpp.CppLinkActionBuilder;
import com.google.devtools.build.lib.rules.cpp.CppSemantics;
import com.google.devtools.build.lib.rules.cpp.LibraryToLink;
import com.google.devtools.build.lib.rules.cpp.Link.LinkTargetType;
import com.google.devtools.build.lib.rules.cpp.Link.LinkingMode;
import com.google.devtools.build.lib.rules.cpp.UserVariablesExtension;
import com.google.devtools.build.lib.rules.objc.ObjcVariablesExtension.VariableCategory;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;
import javax.annotation.Nullable;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.StarlarkValue;

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
public class CompilationSupport implements StarlarkValue {

  @VisibleForTesting static final String OBJC_MODULE_CACHE_DIR_NAME = "_objc_module_cache";

  @VisibleForTesting
  static final String ABSOLUTE_INCLUDES_PATH_FORMAT =
      "The path '%s' is absolute, but only relative paths are allowed.";

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

  /**
   * Frameworks implicitly linked to iOS, visionOS, watchOS, and tvOS binaries when using legacy
   * compilation.
   */
  @VisibleForTesting
  static final NestedSet<String> AUTOMATIC_SDK_FRAMEWORKS =
      NestedSetBuilder.create(Order.STABLE_ORDER, "Foundation", "UIKit");

  /** Iterable wrapper providing strong type safety for arguments to binary linking. */
  static final class ExtraLinkArgs extends IterableWrapper<String> {
    ExtraLinkArgs(String... args) {
      super(args);
    }

    ExtraLinkArgs(Iterable<String> args) {
      super(args);
    }
  }

  /** Iterable wrapper providing strong type safety for extra compile flags. */
  static final class ExtraCompileArgs extends IterableWrapper<String> {
    static final ExtraCompileArgs NONE = new ExtraCompileArgs();

    ExtraCompileArgs(String... args) {
      super(args);
    }
  }

  @VisibleForTesting
  static final String FILE_IN_SRCS_AND_HDRS_WARNING_FORMAT = "File '%s' is in both srcs and hdrs.";

  @VisibleForTesting
  static final String FILE_IN_SRCS_AND_NON_ARC_SRCS_ERROR_FORMAT =
      "File '%s' is present in both srcs and non_arc_srcs which is forbidden.";

  @VisibleForTesting
  static final String BOTH_MODULE_NAME_AND_MODULE_MAP_SPECIFIED =
      "Specifying both module_name and module_map is invalid, please remove one of them.";

  static final ImmutableList<String> DEFAULT_COMPILER_FLAGS = ImmutableList.of("-DOS_IOS");

  /** Returns information about the given rule's compilation artifacts. */
  // TODO(bazel-team): Remove this information from ObjcCommon and move it internal to this class.
  static CompilationArtifacts compilationArtifacts(RuleContext ruleContext) {
    return compilationArtifacts(ruleContext, new IntermediateArtifacts(ruleContext));
  }

  /**
   * Returns information about the given rule's compilation artifacts. Dependencies specified in the
   * current rule's attributes are obtained via {@code ruleContext}. Output locations are determined
   * using the given {@code intermediateArtifacts} object. The fact that these are distinct objects
   * allows the caller to generate compilation actions pertaining to a configuration separate from
   * the current rule's configuration.
   */
  static CompilationArtifacts compilationArtifacts(
      RuleContext ruleContext, IntermediateArtifacts intermediateArtifacts) {
    return new CompilationArtifacts(ruleContext, intermediateArtifacts);
  }

  private final RuleContext ruleContext;
  private final BuildConfigurationValue buildConfiguration;
  private final AppleConfiguration appleConfiguration;
  private final CppSemantics cppSemantics;
  private final CompilationAttributes attributes;
  private final IntermediateArtifacts intermediateArtifacts;
  private final CcToolchainProvider toolchain;
  private Optional<CcCompilationContext> ccCompilationContext;

  @StarlarkMethod(name = "compilation_context", documented = false, structField = true)
  public CcCompilationContext getCcCompilationContext() {
    checkState(ccCompilationContext.isPresent());
    return ccCompilationContext.get();
  }

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
  private CompilationSupport(
      RuleContext ruleContext,
      BuildConfigurationValue buildConfiguration,
      CppSemantics cppSemantics,
      IntermediateArtifacts intermediateArtifacts,
      CompilationAttributes compilationAttributes,
      CcToolchainProvider toolchain)
      throws RuleErrorException {
    this.ruleContext = ruleContext;
    this.buildConfiguration = buildConfiguration;
    this.appleConfiguration = buildConfiguration.getFragment(AppleConfiguration.class);
    this.cppSemantics = cppSemantics;
    this.attributes = compilationAttributes;
    this.intermediateArtifacts = intermediateArtifacts;
    this.ccCompilationContext = Optional.absent();
    if (toolchain == null
        && (ruleContext
                .attributes()
                .has(CcToolchainRule.CC_TOOLCHAIN_DEFAULT_ATTRIBUTE_NAME, BuildType.LABEL)
            || ruleContext
                .attributes()
                .has(
                    CcToolchainRule.CC_TOOLCHAIN_DEFAULT_ATTRIBUTE_NAME_FOR_STARLARK,
                    BuildType.LABEL))) {
      toolchain = CppHelper.getToolchainUsingDefaultCcToolchainAttribute(ruleContext);
    }

    this.toolchain = toolchain;
  }

  /** Builder for {@link CompilationSupport} */
  public static class Builder {
    private final RuleContext ruleContext;
    private final CppSemantics cppSemantics;
    private BuildConfigurationValue buildConfiguration;
    private IntermediateArtifacts intermediateArtifacts;
    private CompilationAttributes compilationAttributes;
    private CcToolchainProvider toolchain;

    public Builder(RuleContext ruleContext, CppSemantics cppSemantics) {
      this.ruleContext = ruleContext;
      this.cppSemantics = cppSemantics;
    }

    /** Sets the {@link BuildConfigurationValue} for the calling target. */
    @CanIgnoreReturnValue
    public Builder setConfig(BuildConfigurationValue buildConfiguration) {
      this.buildConfiguration = buildConfiguration;
      return this;
    }

    /** Sets {@link IntermediateArtifacts} for deriving artifact paths. */
    @CanIgnoreReturnValue
    public Builder setIntermediateArtifacts(IntermediateArtifacts intermediateArtifacts) {
      this.intermediateArtifacts = intermediateArtifacts;
      return this;
    }

    /** Sets {@link CompilationAttributes} for the calling target. */
    @CanIgnoreReturnValue
    public Builder setCompilationAttributes(CompilationAttributes compilationAttributes) {
      this.compilationAttributes = compilationAttributes;
      return this;
    }

    /**
     * Sets {@link CcToolchainProvider} for the calling target.
     *
     * <p>This is needed if it can't correctly be inferred directly from the rule context. Setting
     * to null causes the default to be used as if this was never called.
     */
    @CanIgnoreReturnValue
    public Builder setToolchainProvider(CcToolchainProvider toolchain) {
      this.toolchain = toolchain;
      return this;
    }

    /** Returns a {@link CompilationSupport} instance. */
    public CompilationSupport build() throws InterruptedException, RuleErrorException {
      if (buildConfiguration == null) {
        buildConfiguration = ruleContext.getConfiguration();
      }

      if (intermediateArtifacts == null) {
        intermediateArtifacts = new IntermediateArtifacts(ruleContext, buildConfiguration);
      }

      if (compilationAttributes == null) {
        compilationAttributes = CompilationAttributes.Builder.fromRuleContext(ruleContext).build();
      }

      return new CompilationSupport(
          ruleContext,
          buildConfiguration,
          cppSemantics,
          intermediateArtifacts,
          compilationAttributes,
          toolchain);
    }
  }

  /**
   * Validates compilation-related attributes on this rule.
   *
   * @return this compilation support
   * @throws RuleErrorException if there are attribute errors
   */
  @CanIgnoreReturnValue
  CompilationSupport validateAttributes() throws RuleErrorException {
    for (PathFragment absoluteInclude :
        Iterables.filter(attributes.includes().toList(), PathFragment::isAbsolute)) {
      ruleContext.attributeError(
          "includes", String.format(ABSOLUTE_INCLUDES_PATH_FORMAT, absoluteInclude));
    }

    if (ruleContext.attributes().has("srcs", BuildType.LABEL_LIST)) {
      ImmutableSet<Artifact> hdrsSet = attributes.hdrs().toSet();
      ImmutableSet<Artifact> srcsSet =
          ImmutableSet.copyOf(ruleContext.getPrerequisiteArtifacts("srcs").list());

      // Check for overlap between srcs and hdrs.
      for (Artifact header : Sets.intersection(hdrsSet, srcsSet)) {
        String path = header.getRootRelativePath().toString();
        ruleContext.attributeWarning(
            "srcs", String.format(FILE_IN_SRCS_AND_HDRS_WARNING_FORMAT, path));
      }

      // Check for overlap between srcs and non_arc_srcs.
      ImmutableSet<Artifact> nonArcSrcsSet =
          ImmutableSet.copyOf(ruleContext.getPrerequisiteArtifacts("non_arc_srcs").list());
      for (Artifact conflict : Sets.intersection(nonArcSrcsSet, srcsSet)) {
        String path = conflict.getRootRelativePath().toString();
        ruleContext.attributeError(
            "srcs", String.format(FILE_IN_SRCS_AND_NON_ARC_SRCS_ERROR_FORMAT, path));
      }
    }

    if (ruleContext.attributes().isAttributeValueExplicitlySpecified("module_name")
        && ruleContext.attributes().isAttributeValueExplicitlySpecified("module_map")) {
      ruleContext.attributeError("module_name", BOTH_MODULE_NAME_AND_MODULE_MAP_SPECIFIED);
    }

    ruleContext.assertNoErrors();
    return this;
  }

  private StrippingType getStrippingType(
      ExtraLinkArgs extraLinkArgs, FeatureConfiguration featureConfiguration) {
    if (Iterables.contains(extraLinkArgs, "-dynamiclib")
        || featureConfiguration.isEnabled(ObjcRuleClasses.LINK_DYLIB_FEATURE)) {
      return StrippingType.DYNAMIC_LIB;
    }
    if (Iterables.contains(extraLinkArgs, "-bundle")
        || featureConfiguration.isEnabled(ObjcRuleClasses.LINK_BUNDLE_FEATURE)) {
      return StrippingType.LOADABLE_BUNDLE;
    }
    if (Iterables.contains(extraLinkArgs, "-kext")) {
      return StrippingType.KERNEL_EXTENSION;
    }
    return StrippingType.DEFAULT;
  }

  /**
   * Returns the preferred static library for linking, or {@code null} if there is no static
   * library.
   *
   * @param library the input library.
   */
  @Nullable
  public static Artifact getStaticLibraryForLinking(LibraryToLink library) {
    if (library.getStaticLibrary() != null) {
      return library.getStaticLibrary();
    } else if (library.getPicStaticLibrary() != null) {
      return library.getPicStaticLibrary();
    } else {
      return null;
    }
  }

  /**
   * Returns the preferred variant of the library for linking.
   *
   * @param library the input library.
   */
  public static Artifact getLibraryForLinking(LibraryToLink library) {
    if (library.getStaticLibrary() != null) {
      return library.getStaticLibrary();
    } else if (library.getPicStaticLibrary() != null) {
      return library.getPicStaticLibrary();
    } else if (library.getInterfaceLibrary() != null) {
      return library.getInterfaceLibrary();
    } else {
      return library.getDynamicLibrary();
    }
  }

  private static Pair<ImmutableSet<Artifact>, ImmutableSet<Artifact>>
      classifyLibrariesFromCcLinkingContext(CcLinkingContext ccLinkingContext) {
    ImmutableList<LinkerInput> linkerInputs = ccLinkingContext.getLinkerInputs().toList();
    ImmutableSet.Builder<Artifact> alwaysLinkLibrariesBuilder = ImmutableSet.builder();
    for (LinkerInput linkerInput : linkerInputs) {
      for (LibraryToLink libraryToLink : linkerInput.getLibraries()) {
        if (libraryToLink.getAlwayslink()) {
          Artifact library = getLibraryForLinking(libraryToLink);
          alwaysLinkLibrariesBuilder.add(library);
        }
      }
    }
    ImmutableSet<Artifact> alwaysLinkLibraries = alwaysLinkLibrariesBuilder.build();

    ImmutableSet.Builder<Artifact> asNeededlibrariesBuilder = ImmutableSet.builder();
    for (LinkerInput linkerInput : linkerInputs) {
      for (LibraryToLink libraryToLink : linkerInput.getLibraries()) {
        if (!libraryToLink.getAlwayslink()) {
          Artifact library = getLibraryForLinking(libraryToLink);
          if (!alwaysLinkLibraries.contains(library)) {
            asNeededlibrariesBuilder.add(library);
          }
        }
      }
    }
    return Pair.of(asNeededlibrariesBuilder.build(), alwaysLinkLibraries);
  }

  private static ImmutableList<String> dedupSdkLinkopts(NestedSet<LinkOptions> linkopts) {
    HashSet<String> duplicates = new HashSet<>();
    ImmutableList.Builder<String> finalLinkopts = ImmutableList.builder();

    for (LinkOptions linkOptions : linkopts.toList()) {
      ImmutableList<String> args = linkOptions.get();
      for (Iterator<String> iterator = args.iterator(); iterator.hasNext(); ) {
        String arg = iterator.next();
        if (iterator.hasNext() && (arg.equals("-framework") || arg.equals("-weak_framework"))) {
          String framework = iterator.next();
          String key = arg.charAt(1) + framework;
          if (!duplicates.contains(key)) {
            finalLinkopts.add(arg, framework);
            duplicates.add(key);
          }
        } else if (arg.startsWith("-Wl,-framework,") || arg.startsWith("-Wl,-weak_framework,")) {
          String framework = arg.split(",", -1)[2];
          String key = arg.charAt(5) + framework;
          if (!duplicates.contains(key)) {
            finalLinkopts.add(arg.split(",", -1)[1], framework);
            duplicates.add(key);
          }
        } else if (arg.startsWith("-l")) {
          if (!duplicates.contains(arg)) {
            finalLinkopts.add(arg);
            duplicates.add(arg);
          }
        } else {
          finalLinkopts.add(arg);
        }
      }
    }

    return finalLinkopts.build();
  }

  /**
   * Registers any actions necessary to link this rule and its dependencies. Automatically infers
   * the toolchain from the configuration of this CompilationSupport.
   *
   * <p>Dsym bundle is generated if {@link CppConfiguration#appleGenerateDsym()} is set.
   *
   * <p>When Bazel flags {@code --compilation_mode=opt} and {@code --objc_enable_binary_stripping}
   * are specified, additional optimizations will be performed on the linked binary: all-symbol
   * stripping (using {@code /usr/bin/strip}) and dead-code stripping (using linker flags: {@code
   * -dead_strip}).
   *
   * @param linkingInfoProvider the CcLinkingContext with most of the dependency information
   *     required for linking.
   * @param secondaryObjcProvider the ObjcProvider that provides secondary linking info.
   * @param j2ObjcMappingFileProvider contains mapping files for j2objc transpilation
   * @param j2ObjcEntryClassProvider contains j2objc entry class information for dead code removal
   * @param extraLinkArgs any additional arguments to pass to the linker
   * @param extraLinkInputs any additional input artifacts to pass to the link action
   * @param userVariablesExtension the UserVariablesExtension to pass to the linker action
   * @return this compilation support
   */
  @CanIgnoreReturnValue
  public CompilationSupport registerLinkActions(
      CcLinkingContext linkingInfoProvider,
      ObjcProvider secondaryLinkingInfoProvider,
      StarlarkInfo j2ObjcMappingFileProvider,
      StarlarkInfo j2ObjcEntryClassProvider,
      ExtraLinkArgs extraLinkArgs,
      Iterable<Artifact> extraLinkInputs,
      Iterable<String> extraRequestedFeatures,
      Iterable<String> extraDisabledFeatures,
      boolean isStampingEnabled,
      UserVariablesExtension userVariablesExtension)
      throws InterruptedException, RuleErrorException, EvalException {
    // We need to split input libraries into those that require -force_load and those that don't.
    // Clang loads archives specified in filelists and also specified as -force_load twice,
    // resulting in duplicate symbol errors unless they are deduped.
    Pair<ImmutableSet<Artifact>, ImmutableSet<Artifact>> inputLibrarySet =
        classifyLibrariesFromCcLinkingContext(linkingInfoProvider);

    ImmutableSet<Artifact> asNeededLibrarySet = inputLibrarySet.first;
    ImmutableSet<Artifact> alwaysLinkLibrarySet = inputLibrarySet.second;

    if (stripJ2ObjcDeadCode(j2ObjcEntryClassProvider)
        && !secondaryLinkingInfoProvider.get(ObjcProvider.J2OBJC_LIBRARY).toList().isEmpty()) {
      registerJ2ObjcDeadCodeRemovalActions(
          secondaryLinkingInfoProvider, j2ObjcMappingFileProvider, j2ObjcEntryClassProvider);

      asNeededLibrarySet =
          substituteJ2ObjcPrunedLibraries(asNeededLibrarySet, secondaryLinkingInfoProvider);
      alwaysLinkLibrarySet =
          substituteJ2ObjcPrunedLibraries(alwaysLinkLibrarySet, secondaryLinkingInfoProvider);
    }

    ImmutableList<Artifact> asNeededLibraryList = asNeededLibrarySet.asList();
    ImmutableList<Artifact> alwaysLinkLibraryList = alwaysLinkLibrarySet.asList();

    // Passing large numbers of inputs on the command line triggers a bug in Apple's Clang
    // (b/29094356), so we'll create an input list manually and pass -filelist path/to/input/list.
    // We can't populate this list yet--it needs to contain any linkstamp objects, which we won't
    // know about until we actually create the CppLinkAction--but it needs to go into the
    // CppLinkAction too, so create it now.
    Artifact inputFileList = intermediateArtifacts.linkerObjList();

    ImmutableSet<String> allRequestedFeatures =
        new ImmutableSet.Builder<String>()
            .addAll(ruleContext.getFeatures())
            .addAll(extraRequestedFeatures)
            .build();
    ImmutableSet<String> allDisabledFeatures =
        new ImmutableSet.Builder<String>()
            .addAll(ruleContext.getDisabledFeatures())
            .addAll(extraDisabledFeatures)
            .build();
    FeatureConfiguration featureConfiguration =
        CcCommon.configureFeaturesOrReportRuleError(
            ruleContext,
            buildConfiguration,
            allRequestedFeatures,
            allDisabledFeatures,
            Language.OBJC,
            toolchain,
            cppSemantics);

    ImmutableList<Artifact> staticRuntimes;
    try {
      staticRuntimes = toolchain.getStaticRuntimeLinkInputs(featureConfiguration).toList();
    } catch (EvalException e) {
      throw ruleContext.throwWithRuleError(e);
    }

    ObjcVariablesExtension.Builder extensionBuilder =
        new ObjcVariablesExtension.Builder()
            .setRuleContext(ruleContext)
            .setIntermediateArtifacts(intermediateArtifacts)
            .setForceLoadArtifacts(alwaysLinkLibrarySet)
            .setAttributeLinkopts(attributes.linkopts())
            .setDepLinkopts(dedupSdkLinkopts(linkingInfoProvider.getUserLinkFlags()))
            .addVariableCategory(VariableCategory.EXECUTABLE_LINKING_VARIABLES);

    Artifact binaryToLink = getBinaryToLink();

    Label binaryLabel = null;
    try {
      binaryLabel =
          Label.create(ruleContext.getLabel().getPackageIdentifier(), binaryToLink.getFilename());
    } catch (LabelSyntaxException e) {
      // Formed from existing label, just replacing name with artifact name.
    }

    CppConfiguration cppConfiguration = buildConfiguration.getFragment(CppConfiguration.class);
    CcLinkingHelper executableLinkingHelper =
        new CcLinkingHelper(
                ruleContext,
                binaryLabel,
                ruleContext,
                ruleContext,
                cppSemantics,
                featureConfiguration,
                toolchain,
                toolchain.getFdoContext(),
                buildConfiguration,
                cppConfiguration,
                ruleContext.getSymbolGenerator(),
                TargetUtils.getExecutionInfo(
                    ruleContext.getRule(), ruleContext.isAllowTagsPropagation()))
            .setIsStampingEnabled(isStampingEnabled)
            .setTestOrTestOnlyTarget(ruleContext.isTestOnlyTarget() || ruleContext.isTestTarget())
            .addNonCodeLinkerInputs(asNeededLibraryList)
            .addNonCodeLinkerInputs(alwaysLinkLibraryList)
            .addNonCodeLinkerInputs(linkingInfoProvider.getNonCodeInputs().toList())
            .addNonCodeLinkerInputs(ImmutableList.copyOf(extraLinkInputs))
            .addNonCodeLinkerInputs(ImmutableList.copyOf(attributes.linkInputs()))
            .addNonCodeLinkerInputs(ImmutableList.of(inputFileList))
            .addVariableExtension(userVariablesExtension)
            .setShouldCreateStaticLibraries(false)
            .setDynamicLinkType(LinkTargetType.OBJC_EXECUTABLE)
            .setLinkingMode(LinkingMode.STATIC)
            .addLinkopts(ImmutableList.copyOf(extraLinkArgs));

    ImmutableList.Builder<Artifact> linkerOutputs = ImmutableList.builder();

    if (cppConfiguration.appleGenerateDsym()) {
      Artifact dsymSymbol =
          cppConfiguration.objcShouldStripBinary()
              ? intermediateArtifacts.dsymSymbolForUnstrippedBinary()
              : intermediateArtifacts.dsymSymbolForStrippedBinary();
      extensionBuilder
          .setDsymSymbol(dsymSymbol)
          .addVariableCategory(VariableCategory.DSYM_VARIABLES);
      linkerOutputs.add(dsymSymbol);
    }

    if (cppConfiguration.objcGenerateLinkmap()) {
      Artifact linkmap = intermediateArtifacts.linkmap();
      extensionBuilder.setLinkmap(linkmap).addVariableCategory(VariableCategory.LINKMAP_VARIABLES);
      linkerOutputs.add(linkmap);
    }

    executableLinkingHelper.addVariableExtension(extensionBuilder.build());

    executableLinkingHelper.addLinkerOutputs(linkerOutputs.build());

    CcLinkingContext linkstamps =
        CcLinkingContext.builder()
            .addLinkstamps(linkingInfoProvider.getLinkstamps().toList())
            .build();
    executableLinkingHelper.addCcLinkingContexts(ImmutableList.of(linkstamps));

    executableLinkingHelper.link(CcCompilationOutputs.EMPTY);

    ImmutableCollection<Artifact> linkstampValues =
        CppLinkActionBuilder.mapLinkstampsToOutputs(
                linkstamps.getLinkstamps().toSet(),
                ruleContext,
                ruleContext.getRepository(),
                buildConfiguration,
                binaryToLink,
                CppLinkAction.DEFAULT_ARTIFACT_FACTORY)
            .values();

    // Populate the input file list with both the compiled object files and any linkstamp object
    // files.
    registerObjFilelistAction(
        ImmutableSet.<Artifact>builder()
            .addAll(asNeededLibraryList)
            .addAll(staticRuntimes)
            .addAll(linkstampValues)
            .build(),
        inputFileList);

    if (cppConfiguration.objcShouldStripBinary()) {
      registerBinaryStripAction(
          binaryToLink, getStrippingType(extraLinkArgs, featureConfiguration));
    }

    return this;
  }

  /**
   * Registers an action that writes given set of object files to the given objList. This objList is
   * suitable to signal symbols to archive in a libtool archiving invocation.
   */
  // TODO(ulfjack): Use NestedSet for objFiles.
  @CanIgnoreReturnValue
  private CompilationSupport registerObjFilelistAction(
      ImmutableSet<Artifact> objFiles, Artifact objList) {
    CustomCommandLine.Builder objFilesToLinkParam = new CustomCommandLine.Builder();
    NestedSetBuilder<Artifact> treeObjFiles = NestedSetBuilder.stableOrder();

    for (Artifact objFile : objFiles) {
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
            ParameterFile.ParameterFileType.UNQUOTED));
    return this;
  }

  private <T> NestedSet<T> getField(StarlarkInfo provider, String fieldName, Class<T> type)
      throws EvalException {
    return Depset.cast(provider.getValue(fieldName), type, fieldName);
  }

  /** Returns true if this build should strip J2Objc dead code. */
  private boolean stripJ2ObjcDeadCode(StarlarkInfo j2ObjcEntryClassProvider) throws EvalException {
    J2ObjcConfiguration j2objcConfiguration =
        buildConfiguration.getFragment(J2ObjcConfiguration.class);
    NestedSet<String> entryClasses =
        getField(j2ObjcEntryClassProvider, "entry_classes", String.class);

    // Only perform J2ObjC dead code stripping if flag --j2objc_dead_code_removal is specified and
    // users have specified entry classes.
    return j2objcConfiguration.removeDeadCode() && !entryClasses.isEmpty();
  }

  /** Registers actions to perform J2Objc dead code removal. */
  private void registerJ2ObjcDeadCodeRemovalActions(
      ObjcProvider objcProvider,
      StarlarkInfo j2ObjcMappingFileProvider,
      StarlarkInfo j2ObjcEntryClassProvider)
      throws EvalException {
    NestedSet<String> entryClasses =
        getField(j2ObjcEntryClassProvider, "entry_classes", String.class);
    NestedSet<Artifact> j2ObjcDependencyMappingFiles =
        getField(j2ObjcMappingFileProvider, "dependency_mapping_files", Artifact.class);
    NestedSet<Artifact> j2ObjcHeaderMappingFiles =
        getField(j2ObjcMappingFileProvider, "header_mapping_files", Artifact.class);
    NestedSet<Artifact> j2ObjcArchiveSourceMappingFiles =
        getField(j2ObjcMappingFileProvider, "archive_source_mapping_files", Artifact.class);

    for (Artifact j2objcArchive : objcProvider.get(ObjcProvider.J2OBJC_LIBRARY).toList()) {
      Artifact prunedJ2ObjcArchive = intermediateArtifacts.j2objcPrunedArchive(j2objcArchive);
      Artifact dummyArchive =
          getLibraryForLinking(
              ruleContext
                  .getPrerequisite("$dummy_lib", CcInfo.PROVIDER)
                  .getCcLinkingContext()
                  .getLibraries()
                  .getSingleton());

      CustomCommandLine commandLine =
          CustomCommandLine.builder()
              .addExecPath("--input_archive", j2objcArchive)
              .addExecPath("--output_archive", prunedJ2ObjcArchive)
              .addExecPath("--dummy_archive", dummyArchive)
              .addExecPaths(
                  "--dependency_mapping_files",
                  VectorArg.join(",").each(j2ObjcDependencyMappingFiles))
              .addExecPaths(
                  "--header_mapping_files", VectorArg.join(",").each(j2ObjcHeaderMappingFiles))
              .addExecPaths(
                  "--archive_source_mapping_files",
                  VectorArg.join(",").each(j2ObjcArchiveSourceMappingFiles))
              .add("--entry_classes")
              .addAll(VectorArg.join(",").each(entryClasses))
              .build();

      ruleContext.registerAction(
          new SpawnAction.Builder()
              .setMnemonic("DummyPruner")
              .setExecutable(ruleContext.getExecutablePrerequisite("$j2objc_dead_code_pruner"))
              .addInput(dummyArchive)
              .addInput(j2objcArchive)
              .addTransitiveInputs(j2ObjcDependencyMappingFiles)
              .addTransitiveInputs(j2ObjcHeaderMappingFiles)
              .addTransitiveInputs(j2ObjcArchiveSourceMappingFiles)
              .addCommandLine(
                  commandLine,
                  ParamFileInfo.builder(ParameterFile.ParameterFileType.UNQUOTED)
                      .setCharset(ISO_8859_1)
                      .setUseAlways(true)
                      .build())
              .addOutput(prunedJ2ObjcArchive)
              .setExecGroup("j2objc")
              .build(ruleContext));
    }
  }

  /** Returns a set of libraries with all unpruned J2ObjC libraries substituted with pruned ones. */
  private ImmutableSet<Artifact> substituteJ2ObjcPrunedLibraries(
      ImmutableSet<Artifact> originalLibraries, ObjcProvider objcProvider) {
    ImmutableSet.Builder<Artifact> libraries = new ImmutableSet.Builder<>();

    Set<Artifact> unprunedJ2ObjcLibs = objcProvider.get(ObjcProvider.J2OBJC_LIBRARY).toSet();
    for (Artifact library : originalLibraries) {
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
  private Artifact getBinaryToLink() {

    // When compilation_mode=opt and objc_enable_binary_stripping are specified, the unstripped
    // binary containing debug symbols is generated by the linker, which also needs the debug
    // symbols for dead-code removal. The binary is also used to generate dSYM bundle if
    // --apple_generate_dsym is specified. A symbol strip action is later registered to strip
    // the symbol table from the unstripped binary.
    CppConfiguration cppConfiguration = buildConfiguration.getFragment(CppConfiguration.class);
    return cppConfiguration.objcShouldStripBinary()
        ? intermediateArtifacts.unstrippedSingleArchitectureBinary()
        : intermediateArtifacts.strippedSingleArchitectureBinary();
  }

  private static CommandLine symbolStripCommandLine(
      ImmutableList<String> extraFlags, Artifact unstrippedArtifact, Artifact strippedArtifact) {
    return CustomCommandLine.builder()
        .add("/usr/bin/xcrun")
        .add(STRIP)
        .addAll(extraFlags)
        .addExecPath("-o", strippedArtifact)
        .addPath(unstrippedArtifact.getExecPath())
        .build();
  }

  /** Signals if stripping should include options for dynamic libraries. */
  private enum StrippingType {
    DEFAULT,
    DYNAMIC_LIB,
    LOADABLE_BUNDLE,
    KERNEL_EXTENSION
  }

  /**
   * Registers an action that uses the 'strip' tool to perform binary stripping on the given binary
   * subject to the given {@link StrippingType}.
   */
  private void registerBinaryStripAction(Artifact binaryToLink, StrippingType strippingType) {
    final ImmutableList<String> stripArgs;
    switch (strippingType) {
      case DYNAMIC_LIB:
      case LOADABLE_BUNDLE:
      case KERNEL_EXTENSION:
        // For dylibs, loadable bundles, and kexts, must strip only local symbols.
        stripArgs = ImmutableList.of("-x");
        break;
      case DEFAULT:
        stripArgs = ImmutableList.<String>of();
        break;
      default:
        throw new IllegalArgumentException("Unsupported stripping type " + strippingType);
    }

    Artifact strippedBinary = intermediateArtifacts.strippedSingleArchitectureBinary();

    ruleContext.registerAction(
        ObjcRuleClasses.spawnAppleEnvActionBuilder(
                XcodeConfigInfo.fromRuleContext(ruleContext),
                appleConfiguration.getSingleArchPlatform())
            .setMnemonic("ObjcBinarySymbolStrip")
            .addCommandLine(symbolStripCommandLine(stripArgs, binaryToLink, strippedBinary))
            .addOutput(strippedBinary)
            .addInput(binaryToLink)
            .build(ruleContext));
  }

  public static Optional<Artifact> getCustomModuleMap(RuleContext ruleContext) {
    if (ruleContext.attributes().has("module_map", BuildType.LABEL)) {
      return Optional.fromNullable(ruleContext.getPrerequisiteArtifact("module_map"));
    }
    return Optional.absent();
  }
}
