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
package com.google.devtools.build.lib.rules.cpp;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.ArtifactResolver;
import com.google.devtools.build.lib.actions.CommandAction;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.PackageRootResolutionException;
import com.google.devtools.build.lib.actions.PackageRootResolver;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.extra.CppCompileInfo;
import com.google.devtools.build.lib.actions.extra.ExtraActionInfo;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.ExecutionInfoSpecifier;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.PerLabelOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.CollectionUtils;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppCompileActionContext.Reply;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration.Tool;
import com.google.devtools.build.lib.util.DependencySet;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.util.ShellEscaper;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import javax.annotation.Nullable;
import javax.annotation.concurrent.GuardedBy;

/** Action that represents some kind of C++ compilation step. */
@ThreadCompatible
public class CppCompileAction extends AbstractAction
    implements IncludeScannable, ExecutionInfoSpecifier, CommandAction {
  /**
   * Represents logic that determines if an artifact is a special input, meaning that it may require
   * additional inputs when it is compiled or may not be available to other actions.
   */
  public interface SpecialInputsHandler {
    /** Returns if {@code includedFile} is special, so may not be available to other actions. */
    boolean isSpecialFile(Artifact includedFile);

    /** Returns the set of files to be added for an included file (as returned in the .d file). */
    Collection<Artifact> getInputsForIncludedFile(
        Artifact includedFile, ArtifactResolver artifactResolver);
  }

  static final SpecialInputsHandler VOID_SPECIAL_INPUTS_HANDLER =
      new SpecialInputsHandler() {
        @Override
        public boolean isSpecialFile(Artifact includedFile) {
          return false;
        }

        @Override
        public Collection<Artifact> getInputsForIncludedFile(
            Artifact includedFile, ArtifactResolver artifactResolver) {
          return ImmutableList.of();
        }
      };

  private static final int VALIDATION_DEBUG = 0;  // 0==none, 1==warns/errors, 2==all
  private static final boolean VALIDATION_DEBUG_WARN = VALIDATION_DEBUG >= 1;

  /**
   * A string constant for the c compilation action.
   */
  public static final String C_COMPILE = "c-compile";

  /**
   * A string constant for the c++ compilation action.
   */
  public static final String CPP_COMPILE = "c++-compile";

  /**
   * A string constant for the objc compilation action.
   */
  public static final String OBJC_COMPILE = "objc-compile";

  /**
   * A string constant for the objc++ compile action.
   */
  public static final String OBJCPP_COMPILE = "objc++-compile";
  
  /**
   * A string constant for the c++ header parsing.
   */
  public static final String CPP_HEADER_PARSING = "c++-header-parsing";

  /**
   * A string constant for the c++ header preprocessing.
   */
  public static final String CPP_HEADER_PREPROCESSING = "c++-header-preprocessing";

  /**
   * A string constant for the c++ module compilation action.
   * Note: currently we don't support C module compilation.
   */
  public static final String CPP_MODULE_COMPILE = "c++-module-compile";

  /**
   * A string constant for the assembler actions.
   */
  public static final String ASSEMBLE = "assemble";
  public static final String PREPROCESS_ASSEMBLE = "preprocess-assemble";

  /**
   * A string constant for the clif actions. Bazel enables different features of the toolchain based
   * on the name of the action. This name enables the clif_matcher feature, which switches the
   * "compiler" to the clif_matcher and adds some additional arguments as described in the CROSSTOOL
   * file.
   */
  public static final String CLIF_MATCH = "clif-match";

  // TODO(ulfjack): this is only used to get the local shell environment and to check if coverage is
  // enabled. Move those two things to local fields and drop this. Accessing anything other than
  // these fields can impact correctness!
  private final BuildConfiguration configuration;
  protected final Artifact outputFile;
  private final Label sourceLabel;
  private final Artifact optionalSourceFile;
  private final NestedSet<Artifact> mandatoryInputs;
  private final boolean shouldScanIncludes;
  private final boolean shouldPruneModules;
  private final boolean usePic;
  private final CppCompilationContext context;
  private final Iterable<IncludeScannable> lipoScannables;
  private final ImmutableList<Artifact> builtinIncludeFiles;
  // A list of files to include scan that are not source files, pcm files, lipo scannables, or
  // included via a command-line "-include file.h". Actions that use non C++ files as source
  // files--such as Clif--may use this mechanism.
  private final ImmutableList<Artifact> additionalIncludeScannables;
  @VisibleForTesting public final CppCompileCommandLine cppCompileCommandLine;
  private final ImmutableSet<String> executionRequirements;
  private final ImmutableMap<String, String> environment;

  @VisibleForTesting final CppConfiguration cppConfiguration;
  private final FeatureConfiguration featureConfiguration;
  protected final Class<? extends CppCompileActionContext> actionContext;
  protected final SpecialInputsHandler specialInputsHandler;
  protected final CppSemantics cppSemantics;

  /**
   * Identifier for the actual execution time behavior of the action.
   *
   * <p>Required because the behavior of this class can be modified by injecting code in the
   * constructor or by inheritance, and we want to have different cache keys for those.
   */
  private final UUID actionClassId;

  // This can be read/written from multiple threads, and so accesses should be synchronized.
  @GuardedBy("this")
  private boolean inputsKnown = false;

  /**
   * Set when the action prepares for execution. Used to preserve state between preparation and
   * execution.
   */
  private Collection<Artifact> additionalInputs = null;
  
  private CcToolchainFeatures.Variables overwrittenVariables = null;

  private ImmutableList<Artifact> resolvedInputs = ImmutableList.<Artifact>of();

  /**
   * Creates a new action to compile C/C++ source files.
   *
   * @param owner the owner of the action, usually the configured target that emitted it
   * @param features TODO(bazel-team): Add parameter description.
   * @param featureConfiguration TODO(bazel-team): Add parameter description.
   * @param variables TODO(bazel-team): Add parameter description.
   * @param sourceFile the source file that should be compiled. {@code mandatoryInputs} must contain
   *     this file
   * @param shouldScanIncludes a boolean indicating whether scanning of {@code sourceFile} is to be
   *     performed looking for inclusions.
   * @param usePic TODO(bazel-team): Add parameter description.
   * @param sourceLabel the label of the rule the source file is generated by
   * @param mandatoryInputs any additional files that need to be present for the compilation to
   *     succeed, can be empty but not null, for example, extra sources for FDO.
   * @param outputFile the object file that is written as result of the compilation, or the fake
   *     object for {@link FakeCppCompileAction}s
   * @param dotdFile the .d file that is generated as a side-effect of compilation
   * @param gcnoFile the coverage notes that are written in coverage mode, can be null
   * @param dwoFile the .dwo output file where debug information is stored for Fission builds (null
   *     if Fission mode is disabled)
   * @param optionalSourceFile an additional optional source file (null if unneeded)
   * @param configuration the build configurations
   * @param cppConfiguration TODO(bazel-team): Add parameter description.
   * @param context the compilation context
   * @param actionContext TODO(bazel-team): Add parameter description.
   * @param copts options for the compiler
   * @param coptsFilter regular expression to remove options from {@code copts}
   * @param specialInputsHandler TODO(bazel-team): Add parameter description.
   * @param lipoScannables List of artifacts to include-scan when this action is a lipo action
   * @param additionalIncludeScannables list of additional artifacts to include-scan
   * @param actionClassId TODO(bazel-team): Add parameter description
   * @param executionRequirements out-of-band hints to be passed to the execution backend to signal
   *     platform requirements
   * @param environment TODO(bazel-team): Add parameter description
   * @param actionName a string giving the name of this action for the purpose of toolchain
   *     evaluation
   * @param ruleContext The rule-context that produced this action
   */
  protected CppCompileAction(
      ActionOwner owner,
      // TODO(bazel-team): Eventually we will remove 'features'; all functionality in 'features'
      // will be provided by 'featureConfiguration'.
      ImmutableList<String> features,
      FeatureConfiguration featureConfiguration,
      CcToolchainFeatures.Variables variables,
      Artifact sourceFile,
      boolean shouldScanIncludes,
      boolean shouldPruneModules,
      boolean usePic,
      Label sourceLabel,
      NestedSet<Artifact> mandatoryInputs,
      Artifact outputFile,
      DotdFile dotdFile,
      @Nullable Artifact gcnoFile,
      @Nullable Artifact dwoFile,
      Artifact optionalSourceFile,
      BuildConfiguration configuration,
      CppConfiguration cppConfiguration,
      CppCompilationContext context,
      Class<? extends CppCompileActionContext> actionContext,
      ImmutableList<String> copts,
      Predicate<String> coptsFilter,
      SpecialInputsHandler specialInputsHandler,
      Iterable<IncludeScannable> lipoScannables,
      ImmutableList<Artifact> additionalIncludeScannables,
      UUID actionClassId,
      ImmutableSet<String> executionRequirements,
      ImmutableMap<String, String> environment,
      String actionName,
      RuleContext ruleContext,
      CppSemantics cppSemantics) {
    super(
        owner,
        createInputs(
            ruleContext,
            mandatoryInputs,
            context.getTransitiveCompilationPrerequisites(),
            optionalSourceFile,
            lipoScannables),
        CollectionUtils.asListWithoutNulls(
            outputFile, (dotdFile == null ? null : dotdFile.artifact()), gcnoFile, dwoFile));
    this.configuration = configuration;
    this.sourceLabel = sourceLabel;
    this.outputFile = Preconditions.checkNotNull(outputFile);
    this.optionalSourceFile = optionalSourceFile;
    this.context = context;
    this.specialInputsHandler = specialInputsHandler;
    this.cppConfiguration = cppConfiguration;
    this.featureConfiguration = featureConfiguration;
    // inputsKnown begins as the logical negation of shouldScanIncludes.
    // When scanning includes, the inputs begin as not known, and become
    // known after inclusion scanning. When *not* scanning includes,
    // the inputs are as declared, hence known, and remain so.
    this.shouldScanIncludes = shouldScanIncludes;
    this.shouldPruneModules = shouldPruneModules;
    this.usePic = usePic;
    this.inputsKnown = !shouldScanIncludes;
    this.cppCompileCommandLine =
        new CppCompileCommandLine(
            sourceFile, dotdFile, copts, coptsFilter, features, variables, actionName);
    this.actionContext = actionContext;
    this.lipoScannables = lipoScannables;
    this.actionClassId = actionClassId;
    this.executionRequirements = executionRequirements;
    this.environment = environment;

    // We do not need to include the middleman artifact since it is a generated
    // artifact and will definitely exist prior to this action execution.
    this.mandatoryInputs = mandatoryInputs;
    this.builtinIncludeFiles = CppHelper.getToolchain(ruleContext).getBuiltinIncludeFiles();
    this.cppSemantics = cppSemantics;
    if (cppSemantics.needsIncludeValidation()) {
      verifyIncludePaths(ruleContext);
    }
    this.additionalIncludeScannables = ImmutableList.copyOf(additionalIncludeScannables);
  }

  /**
   * Verifies that the include paths of this action are within the limits of the execution root.
   */
  private void verifyIncludePaths(RuleContext ruleContext) {
    if (ruleContext == null) {
      return;
    }

    Iterable<PathFragment> ignoredDirs = getValidationIgnoredDirs();

    // We currently do not check the output of:
    // - getQuoteIncludeDirs(): those only come from includes attributes, and are checked in
    //   CcCommon.getIncludeDirsFromIncludesAttribute().
    // - getBuiltinIncludeDirs(): while in practice this doesn't happen, bazel can be configured
    //   to use an absolute system root, in which case the builtin include dirs might be absolute.
    for (PathFragment include : Iterables.concat(getIncludeDirs(), getSystemIncludeDirs())) {

      // Ignore headers from built-in include directories.
      if (FileSystemUtils.startsWithAny(include, ignoredDirs)) {
        continue;
      }

      // One starting ../ is okay for getting to a sibling repository.
      PathFragment originalInclude = include;
      if (include.startsWith(new PathFragment(Label.EXTERNAL_PATH_PREFIX))) {
        include = include.relativeTo(Label.EXTERNAL_PATH_PREFIX);
      }

      if (include.isAbsolute()
          || !PathFragment.EMPTY_FRAGMENT.getRelative(include).normalize().isNormalized()) {
        ruleContext.ruleError(
            "The include path '" + include + "' references a path outside of the execution root.");
      }
    }
  }

  private static NestedSet<Artifact> createInputs(
      RuleContext ruleContext,
      NestedSet<Artifact> mandatoryInputs,
      Set<Artifact> prerequisites,
      Artifact optionalSourceFile,
      Iterable<IncludeScannable> lipoScannables) {
    NestedSetBuilder<Artifact> builder = NestedSetBuilder.stableOrder();
    if (optionalSourceFile != null) {
      builder.add(optionalSourceFile);
    }
    builder.addAll(prerequisites);
    builder.addAll(CppHelper.getToolchain(ruleContext).getBuiltinIncludeFiles());
    builder.addTransitive(mandatoryInputs);
    if (lipoScannables != null && lipoScannables.iterator().hasNext()) {
      // We need to add "legal generated scanner files" coming through LIPO scannables here. These
      // usually contain pre-grepped source files, i.e. files just containing the #include lines
      // extracted from generated files. With LIPO, some of these files can be accessed, even though
      // there is no direct dependency on them. Adding the artifacts as inputs to this compile
      // action ensures that the action generating them is actually executed.
      for (IncludeScannable lipoScannable : lipoScannables) {
        for (Artifact value : lipoScannable.getLegalGeneratedScannerFileMap().values()) {
          if (value != null) {
            builder.add(value);
          }
        }
      }
    }
    return builder.build();
  }

  /**
   * Whether we should do "include scanning". Note that this does *not* mean whether we should parse
   * the .d files to determine which include files were used during compilation. Instead, this means
   * whether we should a) run the pre-execution include scanner (see {@code IncludeScanningContext})
   * if one exists and b) whether the action inputs should be modified to match the results of that
   * pre-execution scanning and (if enabled) again after execution to match the results of the .d
   * file parsing.
   *
   * <p>This does *not* have anything to do with "hdrs_check".
   */
  public boolean shouldScanIncludes() {
    return shouldScanIncludes;
  }

  @Override
  public List<PathFragment> getBuiltInIncludeDirectories() {
    return cppConfiguration.getBuiltInIncludeDirectories();
  }

  @Nullable
  @Override
  public List<Artifact> getBuiltInIncludeFiles() {
    return builtinIncludeFiles;
  }

  public List<Artifact> getadditionalIncludeScannables() {
    return additionalIncludeScannables;
  }

  public String getHostSystemName() {
    return cppConfiguration.getHostSystemName();
  }

  @Override
  public NestedSet<Artifact> getMandatoryInputs() {
    return mandatoryInputs;
  }

  @Override
  public synchronized boolean inputsKnown() {
    return inputsKnown;
  }

  /**
   * Returns the list of additional inputs found by dependency discovery, during action preparation,
   * and clears the stored list. {@link #prepare} must be called before this method is called, on
   * each action execution.
   */
  public Collection<? extends ActionInput> getAdditionalInputs() {
    Collection<? extends ActionInput> result = Preconditions.checkNotNull(additionalInputs);
    additionalInputs = null;
    return result;
  }

  @VisibleForTesting
  public void setResolvedInputsForTesting(ImmutableList<Artifact> resolvedInputs) {
    this.resolvedInputs = resolvedInputs;
  }

  @Override
  public boolean discoversInputs() {
    return true;
  }

  @Nullable
  @Override
  public Iterable<Artifact> discoverInputs(ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException, InterruptedException {
    Executor executor = actionExecutionContext.getExecutor();
    Collection<Artifact> initialResult;
    try {
      initialResult = executor.getContext(actionContext)
          .findAdditionalInputs(this, actionExecutionContext);
    } catch (ExecException e) {
      throw e.toActionExecutionException("Include scanning of rule '" + getOwner().getLabel() + "'",
          executor.getVerboseFailures(), this);
    }
    if (initialResult == null) {
      // We will find inputs during execution. Store an empty list to show we did try to discover
      // inputs and return null to inform the caller that inputs will be discovered later.
      this.additionalInputs = ImmutableList.of();
      return null;
    }

    if (shouldPruneModules) {
      Set<Artifact> initialResultSet = Sets.newLinkedHashSet(initialResult);
      List<String> usedModulePaths = Lists.newArrayList();
      for (Artifact usedModule : context.getUsedModules(usePic, initialResultSet)) {
        initialResultSet.add(usedModule);
        usedModulePaths.add(usedModule.getExecPathString());
      }
      CcToolchainFeatures.Variables.Builder variableBuilder =
          new CcToolchainFeatures.Variables.Builder();
      variableBuilder.addSequenceVariable("module_files", usedModulePaths);
      this.overwrittenVariables = variableBuilder.build();
      initialResult = initialResultSet;
    }
    
    this.additionalInputs = initialResult;
    // In some cases, execution backends need extra files for each included file. Add them
    // to the set of inputs the caller may need to be aware of.
    Collection<Artifact> result = new HashSet<>();
    ArtifactResolver artifactResolver =
        executor.getContext(IncludeScanningContext.class).getArtifactResolver();
    for (Artifact artifact : initialResult) {
      result.addAll(specialInputsHandler.getInputsForIncludedFile(artifact, artifactResolver));
    }
    for (Artifact artifact : getInputs()) {
      result.addAll(specialInputsHandler.getInputsForIncludedFile(artifact, artifactResolver));
    }
    // TODO(ulfjack): This only works if include scanning is enabled; the cleanup is in progress,
    // and this needs to be fixed before we can even consider disabling it.
    resolvedInputs = ImmutableList.copyOf(result);
    if (result.isEmpty()) {
      result = initialResult;
    } else {
      result.addAll(initialResult);
    }
    return result;
  }

  @Override
  public Artifact getPrimaryInput() {
    return getSourceFile();
  }

  @Override
  public Artifact getPrimaryOutput() {
    return getOutputFile();
  }

  /**
   * Returns the path of the c/cc source for gcc.
   */
  public final Artifact getSourceFile() {
    return cppCompileCommandLine.sourceFile;
  }

  /**
   * Returns the path where gcc should put its result.
   */
  public Artifact getOutputFile() {
    return outputFile;
  }

  protected PathFragment getInternalOutputFile() {
    return outputFile.getExecPath();
  }

  @Override
  public Map<Artifact, Artifact> getLegalGeneratedScannerFileMap() {
    Map<Artifact, Artifact> legalOuts = new HashMap<>();

    for (Artifact a : context.getDeclaredIncludeSrcs()) {
      if (!a.isSourceArtifact()) {
        legalOuts.put(a, null);
      }
    }
    for (Pair<Artifact, Artifact> pregreppedSrcs : context.getPregreppedHeaders()) {
      Artifact hdr = pregreppedSrcs.getFirst();
      Preconditions.checkState(!hdr.isSourceArtifact(), hdr);
      legalOuts.put(hdr, pregreppedSrcs.getSecond());
    }
    return Collections.unmodifiableMap(legalOuts);
  }

  /**
   * Returns the path where gcc should put the discovered dependency
   * information.
   */
  public DotdFile getDotdFile() {
    return cppCompileCommandLine.dotdFile;
  }

  @VisibleForTesting
  public CppCompilationContext getContext() {
    return context;
  }

  @Override
  public List<PathFragment> getQuoteIncludeDirs() {
    return context.getQuoteIncludeDirs();
  }

  @Override
  public List<PathFragment> getIncludeDirs() {
    ImmutableList.Builder<PathFragment> result = ImmutableList.builder();
    result.addAll(context.getIncludeDirs());
    for (String opt : cppCompileCommandLine.copts) {
      if (opt.startsWith("-I") && opt.length() > 2) {
        // We insist on the combined form "-Idir".
        result.add(new PathFragment(opt.substring(2)));
      }
    }
    return result.build();
  }

  @Override
  public List<PathFragment> getSystemIncludeDirs() {
    // TODO(bazel-team): parsing the command line flags here couples us to gcc-style compiler
    // command lines; use a different way to specify system includes (for example through a
    // system_includes attribute in cc_toolchain); note that that would disallow users from
    // specifying system include paths via the copts attribute.
    // Currently, this works together with the include_paths features because getCommandLine() will
    // get the system include paths from the CppCompilationContext instead.
    ImmutableList.Builder<PathFragment> result = ImmutableList.builder();
    List<String> compilerOptions = getCompilerOptions();
    for (int i = 0; i < compilerOptions.size(); i++) {
      String opt = compilerOptions.get(i);
      if (opt.startsWith("-isystem")) {
        if (opt.length() > 8) {
          result.add(new PathFragment(opt.substring(8).trim()));
        } else if (i + 1 < compilerOptions.size()) {
          i++;
          result.add(new PathFragment(compilerOptions.get(i)));
        } else {
          System.err.println("WARNING: dangling -isystem flag in options for " + prettyPrint());
        }
      }
    }
    return result.build();
  }

  @Override
  public List<String> getCmdlineIncludes() {
    ImmutableList.Builder<String> cmdlineIncludes = ImmutableList.builder();
    List<String> args = getArgv();
    for (Iterator<String> argi = args.iterator(); argi.hasNext();) {
      String arg = argi.next();
      if (arg.equals("-include") && argi.hasNext()) {
        cmdlineIncludes.add(argi.next());
      }
    }
    return cmdlineIncludes.build();
  }

  @Override
  public Artifact getMainIncludeScannerSource() {
    return CppFileTypes.CPP_MODULE_MAP.matches(getSourceFile().getPath())
        ? Iterables.getFirst(context.getHeaderModuleSrcs(), null)
        : getSourceFile();
  }

  @Override
  public Collection<Artifact> getIncludeScannerSources() {
    NestedSetBuilder<Artifact> builder = NestedSetBuilder.stableOrder();
    if (CppFileTypes.CPP_MODULE_MAP.matches(getSourceFile().getPath())) {
      // If this is an action that compiles the header module itself, the source we build is the
      // module map, and we need to include-scan all headers that are referenced in the module map.
      // We need to do include scanning as long as we want to support building code bases that are
      // not fully strict layering clean.
      builder.addAll(context.getHeaderModuleSrcs());
    } else {
      builder.add(getSourceFile());
      builder.addAll(additionalIncludeScannables);
    }
    return builder.build().toCollection();
  }

  @Override
  public Iterable<IncludeScannable> getAuxiliaryScannables() {
    return lipoScannables;
  }

  /**
   * Returns the list of "-D" arguments that should be used by this gcc
   * invocation. Only used for testing.
   */
  @VisibleForTesting
  public ImmutableCollection<String> getDefines() {
    return context.getDefines();
  }

  @Override
  public ImmutableMap<String, String> getEnvironment() {
    Map<String, String> environment = new LinkedHashMap<>(configuration.getLocalShellEnvironment());
    if (configuration.isCodeCoverageEnabled()) {
      environment.put("PWD", "/proc/self/cwd");
    }

    environment.putAll(this.environment);
    environment.putAll(cppCompileCommandLine.getEnvironment());

    // TODO(bazel-team): Check (crosstool) host system name instead of using OS.getCurrent.
    if (OS.getCurrent() == OS.WINDOWS) {
      // TODO(bazel-team): Both GCC and clang rely on their execution directories being on
      // PATH, otherwise they fail to find dependent DLLs (and they fail silently...). On
      // the other hand, Windows documentation says that the directory of the executable
      // is always searched for DLLs first. Not sure what to make of it.
      // Other options are to forward the system path (brittle), or to add a PATH field to
      // the crosstool file.
      //
      // @see com.google.devtools.build.lib.rules.cpp.CppLinkAction#getEnvironment
      environment.put("PATH", cppConfiguration.getToolPathFragment(Tool.GCC).getParentDirectory()
          .getPathString());
   }
    return ImmutableMap.copyOf(environment);
  }

  /**
   * Returns a new, mutable list of command and arguments (argv) to be passed
   * to the gcc subprocess.
   */
  public final List<String> getArgv() {
    return getArgv(getInternalOutputFile());
  }

  @Override
  public List<String> getArguments() {
    return getArgv();
  }

  protected final List<String> getArgv(PathFragment outputFile) {
    return cppCompileCommandLine.getArgv(outputFile, overwrittenVariables);
  }

  @Override
  public boolean extraActionCanAttach() {
    return cppConfiguration.alwaysAttachExtraActions()
        || !specialInputsHandler.isSpecialFile(getPrimaryInput());
  }

  @Override
  public ExtraActionInfo.Builder getExtraActionInfo() {
    CppCompileInfo.Builder info = CppCompileInfo.newBuilder();
    info.setTool(cppConfiguration.getToolPathFragment(Tool.GCC).getPathString());
    for (String option : getCompilerOptions()) {
      info.addCompilerOption(option);
    }
    info.setOutputFile(outputFile.getExecPathString());
    info.setSourceFile(getSourceFile().getExecPathString());
    if (inputsKnown()) {
      info.addAllSourcesAndHeaders(Artifact.toExecPaths(getInputs()));
    } else {
      info.addSourcesAndHeaders(getSourceFile().getExecPathString());
      info.addAllSourcesAndHeaders(
          Artifact.toExecPaths(context.getDeclaredIncludeSrcs()));
    }

    return super.getExtraActionInfo()
        .setExtension(CppCompileInfo.cppCompileInfo, info.build());
  }

  /**
   * Returns the compiler options.
   */
  @VisibleForTesting
  public List<String> getCompilerOptions() {
    return cppCompileCommandLine.getCompilerOptions(/*updatedVariables=*/null);
  }

  @Override
  public Map<String, String> getExecutionInfo() {
    ImmutableMap.Builder<String, String> result = ImmutableMap.<String, String>builder();
    for (String requirement : executionRequirements) {
      result.put(requirement, "");
    }
    return result.build();
  }
  
  /**
   * Enforce that the includes actually visited during the compile were properly
   * declared in the rules.
   *
   * <p>The technique is to walk through all of the reported includes that gcc
   * emits into the .d file, and verify that they came from acceptable
   * relative include directories. This is done in two steps:
   *
   * <p>First, each included file is stripped of any include path prefix from
   * {@code quoteIncludeDirs} to produce an effective relative include dir+name.
   *
   * <p>Second, the remaining directory is looked up in {@code declaredIncludeDirs},
   * a list of acceptable dirs. This list contains a set of dir fragments that
   * have been calculated by the configured target to be allowable for inclusion
   * by this source. If no match is found, an error is reported and an exception
   * is thrown.
   *
   * @throws ActionExecutionException iff there was an undeclared dependency
   */
  @VisibleForTesting
  public void validateInclusions(
      Iterable<Artifact> inputsForValidation,
      ArtifactExpander artifactExpander,
      EventHandler eventHandler)
      throws ActionExecutionException {
    IncludeProblems errors = new IncludeProblems();
    IncludeProblems warnings = new IncludeProblems();
    Set<Artifact> allowedIncludes = new HashSet<>();
    for (Artifact input : mandatoryInputs) {
      if (input.isMiddlemanArtifact() || input.isTreeArtifact()) {
        artifactExpander.expand(input, allowedIncludes);
      }
      allowedIncludes.add(input);
    }
    allowedIncludes.addAll(resolvedInputs);

    if (optionalSourceFile != null) {
      allowedIncludes.add(optionalSourceFile);
    }
    Iterable<PathFragment> ignoreDirs = getValidationIgnoredDirs();

    // Copy the sets to hash sets for fast contains checking.
    // Avoid immutable sets here to limit memory churn.
    Set<PathFragment> declaredIncludeDirs = Sets.newHashSet(context.getDeclaredIncludeDirs());
    Set<PathFragment> warnIncludeDirs = Sets.newHashSet(context.getDeclaredIncludeWarnDirs());
    Set<Artifact> declaredIncludeSrcs = Sets.newHashSet(getDeclaredIncludeSrcs());
    for (Artifact input : inputsForValidation) {
      if (context.getTransitiveCompilationPrerequisites().contains(input)
          || allowedIncludes.contains(input)) {
        continue; // ignore our fixed source in mandatoryInput: we just want includes
      }
      // Ignore headers from built-in include directories.
      if (FileSystemUtils.startsWithAny(input.getExecPath(), ignoreDirs)) {
        continue;
      }
      if (!isDeclaredIn(input, declaredIncludeDirs, declaredIncludeSrcs)) {
        // This call can never match the declared include sources (they would be matched above).
        // There are no declared include sources we need to warn about, so use an empty set here.
        if (isDeclaredIn(input, warnIncludeDirs, ImmutableSet.<Artifact>of())) {
          warnings.add(input.getPath().toString());
        } else {
          errors.add(input.getPath().toString());
        }
      }
    }
    if (VALIDATION_DEBUG_WARN) {
      synchronized (System.err) {
        if (VALIDATION_DEBUG >= 2 || errors.hasProblems() || warnings.hasProblems()) {
          if (errors.hasProblems()) {
            System.err.println("ERROR: Include(s) were not in declared srcs:");
          } else if (warnings.hasProblems()) {
            System.err.println("WARN: Include(s) were not in declared srcs:");
          } else {
            System.err.println("INFO: Include(s) were OK for '" + getSourceFile()
                + "', declared srcs:");
          }
          for (Artifact a : context.getDeclaredIncludeSrcs()) {
            System.err.println("  '" + a.toDetailString() + "'");
          }
          System.err.println(" or under declared dirs:");
          for (PathFragment f : Sets.newTreeSet(context.getDeclaredIncludeDirs())) {
            System.err.println("  '" + f + "'");
          }
          System.err.println(" or under declared warn dirs:");
          for (PathFragment f : Sets.newTreeSet(context.getDeclaredIncludeWarnDirs())) {
            System.err.println("  '" + f + "'");
          }
          System.err.println(" with prefixes:");
          for (PathFragment dirpath : context.getQuoteIncludeDirs()) {
            System.err.println("  '" + dirpath + "'");
          }
        }
      }
    }

    if (warnings.hasProblems()) {
      eventHandler.handle(
          Event.warn(
              getOwner().getLocation(),
              warnings.getMessage(this, getSourceFile()))
              .withTag(Label.print(getOwner().getLabel())));
    }
    errors.assertProblemFree(this, getSourceFile());
  }

  private Iterable<PathFragment> getValidationIgnoredDirs() {
    List<PathFragment> cxxSystemIncludeDirs = cppConfiguration.getBuiltInIncludeDirectories();
    return Iterables.concat(
        cxxSystemIncludeDirs, context.getSystemIncludeDirs());
  }

  /**
   * Returns true if an included artifact is declared in a set of allowed
   * include directories. The simple case is that the artifact's parent
   * directory is contained in the set, or is empty.
   *
   * <p>This check also supports a wildcard suffix of '**' for the cases where the
   * calculations are inexact.
   *
   * <p>It also handles unseen non-nested-package subdirs by walking up the path looking
   * for matches.
   */
  private static boolean isDeclaredIn(
      Artifact input, Set<PathFragment> declaredIncludeDirs, Set<Artifact> declaredIncludeSrcs) {
    // First check if it's listed in "srcs". If so, then its declared & OK.
    if (declaredIncludeSrcs.contains(input)) {
      return true;
    }
    // If it's a derived artifact, then it MUST be listed in "srcs" as checked above.
    // We define derived here as being not source and not under the include link tree.
    if (!input.isSourceArtifact()
        && !input.getRoot().getExecPath().getBaseName().equals("include")) {
      return false;
    }
    // Need to do dir/package matching: first try a quick exact lookup.
    PathFragment includeDir = input.getRootRelativePath().getParentDirectory();
    if (includeDir.segmentCount() == 0 || declaredIncludeDirs.contains(includeDir)) {
      return true;  // OK: quick exact match.
    }
    // Not found in the quick lookup: try the wildcards.
    for (PathFragment declared : declaredIncludeDirs) {
      if (declared.getBaseName().equals("**")) {
        if (includeDir.startsWith(declared.getParentDirectory())) {
          return true;  // OK: under a wildcard dir.
        }
      }
    }
    // Still not found: see if it is in a subdir of a declared package.
    Path root = input.getRoot().getPath();
    for (Path dir = input.getPath().getParentDirectory();;) {
      if (dir.getRelative("BUILD").exists()) {
        return false;  // Bad: this is a sub-package, not a subdir of a declared package.
      }
      dir = dir.getParentDirectory();
      if (dir.equals(root)) {
        return false;  // Bad: at the top, give up.
      }
      if (declaredIncludeDirs.contains(dir.relativeTo(root))) {
        return true;  // OK: found under a declared dir.
      }
    }
  }

  /**
   * Recalculates this action's live input collection, including sources, middlemen.
   *
   * @throws ActionExecutionException iff any errors happen during update.
   */
  @VisibleForTesting
  @ThreadCompatible
  public final synchronized void updateActionInputs(NestedSet<Artifact> discoveredInputs)
      throws ActionExecutionException {
    inputsKnown = false;
    NestedSetBuilder<Artifact> inputs = NestedSetBuilder.stableOrder();
    Profiler.instance().startTask(ProfilerTask.ACTION_UPDATE, this);
    try {
      inputs.addTransitive(mandatoryInputs);
      if (optionalSourceFile != null) {
        inputs.add(optionalSourceFile);
      }
      inputs.addAll(context.getTransitiveCompilationPrerequisites());
      inputs.addTransitive(discoveredInputs);
      inputsKnown = true;
    } finally {
      Profiler.instance().completeTask(ProfilerTask.ACTION_UPDATE);
      synchronized (this) {
        setInputs(inputs.build());
      }
    }
  }

  @Override
  public Iterable<Artifact> resolveInputsFromCache(
      ArtifactResolver artifactResolver,
      PackageRootResolver resolver,
      Collection<PathFragment> inputPaths)
      throws PackageRootResolutionException, InterruptedException {
    // Note that this method may trigger a violation of the desirable invariant that getInputs()
    // is a superset of getMandatoryInputs(). See bug about an "action not in canonical form"
    // error message and the integration test test_crosstool_change_and_failure().
    Map<PathFragment, Artifact> allowedDerivedInputsMap = getAllowedDerivedInputsMap();
    List<Artifact> inputs = new ArrayList<>();
    List<PathFragment> unresolvedPaths = new ArrayList<>();
    for (PathFragment execPath : inputPaths) {
      Artifact artifact = allowedDerivedInputsMap.get(execPath);
      if (artifact != null) {
        inputs.add(artifact);
      } else {
        // Remember this execPath, we will try to resolve it as a source artifact.
        unresolvedPaths.add(execPath);
      }
    }

    Map<PathFragment, Artifact> resolvedArtifacts =
        artifactResolver.resolveSourceArtifacts(unresolvedPaths, resolver);
    if (resolvedArtifacts == null) {
      // We are missing some dependencies. We need to rerun this update later.
      return null;
    }

    for (PathFragment execPath : unresolvedPaths) {
      Artifact artifact = resolvedArtifacts.get(execPath);
      // If PathFragment cannot be resolved into the artifact - ignore it. This could happen if
      // rule definition has changed and action no longer depends on, e.g., additional source file
      // in the separate package and that package is no longer referenced anywhere else.
      // It is safe to ignore such paths because dependency checker would identify change in inputs
      // (ignored path was used before) and will force action execution.
      if (artifact != null) {
        inputs.add(artifact);
      }
    }
    return inputs;
  }

  @Override
  public synchronized void updateInputs(Iterable<Artifact> inputs) {
    inputsKnown = true;
    synchronized (this) {
      setInputs(inputs);
    }
  }

  protected Map<PathFragment, Artifact> getAllowedDerivedInputsMap() {
    Map<PathFragment, Artifact> allowedDerivedInputMap = new HashMap<>();
    addToMap(allowedDerivedInputMap, mandatoryInputs);
    addToMap(allowedDerivedInputMap, getDeclaredIncludeSrcs());
    addToMap(allowedDerivedInputMap, context.getTransitiveCompilationPrerequisites());
    Artifact artifact = getSourceFile();
    if (!artifact.isSourceArtifact()) {
      allowedDerivedInputMap.put(artifact.getExecPath(), artifact);
    }
    return allowedDerivedInputMap;
  }

  private void addToMap(Map<PathFragment, Artifact> map, Iterable<Artifact> artifacts) {
    for (Artifact artifact : artifacts) {
      if (!artifact.isSourceArtifact()) {
        map.put(artifact.getExecPath(), artifact);
      }
    }
  }

  @Override
  protected String getRawProgressMessage() {
    return "Compiling " + getSourceFile().prettyPrint();
  }

  /**
   * Return the directories in which to look for headers (pertains to headers
   * not specifically listed in {@code declaredIncludeSrcs}). The return value
   * may contain duplicate elements.
   */
  public NestedSet<PathFragment> getDeclaredIncludeDirs() {
    return context.getDeclaredIncludeDirs();
  }

  /**
   * Return the directories in which to look for headers and issue a warning.
   * (pertains to headers not specifically listed in {@code
   * declaredIncludeSrcs}). The return value may contain duplicate elements.
   */
  public NestedSet<PathFragment> getDeclaredIncludeWarnDirs() {
    return context.getDeclaredIncludeWarnDirs();
  }

  /**
   * Return explicit header files (i.e., header files explicitly listed). The
   * return value may contain duplicate elements.
   */
  @Override
  public NestedSet<Artifact> getDeclaredIncludeSrcs() {
    if (lipoScannables != null && lipoScannables.iterator().hasNext()) {
      NestedSetBuilder<Artifact> srcs = NestedSetBuilder.stableOrder();
      srcs.addTransitive(context.getDeclaredIncludeSrcs());
      for (IncludeScannable lipoScannable : lipoScannables) {
        srcs.addTransitive(lipoScannable.getDeclaredIncludeSrcs());
      }
      return srcs.build();
    }
    return context.getDeclaredIncludeSrcs();
  }

  @Override
  public ResourceSet estimateResourceConsumption(Executor executor) {
    return executor.getContext(actionContext).estimateResourceConsumption(this);
  }

  @VisibleForTesting
  public Class<? extends CppCompileActionContext> getActionContext() {
    return actionContext;
  }

  /**
   * Estimate resource consumption when this action is executed locally.
   */
  public ResourceSet estimateResourceConsumptionLocal() {
    // We use a local compile, so much of the time is spent waiting for IO,
    // but there is still significant CPU; hence we estimate 50% cpu usage.
    return ResourceSet.createWithRamCpuIo(/*memoryMb=*/200, /*cpuUsage=*/0.5, /*ioUsage=*/0.0);
  }

  @Override
  public String computeKey() {
    Fingerprint f = new Fingerprint();
    f.addUUID(actionClassId);
    f.addStringMap(getEnvironment());
    f.addStrings(getArgv());
    f.addStrings(executionRequirements);

    /*
     * getArgv() above captures all changes which affect the compilation
     * command and hence the contents of the object file.  But we need to
     * also make sure that we reexecute the action if any of the fields
     * that affect whether validateIncludes() will report an error or warning
     * have changed, otherwise we might miss some errors.
     */
    f.addPaths(context.getDeclaredIncludeDirs());
    f.addPaths(context.getDeclaredIncludeWarnDirs());
    for (Artifact declaredIncludeSrc : context.getDeclaredIncludeSrcs()) {
      f.addPath(declaredIncludeSrc.getExecPath());
    }
    f.addInt(0);  // mark the boundary between input types
    for (Artifact input : getMandatoryInputs()) {
      f.addPath(input.getExecPath());
    }
    return f.hexDigestAndReset();
  }

  @Override
  @ThreadCompatible
  public void execute(
      ActionExecutionContext actionExecutionContext)
          throws ActionExecutionException, InterruptedException {
    Executor executor = actionExecutionContext.getExecutor();
    CppCompileActionContext.Reply reply;
    try {
      reply = executor.getContext(actionContext).execWithReply(this, actionExecutionContext);
    } catch (ExecException e) {
      throw e.toActionExecutionException("C++ compilation of rule '" + getOwner().getLabel() + "'",
          executor.getVerboseFailures(), this);
    }
    ensureCoverageNotesFilesExist();

    // This is the .d file scanning part.
    IncludeScanningContext scanningContext = executor.getContext(IncludeScanningContext.class);
    Path execRoot = executor.getExecRoot();

    NestedSet<Artifact> discoveredInputs =
        discoverInputsFromDotdFiles(execRoot, scanningContext.getArtifactResolver(), reply);
    reply = null; // Clear in-memory .d files early.

    // Post-execute "include scanning", which modifies the action inputs to match what the compile
    // action actually used by incorporating the results of .d file parsing.
    //
    // We enable this when "include scanning" itself is enabled, or when hdrs_check is set to loose
    // or warn, as otherwise the action might be missing inputs that the compiler used and rebuilds
    // become incorrect.
    //
    // Note that this effectively disables post-execute "include scanning" in Bazel, because
    // hdrs_check is forced to "strict" and "include scanning" is forced to off.
    boolean usesStrictHdrsChecks = context.getDeclaredIncludeDirs().isEmpty()
        && context.getDeclaredIncludeWarnDirs().isEmpty();
    if (shouldScanIncludes() || !usesStrictHdrsChecks) {
      updateActionInputs(discoveredInputs);
    }

    // hdrs_check: This cannot be switched off, because doing so would allow for incorrect builds.
    validateInclusions(
        discoveredInputs, actionExecutionContext.getArtifactExpander(), executor.getEventHandler());
  }

  @VisibleForTesting
  public NestedSet<Artifact> discoverInputsFromDotdFiles(
      Path execRoot, ArtifactResolver artifactResolver, Reply reply)
      throws ActionExecutionException {
    if (getDotdFile() == null) {
      return NestedSetBuilder.<Artifact>stableOrder().build();
    }
    HeaderDiscovery.Builder discoveryBuilder =
        new HeaderDiscovery.Builder()
            .setAction(this)
            .setDotdFile(getDotdFile())
            .setSourceFile(getSourceFile())
            .setSpecialInputsHandler(specialInputsHandler)
            .setDependencySet(processDepset(execRoot, reply))
            .setPermittedSystemIncludePrefixes(getPermittedSystemIncludePrefixes(execRoot))
            .setAllowedDerivedinputsMap(getAllowedDerivedInputsMap());

    if (cppSemantics.needsIncludeValidation()) {
      discoveryBuilder.shouldValidateInclusions();
    }

    return discoveryBuilder.build().discoverInputsFromDotdFiles(execRoot, artifactResolver);
  }

  public DependencySet processDepset(Path execRoot, Reply reply) throws ActionExecutionException {
    try {
      DotdFile dotdFile = getDotdFile();
      Preconditions.checkNotNull(dotdFile);
      DependencySet depSet = new DependencySet(execRoot);
      // artifact() is null if we are using in-memory .d files. We also want to prepare for the
      // case where we expected an in-memory .d file, but we did not get an appropriate response.
      // Perhaps we produced the file locally.
      if (dotdFile.artifact() != null || reply == null) {
        return depSet.read(dotdFile.getPath());
      } else {
        // This is an in-memory .d file.
        return depSet.process(reply.getContents());
      }
    } catch (IOException e) {
      // Some kind of IO or parse exception--wrap & rethrow it to stop the build.
      throw new ActionExecutionException("error while parsing .d file", e, this, false);
    }
  }

  public List<Path> getPermittedSystemIncludePrefixes(Path execRoot) {
    CppConfiguration toolchain = cppConfiguration;
    List<Path> systemIncludePrefixes = new ArrayList<>();
    for (PathFragment includePath : toolchain.getBuiltInIncludeDirectories()) {
      if (includePath.isAbsolute()) {
        systemIncludePrefixes.add(execRoot.getFileSystem().getPath(includePath));
      }
    }
    return systemIncludePrefixes;
  }

  /**
   * Gcc only creates ".gcno" files if the compilation unit is non-empty.
   * To ensure that the set of outputs for a CppCompileAction remains consistent
   * and doesn't vary dynamically depending on the _contents_ of the input files,
   * we create empty ".gcno" files if gcc didn't create them.
   */
  private void ensureCoverageNotesFilesExist() throws ActionExecutionException {
    for (Artifact output : getOutputs()) {
      if (CppFileTypes.COVERAGE_NOTES.matches(output.getFilename()) // ".gcno"
          && !output.getPath().exists()) {
        try {
          FileSystemUtils.createEmptyFile(output.getPath());
        } catch (IOException e) {
          throw new ActionExecutionException(
              "Error creating file '" + output.getPath() + "': " + e.getMessage(), e, this, false);
        }
      }
    }
  }

  /**
   * Provides list of include files needed for performing extra actions on this action when run
   * remotely. The list of include files is created by performing a header scan on the known input
   * files.
   */
  @Override
  public Iterable<Artifact> getInputFilesForExtraAction(
      ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException, InterruptedException {
    Collection<Artifact> scannedIncludes =
        actionExecutionContext.getExecutor().getContext(actionContext)
        .getScannedIncludeFiles(this, actionExecutionContext);
    // Use a set to eliminate duplicates.
    ImmutableSet.Builder<Artifact> result = ImmutableSet.builder();
    return result.addAll(getInputs()).addAll(scannedIncludes).build();
  }

  @Override
  public String getMnemonic() { return "CppCompile"; }

  @Override
  public String describeKey() {
    StringBuilder message = new StringBuilder();
    message.append(getProgressMessage());
    message.append('\n');
    // Outputting one argument per line makes it easier to diff the results.
    // The first element in getArgv() is actually the command to execute.
    String legend = "  Command: ";
    for (String argument : ShellEscaper.escapeAll(getArgv())) {
      message.append(legend);
      message.append(argument);
      message.append('\n');
      legend = "  Argument: ";
    }

    for (PathFragment path : context.getDeclaredIncludeDirs()) {
      message.append("  Declared include directory: ");
      message.append(ShellEscaper.escapeString(path.getPathString()));
      message.append('\n');
    }

    for (Artifact src : getDeclaredIncludeSrcs()) {
      message.append("  Declared include source: ");
      message.append(ShellEscaper.escapeString(src.getExecPathString()));
      message.append('\n');
    }

    return message.toString();
  }

  /**
   * The compile command line for the enclosing C++ compile action.
   */
  public final class CppCompileCommandLine {
    private final Artifact sourceFile;
    private final DotdFile dotdFile;
    private final List<String> copts;
    private final Predicate<String> coptsFilter;
    private final Collection<String> features;
    @VisibleForTesting public final CcToolchainFeatures.Variables variables;
    private final String actionName;

    public CppCompileCommandLine(
        Artifact sourceFile,
        DotdFile dotdFile,
        ImmutableList<String> copts,
        Predicate<String> coptsFilter,
        Collection<String> features,
        CcToolchainFeatures.Variables variables,
        String actionName) {
      this.sourceFile = Preconditions.checkNotNull(sourceFile);
      this.dotdFile = CppFileTypes.mustProduceDotdFile(sourceFile.getPath().toString())
                      ? Preconditions.checkNotNull(dotdFile) : null;
      this.copts = Preconditions.checkNotNull(copts);
      this.coptsFilter = coptsFilter;
      this.features = Preconditions.checkNotNull(features);
      this.variables = variables;
      this.actionName = actionName;
    }

    /**
     * Returns the environment variables that should be set for C++ compile actions.
     */
    protected Map<String, String> getEnvironment() {
      return featureConfiguration.getEnvironmentVariables(actionName, variables);
    }

    protected List<String> getArgv(
        PathFragment outputFile, CcToolchainFeatures.Variables overwrittenVariables) {
      List<String> commandLine = new ArrayList<>();

      // first: The command name.
      if (!featureConfiguration.actionIsConfigured(actionName)) {
        commandLine.add(cppConfiguration.getToolPathFragment(Tool.GCC).getPathString());
      } else {
        commandLine.add(
            featureConfiguration
                .getToolForAction(actionName)
                .getToolPath(cppConfiguration.getCrosstoolTopPathFragment())
                .getPathString());
      }

      // second: The compiler options.
      commandLine.addAll(getCompilerOptions(overwrittenVariables));

      if (!featureConfiguration.isEnabled("compile_action_flags_in_flag_set")) {
        // third: The file to compile!
        commandLine.add("-c");
        commandLine.add(sourceFile.getExecPathString());

        // finally: The output file. (Prefixed with -o).
        commandLine.add("-o");
        commandLine.add(outputFile.getPathString());
      }

      return commandLine;
    }

    public List<String> getCompilerOptions(
        @Nullable CcToolchainFeatures.Variables overwrittenVariables) {
      List<String> options = new ArrayList<>();
      CppConfiguration toolchain = cppConfiguration;

      addFilteredOptions(options, toolchain.getCompilerOptions(features));

      String sourceFilename = sourceFile.getExecPathString();
      if (CppFileTypes.C_SOURCE.matches(sourceFilename)) {
        addFilteredOptions(options, toolchain.getCOptions());
      }
      if (CppFileTypes.CPP_SOURCE.matches(sourceFilename)
          || CppFileTypes.CPP_HEADER.matches(sourceFilename)
          || CppFileTypes.CPP_MODULE_MAP.matches(sourceFilename)
          || CppFileTypes.CLIF_INPUT_PROTO.matches(sourceFilename)) {
        addFilteredOptions(options, toolchain.getCxxOptions(features));
      }

      // TODO(bazel-team): This needs to be before adding getUnfilteredCompilerOptions() and after
      // adding the warning flags until all toolchains are migrated; currently toolchains use the
      // unfiltered compiler options to inject include paths, which is superseded by the feature
      // configuration; on the other hand toolchains switch off warnings for the layering check
      // that will be re-added by the feature flags.
      CcToolchainFeatures.Variables updatedVariables = variables;
      if (overwrittenVariables != null) {
        CcToolchainFeatures.Variables.Builder variablesBuilder =
            new CcToolchainFeatures.Variables.Builder();
        variablesBuilder.addAll(variables);
        variablesBuilder.addAll(overwrittenVariables);
        updatedVariables = variablesBuilder.build();
      }
      addFilteredOptions(
          options, featureConfiguration.getCommandLine(actionName, updatedVariables));

      // Users don't expect the explicit copts to be filtered by coptsFilter, add them verbatim.
      // Make sure these are added after the options from the feature configuration, so that
      // those options can be overriden.
      options.addAll(copts);

      // Unfiltered compiler options contain system include paths. These must be added after
      // the user provided options, otherwise users adding include paths will not pick up their
      // own include paths first.
      options.addAll(toolchain.getUnfilteredCompilerOptions(features));

      // Add the options of --per_file_copt, if the label or the base name of the source file
      // matches the specified regular expression filter.
      for (PerLabelOptions perLabelOptions : cppConfiguration.getPerFileCopts()) {
        if ((sourceLabel != null && perLabelOptions.isIncluded(sourceLabel))
            || perLabelOptions.isIncluded(sourceFile)) {
          options.addAll(perLabelOptions.getOptions());
        }
      }

      if (!featureConfiguration.isEnabled("compile_action_flags_in_flag_set")) {
        if (FileType.contains(outputFile, CppFileTypes.ASSEMBLER, CppFileTypes.PIC_ASSEMBLER)) {
          options.add("-S");
        } else if (FileType.contains(outputFile, CppFileTypes.PREPROCESSED_C,
            CppFileTypes.PREPROCESSED_CPP, CppFileTypes.PIC_PREPROCESSED_C,
            CppFileTypes.PIC_PREPROCESSED_CPP)) {
          options.add("-E");
        }
      }

      return options;
    }

    // For each option in 'in', add it to 'out' unless it is matched by the 'coptsFilter' regexp.
    private void addFilteredOptions(List<String> out, List<String> in) {
      Iterables.addAll(out, Iterables.filter(in, coptsFilter));
    }
  }

  /**
   * A reference to a .d file. There are two modes:
   * <ol>
   *   <li>an Artifact that represents a real on-disk file
   *   <li>just an execPath that refers to a virtual .d file that is not written to disk
   * </ol>
   */
  public static class DotdFile {
    private final Artifact artifact;
    private final PathFragment execPath;

    public DotdFile(Artifact artifact) {
      this.artifact = artifact;
      this.execPath = null;
    }

    public DotdFile(PathFragment execPath) {
      this.artifact = null;
      this.execPath = execPath;
    }

    /**
     * @return the Artifact or null
     */
    public Artifact artifact() {
      return artifact;
    }

    /**
     * @return Gets the execPath regardless of whether this is a real Artifact
     */
    public PathFragment getSafeExecPath() {
      return execPath == null ? artifact.getExecPath() : execPath;
    }

    /**
     * @return the on-disk location of the .d file or null
     */
    public Path getPath() {
      return artifact.getPath();
    }
  }
}
