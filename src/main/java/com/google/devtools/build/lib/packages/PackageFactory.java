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

package com.google.devtools.build.lib.packages;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.cmdline.LabelValidator;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.concurrent.NamedForkJoinPool;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.events.ExtendedEventHandler.Postable;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.Globber.BadGlobException;
import com.google.devtools.build.lib.packages.License.DistributionType;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.packages.RuleFactory.BuildLangTypedAttributeValuesMap;
import com.google.devtools.build.lib.syntax.Argument;
import com.google.devtools.build.lib.syntax.BaseFunction;
import com.google.devtools.build.lib.syntax.ClassObject;
import com.google.devtools.build.lib.syntax.DefStatement;
import com.google.devtools.build.lib.syntax.Dict;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.Expression;
import com.google.devtools.build.lib.syntax.ForStatement;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.FunctionSignature;
import com.google.devtools.build.lib.syntax.Identifier;
import com.google.devtools.build.lib.syntax.IfStatement;
import com.google.devtools.build.lib.syntax.IntegerLiteral;
import com.google.devtools.build.lib.syntax.ListExpression;
import com.google.devtools.build.lib.syntax.Module;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.syntax.Node;
import com.google.devtools.build.lib.syntax.NodeVisitor;
import com.google.devtools.build.lib.syntax.NoneType;
import com.google.devtools.build.lib.syntax.ParserInput;
import com.google.devtools.build.lib.syntax.Printer;
import com.google.devtools.build.lib.syntax.Starlark;
import com.google.devtools.build.lib.syntax.StarlarkFile;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import com.google.devtools.build.lib.syntax.StarlarkThread;
import com.google.devtools.build.lib.syntax.StarlarkThread.Extension;
import com.google.devtools.build.lib.syntax.Statement;
import com.google.devtools.build.lib.syntax.StringLiteral;
import com.google.devtools.build.lib.syntax.Tuple;
import com.google.devtools.build.lib.syntax.ValidationEnvironment;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.UnixGlob;
import java.io.IOException;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicReference;
import java.util.logging.Logger;
import javax.annotation.Nullable;

/**
 * The package factory is responsible for constructing Package instances
 * from a BUILD file's abstract syntax tree (AST).
 *
 * <p>A PackageFactory is a heavy-weight object; create them sparingly.
 * Typically only one is needed per client application.
 */
public final class PackageFactory {
  /**
   * An argument to the {@code package()} function.
   */
  public abstract static class PackageArgument<T> {
    private final String name;
    private final Type<T> type;

    protected PackageArgument(String name, Type<T> type) {
      this.name = name;
      this.type = type;
    }

    public String getName() {
      return name;
    }

    private void convertAndProcess(
        Package.Builder pkgBuilder, Location location, Object value)
        throws EvalException {
      T typedValue = type.convert(value, "'package' argument", pkgBuilder.getBuildFileLabel());
      process(pkgBuilder, location, typedValue);
    }

    /**
     * Process an argument.
     *
     * @param pkgBuilder the package builder to be mutated
     * @param location the location of the {@code package} function for error reporting
     * @param value the value of the argument. Typically passed to {@link Type#convert}
     */
    protected abstract void process(
        Package.Builder pkgBuilder, Location location, T value)
        throws EvalException;
  }

  /** An extension to the global namespace of the BUILD language. */
  // TODO(bazel-team): this is largely unrelated to syntax.StarlarkThread.Extension,
  // and should probably be renamed PackageFactory.RuntimeExtension, since really,
  // we're extending the Runtime with more classes.
  public interface EnvironmentExtension {
    /** Update the predeclared environment with the identifiers this extension contributes. */
    void update(ImmutableMap.Builder<String, Object> env);

    /** Update the predeclared environment of WORKSPACE files. */
    void updateWorkspace(ImmutableMap.Builder<String, Object> env);

    /** Update the environment of the native module. */
    void updateNative(ImmutableMap.Builder<String, Object> env);

    /**
     * Returns the extra arguments to the {@code package()} statement.
     */
    Iterable<PackageArgument<?>> getPackageArguments();
  }

  private static class DefaultVisibility extends PackageArgument<List<Label>> {
    private DefaultVisibility() {
      super("default_visibility", BuildType.LABEL_LIST);
    }

    @Override
    protected void process(Package.Builder pkgBuilder, Location location,
        List<Label> value) throws EvalException{
      try {
        pkgBuilder.setDefaultVisibility(getVisibility(pkgBuilder.getBuildFileLabel(), value));
      } catch (EvalException e) {
        throw new EvalException(location, e.getMessage());
      }
    }
  }

  private static class DefaultTestOnly extends PackageArgument<Boolean> {
    private DefaultTestOnly() {
      super("default_testonly", Type.BOOLEAN);
    }

    @Override
    protected void process(Package.Builder pkgBuilder, Location location,
        Boolean value) {
      pkgBuilder.setDefaultTestonly(value);
    }
  }

  private static class DefaultDeprecation extends PackageArgument<String> {
    private DefaultDeprecation() {
      super("default_deprecation", Type.STRING);
    }

    @Override
    protected void process(Package.Builder pkgBuilder, Location location,
        String value) {
      pkgBuilder.setDefaultDeprecation(value);
    }
  }

  private static class Features extends PackageArgument<List<String>> {
    private Features() {
      super("features", Type.STRING_LIST);
    }

    @Override
    protected void process(Package.Builder pkgBuilder, Location location,
        List<String> value) {
      pkgBuilder.addFeatures(value);
    }
  }

  private static class DefaultLicenses extends PackageArgument<License> {
    private DefaultLicenses() {
      super("licenses", BuildType.LICENSE);
    }

    @Override
    protected void process(Package.Builder pkgBuilder, Location location,
        License value) {
      pkgBuilder.setDefaultLicense(value);
    }
  }

  private static class DefaultDistribs extends PackageArgument<Set<DistributionType>> {
    private DefaultDistribs() {
      super("distribs", BuildType.DISTRIBUTIONS);
    }

    @Override
    protected void process(Package.Builder pkgBuilder, Location location,
        Set<DistributionType> value) {
      pkgBuilder.setDefaultDistribs(value);
    }
  }

  /**
   * Declares the package() attribute specifying the default value for {@link
   * com.google.devtools.build.lib.packages.RuleClass#COMPATIBLE_ENVIRONMENT_ATTR} when not
   * explicitly specified.
   */
  private static class DefaultCompatibleWith extends PackageArgument<List<Label>> {
    private static final String DEFAULT_COMPATIBLE_WITH_ATTRIBUTE = "default_compatible_with";

    private DefaultCompatibleWith() {
      super(DEFAULT_COMPATIBLE_WITH_ATTRIBUTE, BuildType.LABEL_LIST);
    }

    @Override
    protected void process(Package.Builder pkgBuilder, Location location,
        List<Label> value) {
      pkgBuilder.setDefaultCompatibleWith(value, DEFAULT_COMPATIBLE_WITH_ATTRIBUTE, location);
    }
  }

  /**
   * Declares the package() attribute specifying the default value for {@link
   * com.google.devtools.build.lib.packages.RuleClass#RESTRICTED_ENVIRONMENT_ATTR} when not
   * explicitly specified.
   */
  private static class DefaultRestrictedTo extends PackageArgument<List<Label>> {
    private static final String DEFAULT_RESTRICTED_TO_ATTRIBUTE = "default_restricted_to";

    private DefaultRestrictedTo() {
      super(DEFAULT_RESTRICTED_TO_ATTRIBUTE, BuildType.LABEL_LIST);
    }

    @Override
    protected void process(Package.Builder pkgBuilder, Location location,
        List<Label> value) {
      pkgBuilder.setDefaultRestrictedTo(value, DEFAULT_RESTRICTED_TO_ATTRIBUTE, location);
    }
  }

  /** {@link Globber} that uses the legacy GlobCache. */
  public static class LegacyGlobber implements Globber {
    private final GlobCache globCache;
    private final boolean sort;

    private LegacyGlobber(GlobCache globCache, boolean sort) {
      this.globCache = globCache;
      this.sort = sort;
    }

    private static class Token extends Globber.Token {
      public final List<String> includes;
      public final List<String> excludes;
      public final boolean excludeDirs;
      public final boolean allowEmpty;

      public Token(
          List<String> includes, List<String> excludes, boolean excludeDirs, boolean allowEmpty) {
        this.includes = includes;
        this.excludes = excludes;
        this.excludeDirs = excludeDirs;
        this.allowEmpty = allowEmpty;
      }
    }

    @Override
    public Token runAsync(
        List<String> includes, List<String> excludes, boolean excludeDirs, boolean allowEmpty)
        throws BadGlobException {
      for (String pattern : includes) {
        @SuppressWarnings("unused")
        Future<?> possiblyIgnoredError = globCache.getGlobUnsortedAsync(pattern, excludeDirs);
      }
      return new Token(includes, excludes, excludeDirs, allowEmpty);
    }

    @Override
    public List<String> fetch(Globber.Token token)
        throws BadGlobException, IOException, InterruptedException {
      List<String> result;
      Token legacyToken = (Token) token;
      result =
          globCache.globUnsorted(
              legacyToken.includes,
              legacyToken.excludes,
              legacyToken.excludeDirs,
              legacyToken.allowEmpty);
      if (sort) {
        Collections.sort(result);
      }
      return result;
    }

    @Override
    public void onInterrupt() {
      globCache.cancelBackgroundTasks();
    }

    @Override
    public void onCompletion() {
      globCache.finishBackgroundTasks();
    }
  }

  private static final Logger logger = Logger.getLogger(PackageFactory.class.getName());

  private final RuleFactory ruleFactory;
  private final ImmutableMap<String, BuiltinRuleFunction> ruleFunctions;
  private final RuleClassProvider ruleClassProvider;

  private AtomicReference<? extends UnixGlob.FilesystemCalls> syscalls;

  private ForkJoinPool executor;

  private int maxDirectoriesToEagerlyVisitInGlobbing;

  private final ImmutableList<EnvironmentExtension> environmentExtensions;
  private final ImmutableMap<String, PackageArgument<?>> packageArguments;

  private final Package.Builder.Helper packageBuilderHelper;

  /** Builder for {@link PackageFactory} instances. Intended to only be used by unit tests. */
  @VisibleForTesting
  public abstract static class BuilderForTesting {
    protected final String version = "test";
    protected Iterable<EnvironmentExtension> environmentExtensions = ImmutableList.of();
    protected boolean doChecksForTesting = true;

    public BuilderForTesting setEnvironmentExtensions(
        Iterable<EnvironmentExtension> environmentExtensions) {
      this.environmentExtensions = environmentExtensions;
      return this;
    }

    public BuilderForTesting disableChecks() {
      this.doChecksForTesting = false;
      return this;
    }

    public abstract PackageFactory build(RuleClassProvider ruleClassProvider, FileSystem fs);
  }

  @VisibleForTesting
  public Package.Builder.Helper getPackageBuilderHelperForTesting() {
    return packageBuilderHelper;
  }

  /**
   * Constructs a {@code PackageFactory} instance with a specific glob path translator
   * and rule factory.
   *
   * <p>Only intended to be called by BlazeRuntime or {@link BuilderForTesting#build}.
   *
   * <p>Do not call this constructor directly in tests; please use
   * TestConstants#PACKAGE_FACTORY_BUILDER_FACTORY_FOR_TESTING instead.
   */
  public PackageFactory(
      RuleClassProvider ruleClassProvider,
      Iterable<EnvironmentExtension> environmentExtensions,
      String version,
      Package.Builder.Helper packageBuilderHelper) {
    this.ruleFactory = new RuleFactory(ruleClassProvider);
    this.ruleFunctions = buildRuleFunctions(ruleFactory);
    this.ruleClassProvider = ruleClassProvider;
    setGlobbingThreads(100);
    this.environmentExtensions = ImmutableList.copyOf(environmentExtensions);
    this.packageArguments = createPackageArguments();
    this.nativeModule = newNativeModule();
    this.workspaceNativeModule = WorkspaceFactory.newNativeModule(ruleClassProvider, version);
    this.packageBuilderHelper = packageBuilderHelper;
  }

 /**
   * Sets the syscalls cache used in globbing.
   */
  public void setSyscalls(AtomicReference<? extends UnixGlob.FilesystemCalls> syscalls) {
    this.syscalls = Preconditions.checkNotNull(syscalls);
  }

  /**
   * Sets the max number of threads to use for globbing.
   */
  public void setGlobbingThreads(int globbingThreads) {
    if (executor == null || executor.getParallelism() != globbingThreads) {
      executor = NamedForkJoinPool.newNamedPool("globbing pool", globbingThreads);
    }
  }

  /**
   * Sets the number of directories to eagerly traverse on the first glob for a given package, in
   * order to warm the filesystem. -1 means do no eager traversal. See {@code
   * PackageCacheOptions#maxDirectoriesToEagerlyVisitInGlobbing}. -2 means do the eager traversal
   * using the regular globbing infrastructure, i.e. sharing the globbing threads and caching the
   * actual glob results.
   */
  public void setMaxDirectoriesToEagerlyVisitInGlobbing(
      int maxDirectoriesToEagerlyVisitInGlobbing) {
    this.maxDirectoriesToEagerlyVisitInGlobbing = maxDirectoriesToEagerlyVisitInGlobbing;
  }

  /**
   * Returns the immutable, unordered set of names of all the known rule
   * classes.
   */
  public Set<String> getRuleClassNames() {
    return ruleFactory.getRuleClassNames();
  }

  /**
   * Returns the {@link com.google.devtools.build.lib.packages.RuleClass} for the specified rule
   * class name.
   */
  public RuleClass getRuleClass(String ruleClassName) {
    return ruleFactory.getRuleClass(ruleClassName);
  }

  /**
   * Returns the {@link RuleClassProvider} of this {@link PackageFactory}.
   */
  public RuleClassProvider getRuleClassProvider() {
    return ruleClassProvider;
  }

  public ImmutableList<EnvironmentExtension> getEnvironmentExtensions() {
    return environmentExtensions;
  }

  /**
   * Creates the list of arguments for the 'package' function.
   */
  private ImmutableMap<String, PackageArgument<?>> createPackageArguments() {
    ImmutableList.Builder<PackageArgument<?>> arguments =
        ImmutableList.<PackageArgument<?>>builder()
            .add(new DefaultDeprecation())
            .add(new DefaultDistribs())
            .add(new DefaultLicenses())
            .add(new DefaultTestOnly())
            .add(new DefaultVisibility())
            .add(new Features())
            .add(new DefaultCompatibleWith())
            .add(new DefaultRestrictedTo());

    for (EnvironmentExtension extension : environmentExtensions) {
      arguments.addAll(extension.getPackageArguments());
    }

    ImmutableMap.Builder<String, PackageArgument<?>> packageArguments = ImmutableMap.builder();
    for (PackageArgument<?> argument : arguments.build()) {
      packageArguments.put(argument.getName(), argument);
    }
    return packageArguments.build();
  }

  static class NotRepresentableException extends EvalException {
    NotRepresentableException(String msg) {
      super(null, msg);
    }
  };

  public static RuleVisibility getVisibility(Label ruleLabel, List<Label> original)
      throws EvalException {
    RuleVisibility result;

    result = ConstantRuleVisibility.tryParse(original);
    if (result != null) {
      return result;
    }

    result = PackageGroupsRuleVisibility.tryParse(ruleLabel, original);
    return result;
  }

  /** Returns a function-value implementing "package" in the specified package context. */
  // TODO(cparsons): Migrate this function to be defined with @SkylarkCallable.
  // TODO(adonovan): don't call this function twice (once for BUILD files and
  // once for the native module) as it results in distinct objects. (Using
  // @SkylarkCallable may accomplish that.)
  private static BaseFunction newPackageFunction(
      final ImmutableMap<String, PackageArgument<?>> packageArguments) {
    FunctionSignature signature =
        FunctionSignature.namedOnly(0, packageArguments.keySet().toArray(new String[0]));

    return new BaseFunction() {
      @Override
      public String getName() {
        return "package";
      }

      @Override
      public FunctionSignature getSignature() {
        return signature; // (only for documentation)
      }

      @Override
      public Object call(
          StarlarkThread thread, Location loc, Tuple<Object> args, Dict<String, Object> kwargs)
          throws EvalException {
        if (!args.isEmpty()) {
          throw new EvalException(null, "unexpected positional arguments");
        }
        Package.Builder pkgBuilder = getContext(thread, loc).pkgBuilder;

        // Validate parameter list
        if (pkgBuilder.isPackageFunctionUsed()) {
          throw new EvalException(null, "'package' can only be used once per BUILD file");
        }
        pkgBuilder.setPackageFunctionUsed();

        // Each supplied argument must name a PackageArgument.
        if (kwargs.isEmpty()) {
          throw new EvalException(
              null, "at least one argument must be given to the 'package' function");
        }
        for (Map.Entry<String, Object> kwarg : kwargs.entrySet()) {
          String name = kwarg.getKey();
          PackageArgument<?> pkgarg = packageArguments.get(name);
          if (pkgarg == null) {
            throw new EvalException(null, "unexpected keyword argument: " + name);
          }
          pkgarg.convertAndProcess(pkgBuilder, loc, kwarg.getValue());
        }
        return Starlark.NONE;
      }
    };
  }

  /** Get the PackageContext by looking up in the environment. */
  public static PackageContext getContext(StarlarkThread thread, Location location)
      throws EvalException {
    PackageContext value = thread.getThreadLocal(PackageContext.class);
    if (value == null) {
      // if PackageContext is missing, we're not called from a BUILD file. This happens if someone
      // uses native.some_func() in the wrong place.
      throw new EvalException(
          location,
          "The native module can be accessed only from a BUILD thread. "
              + "Wrap the function in a macro and call it from a BUILD file");
    }
    return value;
  }

  private static ImmutableMap<String, BuiltinRuleFunction> buildRuleFunctions(
      RuleFactory ruleFactory) {
    ImmutableMap.Builder<String, BuiltinRuleFunction> result = ImmutableMap.builder();
    for (String ruleClassName : ruleFactory.getRuleClassNames()) {
      RuleClass cl = ruleFactory.getRuleClass(ruleClassName);
      if (cl.getRuleClassType() == RuleClassType.NORMAL
          || cl.getRuleClassType() == RuleClassType.TEST) {
        result.put(ruleClassName, new BuiltinRuleFunction(cl));
      }
    }
    return result.build();
  }

  /**
   * {@link BaseFunction} adapter for creating {@link Rule}s for native {@link
   * com.google.devtools.build.lib.packages.RuleClass}es.
   */
  private static class BuiltinRuleFunction extends BaseFunction implements RuleFunction {
    private final RuleClass ruleClass;

    BuiltinRuleFunction(RuleClass ruleClass) {
      this.ruleClass = Preconditions.checkNotNull(ruleClass);
    }

    @Override
    public FunctionSignature getSignature() {
      return FunctionSignature.KWARGS; // just for documentation
    }

    @Override
    public NoneType call(
        StarlarkThread thread, Location loc, Tuple<Object> args, Dict<String, Object> kwargs)
        throws EvalException, InterruptedException {
      if (!args.isEmpty()) {
        throw new EvalException(null, "unexpected positional arguments");
      }
      BazelStarlarkContext.from(thread).checkLoadingOrWorkspacePhase(ruleClass.getName());
      try {
        addRule(getContext(thread, loc), kwargs, loc, thread);
      } catch (RuleFactory.InvalidRuleException | Package.NameConflictException e) {
        throw new EvalException(loc, e.getMessage());
      }
      return Starlark.NONE;
    }

    private void addRule(
        PackageContext context, Map<String, Object> kwargs, Location loc, StarlarkThread thread)
        throws RuleFactory.InvalidRuleException, Package.NameConflictException,
            InterruptedException {
      BuildLangTypedAttributeValuesMap attributeValues =
          new BuildLangTypedAttributeValuesMap(kwargs);
      RuleFactory.createAndAddRule(
          context, ruleClass, attributeValues, loc, thread, new AttributeContainer(ruleClass));
    }

    @Override
    public RuleClass getRuleClass() {
      return ruleClass;
    }

    @Override
    public String getName() {
      return ruleClass.getName();
    }

    @Override
    public void repr(Printer printer) {
      printer.append("<built-in rule " + getName() + ">");
    }
  }

  // Exposed to skyframe.PackageFunction.
  public static StarlarkFile parseBuildFile(
      PackageIdentifier packageId, ParserInput input, List<Statement> preludeStatements) {
    // Log messages are expected as signs of progress by a single very old test:
    // testCreatePackageIsolatedFromOuterErrors, see CL 6198296.
    // Removing them will cause it to time out. TODO(adonovan): clean this up.
    logger.fine("Starting to parse " + packageId);
    StarlarkFile file = StarlarkFile.parseWithPrelude(input, preludeStatements);
    logger.fine("Finished parsing of " + packageId);
    return file;
  }

  // Exposed to skyframe.PackageFunction.
  public Package.Builder createPackageFromAst(
      String workspaceName,
      ImmutableMap<RepositoryName, RepositoryName> repositoryMapping,
      PackageIdentifier packageId,
      RootedPath buildFile,
      StarlarkFile file,
      Map<String, Extension> imports,
      ImmutableList<Label> skylarkFileDependencies,
      RuleVisibility defaultVisibility,
      StarlarkSemantics starlarkSemantics,
      Globber globber)
      throws InterruptedException {
    try {
      // At this point the package is guaranteed to exist.  It may have parse or
      // evaluation errors, resulting in a diminished number of rules.
      return evaluateBuildFile(
          workspaceName,
          packageId,
          file,
          buildFile,
          globber,
          defaultVisibility,
          starlarkSemantics,
          imports,
          skylarkFileDependencies,
          repositoryMapping);
    } catch (InterruptedException e) {
      globber.onInterrupt();
      throw e;
    } finally {
      globber.onCompletion();
    }
  }

  @VisibleForTesting
  public Package.Builder newExternalPackageBuilder(
      RootedPath workspacePath, String runfilesPrefix, StarlarkSemantics starlarkSemantics) {
    return Package.newExternalPackageBuilder(
        packageBuilderHelper, workspacePath, runfilesPrefix, starlarkSemantics);
  }

  @VisibleForTesting
  public Package.Builder newPackageBuilder(
      PackageIdentifier packageId, String runfilesPrefix, StarlarkSemantics starlarkSemantics) {
    return new Package.Builder(packageBuilderHelper, packageId, runfilesPrefix, starlarkSemantics);
  }

  @VisibleForTesting
  public Package createPackageForTesting(
      PackageIdentifier packageId,
      RootedPath buildFile,
      CachingPackageLocator locator,
      ExtendedEventHandler eventHandler)
      throws NoSuchPackageException, InterruptedException {
    Package externalPkg =
        newExternalPackageBuilder(
                RootedPath.toRootedPath(
                    buildFile.getRoot(),
                    buildFile
                        .getRootRelativePath()
                        .getRelative(LabelConstants.WORKSPACE_FILE_NAME)),
                "TESTING",
                StarlarkSemantics.DEFAULT_SEMANTICS)
            .build();
    return createPackageForTesting(
        packageId,
        externalPkg,
        buildFile,
        locator,
        eventHandler,
        StarlarkSemantics.DEFAULT_SEMANTICS);
  }

  /**
   * Same as createPackage, but does the required validation of "packageName" first, throwing a
   * {@link NoSuchPackageException} if the name is invalid.
   */
  @VisibleForTesting
  public Package createPackageForTesting(
      PackageIdentifier packageId,
      Package externalPkg,
      RootedPath buildFile,
      CachingPackageLocator locator,
      ExtendedEventHandler eventHandler,
      StarlarkSemantics semantics)
      throws NoSuchPackageException, InterruptedException {
    String error =
        LabelValidator.validatePackageName(packageId.getPackageFragment().getPathString());
    if (error != null) {
      throw new BuildFileNotFoundException(
          packageId, "illegal package name: '" + packageId + "' (" + error + ")");
    }
    byte[] buildFileBytes = maybeGetBuildFileBytes(buildFile.asPath(), eventHandler);
    if (buildFileBytes == null) {
      throw new BuildFileContainsErrorsException(packageId, "IOException occurred");
    }

    Globber globber =
        createLegacyGlobber(
            buildFile.asPath().getParentDirectory(), packageId, ImmutableSet.of(), locator);
    ParserInput input =
        ParserInput.create(
            FileSystemUtils.convertFromLatin1(buildFileBytes), buildFile.asPath().asFragment());
    StarlarkFile file =
        parseBuildFile(packageId, input, /*preludeStatements=*/ ImmutableList.<Statement>of());
    Package result =
        createPackageFromAst(
                externalPkg.getWorkspaceName(),
                /*repositoryMapping=*/ ImmutableMap.of(),
                packageId,
                buildFile,
                file,
                /*imports=*/ ImmutableMap.<String, Extension>of(),
                /*skylarkFileDependencies=*/ ImmutableList.<Label>of(),
                /*defaultVisibility=*/ ConstantRuleVisibility.PUBLIC,
                semantics,
                globber)
            .build();
    for (Postable post : result.getPosts()) {
      eventHandler.post(post);
    }
    Event.replayEventsOn(eventHandler, result.getEvents());
    return result;
  }

  /** Returns a new {@link LegacyGlobber}. */
  public LegacyGlobber createLegacyGlobber(
      Path packageDirectory,
      PackageIdentifier packageId,
      ImmutableSet<PathFragment> blacklistedGlobPrefixes,
      CachingPackageLocator locator) {
    return createLegacyGlobber(
        new GlobCache(
            packageDirectory,
            packageId,
            blacklistedGlobPrefixes,
            locator,
            syscalls,
            executor,
            maxDirectoriesToEagerlyVisitInGlobbing));
  }

  /** Returns a new {@link LegacyGlobber}. */
  public static LegacyGlobber createLegacyGlobber(GlobCache globCache) {
    return new LegacyGlobber(globCache, /*sort=*/ true);
  }

  @Nullable
  private byte[] maybeGetBuildFileBytes(Path buildFile, ExtendedEventHandler eventHandler) {
    try {
      return FileSystemUtils.readWithKnownFileSize(buildFile, buildFile.getFileSize());
    } catch (IOException e) {
      eventHandler.handle(Event.error(Location.fromFile(buildFile), e.getMessage()));
      return null;
    }
  }

  /**
   * This class holds state associated with the construction of a single package for the duration of
   * execution of one BUILD file. (We use a PackageContext object in preference to storing these
   * values in mutable fields of the PackageFactory.)
   *
   * <p>PLEASE NOTE: the PackageContext is referred to by the StarlarkThread, but should become
   * unreachable once the StarlarkThread is discarded at the end of evaluation. Please be aware of
   * your memory footprint when making changes here!
   */
  // TODO(adonovan): is there any reason not to merge this with Package.Builder?
  public static class PackageContext {
    final Package.Builder pkgBuilder;
    final Globber globber;
    final ExtendedEventHandler eventHandler;

    @VisibleForTesting
    public PackageContext(
        Package.Builder pkgBuilder, Globber globber, ExtendedEventHandler eventHandler) {
      this.pkgBuilder = pkgBuilder;
      this.eventHandler = eventHandler;
      this.globber = globber;
    }

    /**
     * Returns the Label of this Package.
     */
    public Label getLabel() {
      return pkgBuilder.getBuildFileLabel();
    }

    /**
     * Sets a Make variable.
     */
    public void setMakeVariable(String name, String value) {
      pkgBuilder.setMakeVariable(name, value);
    }

    /**
     * Returns the builder of this Package.
     */
    public Package.Builder getBuilder() {
      return pkgBuilder;
    }
  }

  private final ClassObject nativeModule;
  private final ClassObject workspaceNativeModule;

  /** @return the Skylark struct to bind to "native" */
  public ClassObject getNativeModule(boolean workspace) {
    return workspace ? workspaceNativeModule : nativeModule;
  }

  /**
   * Returns a native module with the functions created using the {@link RuleClassProvider}
   * of this {@link PackageFactory}.
   */
  private ClassObject newNativeModule() {
    ImmutableMap.Builder<String, Object> builder = new ImmutableMap.Builder<>();
    builder.putAll(SkylarkNativeModule.BINDINGS_FOR_BUILD_FILES);
    builder.putAll(ruleFunctions);
    builder.put("package", newPackageFunction(packageArguments));
    for (EnvironmentExtension ext : environmentExtensions) {
      ext.updateNative(builder);
    }
    return StructProvider.STRUCT.create(builder.build(), "no native function or rule '%s'");
  }

  private void populateEnvironment(ImmutableMap.Builder<String, Object> env) {
    env.putAll(Starlark.UNIVERSE);
    env.putAll(StarlarkBuildLibrary.BINDINGS);
    env.putAll(SkylarkNativeModule.BINDINGS_FOR_BUILD_FILES);
    env.put("package", newPackageFunction(packageArguments));
    env.putAll(ruleFunctions);

    for (EnvironmentExtension ext : environmentExtensions) {
      ext.update(env);
    }
  }

  /**
   * Called by a caller of {@link #createPackageFromAst} after this caller has fully loaded the
   * package.
   */
  public void afterDoneLoadingPackage(
      Package pkg, StarlarkSemantics starlarkSemantics, long loadTimeNanos) {
    packageBuilderHelper.onLoadingComplete(pkg, starlarkSemantics, loadTimeNanos);
  }

  /**
   * Constructs a Package instance, evaluates the BUILD-file AST inside the build environment, and
   * populates the package with Rule instances as it goes. As with most programming languages,
   * evaluation stops when an exception is encountered: no further rules after the point of failure
   * will be constructed. We assume that rules constructed before the point of failure are valid;
   * this assumption is not entirely correct, since a "vardef" after a rule declaration can affect
   * the behavior of that rule.
   *
   * <p>Rule attribute checking is performed during evaluation. Each attribute must conform to the
   * type specified for that <i>(rule class, attribute name)</i> pair. Errors reported at this stage
   * include: missing value for mandatory attribute, value of wrong type. Such error cause Rule
   * construction to be aborted, so the resulting package will have missing members.
   *
   * @see PackageFactory#PackageFactory
   */
  @VisibleForTesting // used by PackageFactoryApparatus
  public Package.Builder evaluateBuildFile(
      String workspaceName,
      PackageIdentifier packageId,
      StarlarkFile file,
      RootedPath buildFilePath,
      Globber globber,
      RuleVisibility defaultVisibility,
      StarlarkSemantics semantics,
      Map<String, Extension> imports,
      ImmutableList<Label> skylarkFileDependencies,
      ImmutableMap<RepositoryName, RepositoryName> repositoryMapping)
      throws InterruptedException {
    Package pkg =
        packageBuilderHelper.createFreshPackage(packageId, ruleClassProvider.getRunfilesPrefix());
    Package.Builder pkgBuilder =
        new Package.Builder(pkg, semantics)
            .setFilename(buildFilePath)
            .setDefaultVisibility(defaultVisibility)
            // "defaultVisibility" comes from the command line.
            // Let's give the BUILD file a chance to set default_visibility once,
            // by resetting the PackageBuilder.defaultVisibilitySet flag.
            .setDefaultVisibilitySet(false)
            .setSkylarkFileDependencies(skylarkFileDependencies)
            .setWorkspaceName(workspaceName)
            .setRepositoryMapping(repositoryMapping)
            .setThirdPartyLicenceExistencePolicy(
                ruleClassProvider.getThirdPartyLicenseExistencePolicy());
    StoredEventHandler eventHandler = new StoredEventHandler();
    if (!buildPackage(
        pkgBuilder,
        packageId,
        file,
        semantics,
        imports,
        new PackageContext(pkgBuilder, globber, eventHandler))) {
      pkgBuilder.setContainsErrors();
    }
    pkgBuilder.addPosts(eventHandler.getPosts());
    pkgBuilder.addEvents(eventHandler.getEvents());
    return pkgBuilder;
  }

  // Validates and executes a parsed BUILD file, returning true on success,
  // or reporting errors to pkgContext.eventHandler on failure.
  private boolean buildPackage(
      Package.Builder pkgBuilder,
      PackageIdentifier packageId,
      StarlarkFile file,
      StarlarkSemantics semantics,
      Map<String, Extension> imports,
      PackageContext pkgContext)
      throws InterruptedException {

    // Report scan/parse errors.
    if (!file.ok()) {
      Event.replayEventsOn(pkgContext.eventHandler, file.errors());
      return false;
    }

    // Validate the package identifier.
    // TODO(adonovan): it's kinda late to be doing this check.
    // after we've parsed the BUILD file and created the Package.
    String error = LabelValidator.validatePackageName(packageId.getPackageFragment().toString());
    if (error != null) {
      pkgContext.eventHandler.handle(Event.error(file.getLocation(), error));
      return false;
    }

    // Construct environment.
    ImmutableMap.Builder<String, Object> env = ImmutableMap.builder();
    populateEnvironment(env);

    // TODO(adonovan): defer creation of Mutability + Thread till after validation,
    // once the validate API is rationalized.
    try (Mutability mutability = Mutability.create("package", packageId)) {
      StarlarkThread thread =
          StarlarkThread.builder(mutability)
              .setGlobals(Module.createForBuiltins(env.build()))
              .setSemantics(semantics)
              .setEventHandler(pkgContext.eventHandler) // for print statements
              .setImportedExtensions(imports)
              .build();

      // Validate.
      ValidationEnvironment.validateFile(
          file, thread.getGlobals(), semantics, /*isBuildFile=*/ true);
      if (!file.ok()) {
        Event.replayEventsOn(pkgContext.eventHandler, file.errors());
        return false;
      }

      // Check syntax. Make a pass over the syntax tree to:
      // - reject forbidden BUILD syntax
      // - extract literal glob patterns for prefetching
      // - record the generator_name of each top-level macro call
      Set<String> globs = new HashSet<>();
      Set<String> globsWithDirs = new HashSet<>();
      if (!checkBuildSyntax(
          file,
          globs,
          globsWithDirs,
          pkgBuilder.getGeneratorNameByLocation(),
          pkgContext.eventHandler)) {
        return false;
      }

      // Prefetch glob patterns asynchronously.
      if (maxDirectoriesToEagerlyVisitInGlobbing == -2) {
        try {
          pkgContext.globber.runAsync(
              ImmutableList.copyOf(globs),
              ImmutableList.of(),
              /*excludeDirs=*/ true,
              /*allowEmpty=*/ true);
          pkgContext.globber.runAsync(
              ImmutableList.copyOf(globsWithDirs),
              ImmutableList.of(),
              /*excludeDirs=*/ false,
              /*allowEmpty=*/ true);
        } catch (BadGlobException ex) {
          // Ignore exceptions.
          // Errors will be properly reported when the actual globbing is done.
        }
      }

      new BazelStarlarkContext(
              BazelStarlarkContext.Phase.LOADING,
              ruleClassProvider.getToolsRepository(),
              /*fragmentNameToClass=*/ null,
              pkgBuilder.getRepositoryMapping(),
              new SymbolGenerator<>(packageId),
              /*analysisRuleLabel=*/ null)
          .storeInThread(thread);

      // TODO(adonovan): save this as a field in BazelSkylarkContext.
      // It needn't be a second thread-local.
      thread.setThreadLocal(PackageContext.class, pkgContext);

      // Execute.
      try {
        EvalUtils.exec(file, thread);
      } catch (EvalException ex) {
        pkgContext.eventHandler.handle(Event.error(ex.getLocation(), ex.getMessage()));
        return false;
      }
    }

    return true; // success
  }

  /**
   * checkBuildSyntax is a static pass over the syntax tree of a BUILD (not .bzl) file.
   *
   * <p>It reports an error to the event handler if it discovers a {@code def}, {@code if}, or
   * {@code for} statement, or a {@code f(*args)} or {@code f(**kwargs)} call.
   *
   * <p>It extracts literal {@code glob(include="pattern")} patterns and adds them to {@code globs},
   * or to {@code globsWithDirs} if the call had a {@code exclude_directories=0} argument.
   *
   * <p>It records in {@code generatorNameByLocation} all calls of the form {@code f(name="foo",
   * ...)} so that any rules instantiated during the call to {@code f} can be ascribed a "generator
   * name" of {@code "foo"}.
   *
   * <p>It returns true if it reported no errors.
   */
  // TODO(adonovan): restructure so that this is called from the sole place that executes BUILD
  // files. Also, make private; there's reason for tests to call this directly.
  public static boolean checkBuildSyntax(
      StarlarkFile file,
      Collection<String> globs,
      Collection<String> globsWithDirs,
      Map<Location, String> generatorNameByLocation,
      EventHandler eventHandler) {
    final boolean[] success = {true};
    NodeVisitor checker =
        new NodeVisitor() {
          void error(Node node, String message) {
            eventHandler.handle(Event.error(node.getLocation(), message));
            success[0] = false;
          }

          // Extract literal glob patterns from calls of the form:
          //   glob(include = ["pattern"])
          //   glob(["pattern"])
          // This may spuriously match user-defined functions named glob;
          // that's ok, it's only a heuristic.
          void extractGlobPatterns(FuncallExpression call) {
            if (call.getFunction() instanceof Identifier
                && ((Identifier) call.getFunction()).getName().equals("glob")) {
              Expression excludeDirectories = null, include = null;
              List<Argument> arguments = call.getArguments();
              for (int i = 0; i < arguments.size(); i++) {
                Argument arg = arguments.get(i);
                String name = arg.getName();
                if (name == null) {
                  if (i == 0) { // first positional argument
                    include = arg.getValue();
                  }
                } else if (name.equals("include")) {
                  include = arg.getValue();
                } else if (name.equals("exclude_directories")) {
                  excludeDirectories = arg.getValue();
                }
              }
              if (include instanceof ListExpression) {
                for (Expression elem : ((ListExpression) include).getElements()) {
                  if (elem instanceof StringLiteral) {
                    String pattern = ((StringLiteral) elem).getValue();
                    // exclude_directories is (oddly) an int with default 1.
                    boolean exclude = true;
                    if (excludeDirectories instanceof IntegerLiteral
                        && ((IntegerLiteral) excludeDirectories).getValue() == 0) {
                      exclude = false;
                    }
                    (exclude ? globs : globsWithDirs).add(pattern);
                  }
                }
              }
            }
          }

          // Reject f(*args) and f(**kwargs) calls in BUILD files.
          void rejectStarArgs(FuncallExpression call) {
            for (Argument arg : call.getArguments()) {
              if (arg instanceof Argument.StarStar) {
                error(
                    call,
                    "**kwargs arguments are not allowed in BUILD files. Pass the arguments in "
                        + "explicitly.");
              } else if (arg instanceof Argument.Star) {
                error(
                    call,
                    "*args arguments are not allowed in BUILD files. Pass the arguments in "
                        + "explicitly.");
              }
            }
          }

          // Record calls of the form f(name="foo", ...)
          // so that we can later ascribe "foo" as the "generator name"
          // of any rules instantiated during the call of f.
          void recordGeneratorName(FuncallExpression call) {
            for (Argument arg : call.getArguments()) {
              if (arg instanceof Argument.Keyword
                  && arg.getName().equals("name")
                  && arg.getValue() instanceof StringLiteral) {
                generatorNameByLocation.put(
                    call.getLocation(), ((StringLiteral) arg.getValue()).getValue());
              }
            }
          }

          // We prune the traversal if we encounter def/if/for,
          // as we have already reported the root error and there's
          // no point reporting more.

          @Override
          public void visit(DefStatement node) {
            error(
                node,
                "function definitions are not allowed in BUILD files. You may move the function to "
                    + "a .bzl file and load it.");
          }

          @Override
          public void visit(ForStatement node) {
            error(
                node,
                "for statements are not allowed in BUILD files. You may inline the loop, move it "
                    + "to a function definition (in a .bzl file), or as a last resort use a list "
                    + "comprehension.");
          }

          @Override
          public void visit(IfStatement node) {
            error(
                node,
                "if statements are not allowed in BUILD files. You may move conditional logic to a "
                    + "function definition (in a .bzl file), or for simple cases use an if "
                    + "expression.");
          }

          @Override
          public void visit(FuncallExpression node) {
            extractGlobPatterns(node);
            rejectStarArgs(node);
            recordGeneratorName(node);
            // Continue traversal so as not to miss nested calls
            // like cc_binary(..., f(**kwargs), srcs=glob(...), ...).
            super.visit(node);
          }
        };
    checker.visit(file);
    return success[0];
  }
}
