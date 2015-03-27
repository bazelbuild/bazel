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

package com.google.devtools.build.lib.packages;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.devtools.build.lib.cmdline.LabelValidator;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.events.NullEventHandler;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.GlobCache.BadGlobException;
import com.google.devtools.build.lib.packages.License.DistributionType;
import com.google.devtools.build.lib.packages.Type.ConversionException;
import com.google.devtools.build.lib.syntax.AbstractFunction;
import com.google.devtools.build.lib.syntax.AssignmentStatement;
import com.google.devtools.build.lib.syntax.BuildFileAST;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.Environment.NoSuchVariableException;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Expression;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.Function;
import com.google.devtools.build.lib.syntax.GlobList;
import com.google.devtools.build.lib.syntax.Ident;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.syntax.MixedModeFunction;
import com.google.devtools.build.lib.syntax.ParserInputSource;
import com.google.devtools.build.lib.syntax.SkylarkEnvironment;
import com.google.devtools.build.lib.syntax.Statement;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.UnixGlob;

import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
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
        Package.LegacyBuilder pkgBuilder, Location location, Object value)
        throws EvalException, ConversionException {
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
        Package.LegacyBuilder pkgBuilder, Location location, T value)
        throws EvalException;
  }

  /** Interface for evaluating globs during package loading. */
  public static interface Globber {
    /** An opaque token for fetching the result of a glob computation. */
    abstract static class Token {}

    /**
     * Asynchronously starts the given glob computation and returns a token for fetching the
     * result.
     */
    Token runAsync(List<String> includes, List<String> excludes, boolean excludeDirs)
        throws BadGlobException;

    /** Fetches the result of a previously started glob computation. */
    List<String> fetch(Token token) throws IOException, InterruptedException;

    /** Should be called when the globber is about to be discarded due to an interrupt. */
    void onInterrupt();

    /** Should be called when the globber is no longer needed. */
    void onCompletion();

    /** Returns all the glob computations requested before {@link #onCompletion} was called. */
    Set<Pair<String, Boolean>> getGlobPatterns();
  }

  /**
   * An extension to the global namespace of the BUILD language.
   */
  public interface EnvironmentExtension {
    /**
     * Update the global environment with the identifiers this extension contributes.
     */
    void update(Environment environment, Label buildFileLabel);

    /**
     * Returns the extra functions needed to be added to the Skylark native module.
     */
    ImmutableList<Function> nativeModuleFunctions();

    Iterable<PackageArgument<?>> getPackageArguments();
  }

  private static final int EXCLUDE_DIR_DEFAULT = 1;

  private static class DefaultVisibility extends PackageArgument<List<Label>> {
    private DefaultVisibility() {
      super("default_visibility", Type.LABEL_LIST);
    }

    @Override
    protected void process(Package.LegacyBuilder pkgBuilder, Location location,
        List<Label> value) {
      pkgBuilder.setDefaultVisibility(getVisibility(value));
    }
  }

  private static class DefaultTestOnly extends PackageArgument<Boolean> {
    private DefaultTestOnly() {
      super("default_testonly", Type.BOOLEAN);
    }

    @Override
    protected void process(Package.LegacyBuilder pkgBuilder, Location location,
        Boolean value) {
      pkgBuilder.setDefaultTestonly(value);
    }
  }

  private static class DefaultDeprecation extends PackageArgument<String> {
    private DefaultDeprecation() {
      super("default_deprecation", Type.STRING);
    }

    @Override
    protected void process(Package.LegacyBuilder pkgBuilder, Location location,
        String value) {
      pkgBuilder.setDefaultDeprecation(value);
    }
  }

  private static class Features extends PackageArgument<List<String>> {
    private Features() {
      super("features", Type.STRING_LIST);
    }

    @Override
    protected void process(Package.LegacyBuilder pkgBuilder, Location location,
        List<String> value) {
      pkgBuilder.addFeatures(value);
    }
  }

  private static class DefaultLicenses extends PackageArgument<License> {
    private DefaultLicenses() {
      super("licenses", Type.LICENSE);
    }

    @Override
    protected void process(Package.LegacyBuilder pkgBuilder, Location location,
        License value) {
      pkgBuilder.setDefaultLicense(value);
    }
  }

  private static class DefaultDistribs extends PackageArgument<Set<DistributionType>> {
    private DefaultDistribs() {
      super("distribs", Type.DISTRIBUTIONS);
    }

    @Override
    protected void process(Package.LegacyBuilder pkgBuilder, Location location,
        Set<DistributionType> value) {
      pkgBuilder.setDefaultDistribs(value);
    }
  }

  /**
   * Declares the package() attribute specifying the default value for
   * {@link RuleClass#COMPATIBLE_ENVIRONMENT_ATTR} when not explicitly specified.
   */
  private static class DefaultCompatibleWith extends PackageArgument<List<Label>> {
    private DefaultCompatibleWith() {
      super(Package.DEFAULT_COMPATIBLE_WITH_ATTRIBUTE, Type.LABEL_LIST);
    }

    @Override
    protected void process(Package.LegacyBuilder pkgBuilder, Location location,
        List<Label> value) {
      pkgBuilder.setDefaultCompatibleWith(value, Package.DEFAULT_COMPATIBLE_WITH_ATTRIBUTE,
          location);
    }
  }

  /**
   * Declares the package() attribute specifying the default value for
   * {@link RuleClass#RESTRICTED_ENVIRONMENT_ATTR} when not explicitly specified.
   */
  private static class DefaultRestrictedTo extends PackageArgument<List<Label>> {
    private DefaultRestrictedTo() {
      super(Package.DEFAULT_RESTRICTED_TO_ATTRIBUTE, Type.LABEL_LIST);
    }

    @Override
    protected void process(Package.LegacyBuilder pkgBuilder, Location location,
        List<Label> value) {
      pkgBuilder.setDefaultRestrictedTo(value, Package.DEFAULT_RESTRICTED_TO_ATTRIBUTE, location);
    }
  }

  public static final String PKG_CONTEXT = "$pkg_context";

  // Used outside of Bazel!
  /** {@link Globber} that uses the legacy GlobCache. */
  public static class LegacyGlobber implements Globber {

    private final GlobCache globCache;

    public LegacyGlobber(GlobCache globCache) {
      this.globCache = globCache;
    }

    private class Token extends Globber.Token {
      public final List<String> includes;
      public final List<String> excludes;
      public final boolean excludeDirs;

      public Token(List<String> includes, List<String> excludes, boolean excludeDirs) {
        this.includes = includes;
        this.excludes = excludes;
        this.excludeDirs = excludeDirs;
      }
    }

    @Override
    public Set<Pair<String, Boolean>> getGlobPatterns() {
      return globCache.getKeySet();
    }

    @Override
    public Token runAsync(List<String> includes, List<String> excludes, boolean excludeDirs)
        throws BadGlobException {
      for (String pattern : Iterables.concat(includes, excludes)) {
        globCache.getGlobAsync(pattern, excludeDirs);
      }
      return new Token(includes, excludes, excludeDirs);
    }

    @Override
    public List<String> fetch(Globber.Token token) throws IOException, InterruptedException {
      Token legacyToken = (Token) token;
      try {
        return globCache.glob(legacyToken.includes, legacyToken.excludes,
            legacyToken.excludeDirs);
      } catch (BadGlobException e) {
        throw new IllegalStateException(e);
      }
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

  private static final Logger LOG = Logger.getLogger(PackageFactory.class.getName());

  private final RuleFactory ruleFactory;
  private final RuleClassProvider ruleClassProvider;
  private final Environment globalEnv;

  private AtomicReference<? extends UnixGlob.FilesystemCalls> syscalls;
  private Preprocessor.Factory preprocessorFactory = Preprocessor.Factory.NullFactory.INSTANCE;

  private final ThreadPoolExecutor threadPool;
  private Map<String, String> platformSetRegexps;

  private final ImmutableList<EnvironmentExtension> environmentExtensions;
  private final ImmutableMap<String, PackageArgument<?>> packageArguments;

  /**
   * Constructs a {@code PackageFactory} instance with the given rule factory.
   */
  @VisibleForTesting
  public PackageFactory(RuleClassProvider ruleClassProvider) {
    this(ruleClassProvider, null, ImmutableList.<EnvironmentExtension>of());
  }

  @VisibleForTesting
  public PackageFactory(RuleClassProvider ruleClassProvider,
      EnvironmentExtension environmentExtension) {
    this(ruleClassProvider, null, ImmutableList.of(environmentExtension));
  }

  @VisibleForTesting
  public PackageFactory(RuleClassProvider ruleClassProvider,
      Iterable<EnvironmentExtension> environmentExtensions) {
    this(ruleClassProvider, null, environmentExtensions);
  }

  /**
   * Constructs a {@code PackageFactory} instance with a specific glob path translator
   * and rule factory.
   */
  public PackageFactory(RuleClassProvider ruleClassProvider,
      Map<String, String> platformSetRegexps,
      Iterable<EnvironmentExtension> environmentExtensions) {
    this.platformSetRegexps = platformSetRegexps;
    this.ruleFactory = new RuleFactory(ruleClassProvider);
    this.ruleClassProvider = ruleClassProvider;
    globalEnv = newGlobalEnvironment();
    threadPool = new ThreadPoolExecutor(100, 100, 3L, TimeUnit.SECONDS,
        new LinkedBlockingQueue<Runnable>(),
        new ThreadFactoryBuilder().setNameFormat("PackageFactory %d").build());
    // Do not consume threads when not in use.
    threadPool.allowCoreThreadTimeOut(true);
    this.environmentExtensions = ImmutableList.copyOf(environmentExtensions);
    this.packageArguments = createPackageArguments();
  }

  /**
   * Sets the preprocessor used.
   */
  public void setPreprocessorFactory(Preprocessor.Factory preprocessorFactory) {
    this.preprocessorFactory = preprocessorFactory;
  }

 /**
   * Sets the syscalls cache used in globbing.
   */
  public void setSyscalls(AtomicReference<? extends UnixGlob.FilesystemCalls> syscalls) {
    this.syscalls = Preconditions.checkNotNull(syscalls);
  }

  /**
   * Returns the static environment initialized once and shared by all packages
   * created by this factory. No updates occur to this environment once created.
   */
  @VisibleForTesting
  public Environment getEnvironment() {
    return globalEnv;
  }

  /**
   * Returns the immutable, unordered set of names of all the known rule
   * classes.
   */
  public Set<String> getRuleClassNames() {
    return ruleFactory.getRuleClassNames();
  }

  /**
   * Returns the {@link RuleClass} for the specified rule class name.
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

  /****************************************************************************
   * Environment function factories.
   */

  /**
   * Returns a function-value implementing "glob" in the specified package
   * context.
   *
   * @param async if true, start globs in the background but don't block on their completion.
   *        Only use this for heuristic preloading.
   */
  private static Function newGlobFunction(
      final PackageContext originalContext, final boolean async) {
    List<String> params = ImmutableList.of("include", "exclude", "exclude_directories");
    return new MixedModeFunction("glob", params, 1, false) {
        @Override
        public Object call(Object[] namedArguments, FuncallExpression ast, Environment env)
                throws EvalException, ConversionException, InterruptedException {
          return callGlob(originalContext, async, ast, env, namedArguments);
        }
      };
  }

  static Object callGlob(@Nullable PackageContext originalContext, boolean async,
      FuncallExpression ast, Environment env, Object[] namedArguments)
          throws EvalException, ConversionException, InterruptedException {
    // Skylark build extensions need to get the PackageContext from the Environment;
    // async glob functions cannot do the same because the Environment is not thread safe.
    PackageContext context;
    if (originalContext == null) {
      Preconditions.checkArgument(!async);
      context = getContext(env, ast);
    } else {
      context = originalContext;
    }

    List<String> includes = Type.STRING_LIST.convert(namedArguments[0], "'glob' argument");
    List<String> excludes = namedArguments[1] == null
        ? Collections.<String>emptyList()
        : Type.STRING_LIST.convert(namedArguments[1], "'glob' argument");
    int excludeDirs = namedArguments[2] == null
      ? EXCLUDE_DIR_DEFAULT
      : Type.INTEGER.convert(namedArguments[2], "'glob' argument");

    if (async) {
      try {
        context.globber.runAsync(includes, excludes, excludeDirs != 0);
      } catch (GlobCache.BadGlobException e) {
        // Ignore: errors will appear during the actual evaluation of the package.
      }
      return GlobList.captureResults(includes, excludes, ImmutableList.<String>of());
    } else {
      return handleGlob(includes, excludes, excludeDirs != 0, context, ast);
    }
  }

  /**
   * Adds a glob to the package, reporting any errors it finds.
   *
   * @param includes the list of includes which must be non-null
   * @param excludes the list of excludes which must be non-null
   * @param context the package context
   * @param ast the AST
   * @return the list of matches
   * @throws EvalException if globbing failed
   */
  private static GlobList<String> handleGlob(List<String> includes, List<String> excludes,
      boolean excludeDirs, PackageContext context, FuncallExpression ast)
        throws EvalException, InterruptedException {
    try {
      Globber.Token globToken = context.globber.runAsync(includes, excludes, excludeDirs);
      List<String> matches = context.globber.fetch(globToken);
      return GlobList.captureResults(includes, excludes, matches);
    } catch (IOException expected) {
      context.eventHandler.handle(Event.error(ast.getLocation(),
              "error globbing [" + Joiner.on(", ").join(includes) + "]: " + expected.getMessage()));
      context.pkgBuilder.setContainsTemporaryErrors();
      return GlobList.captureResults(includes, excludes, ImmutableList.<String>of());
    } catch (GlobCache.BadGlobException e) {
      throw new EvalException(ast.getLocation(), e.getMessage());
    }
  }

  /**
   * Returns a function value implementing the "mocksubinclude" function,
   * emitted by the PythonPreprocessor.  We annotate the
   * package with additional dependencies.  (A 'real' subinclude will never be
   * seen by the parser, because the presence of "subinclude" triggers
   * preprocessing.)
   */
  private static Function newMockSubincludeFunction(final PackageContext context) {
    return new MixedModeFunction("mocksubinclude", ImmutableList.of("label", "path"), 2, false) {
        @Override
        public Object call(Object[] args, FuncallExpression ast)
            throws ConversionException {
          Label label = Type.LABEL.convert(args[0], "'mocksubinclude' argument",
                                           context.pkgBuilder.getBuildFileLabel());
          String pathString = Type.STRING.convert(args[1], "'mocksubinclude' argument");
          Path path = pathString.isEmpty()
              ? null
              : context.pkgBuilder.getFilename().getRelative(pathString);
          // A subinclude within a package counts as a file declaration.
          if (label.getPackageIdentifier().equals(context.pkgBuilder.getPackageIdentifier())) {
            Location location = ast.getLocation();
            if (location == null) {
              location = Location.fromFile(context.pkgBuilder.getFilename());
            }
            context.pkgBuilder.createInputFileMaybe(label, location);
          }

          context.pkgBuilder.addSubinclude(label, path);
          return Environment.NONE;
        }
      };
  }

  /**
   * Fake function: subinclude calls are ignored
   * They will disappear after the Python preprocessing.
   */
  private static Function newSubincludeFunction() {
    return new MixedModeFunction("subinclude", ImmutableList.of("file"), 1, false) {
        @Override
        public Object call(Object[] args, FuncallExpression ast) {
          return Environment.NONE;
        }
      };
  }

  /**
   * Returns a function value implementing "environment_group" in the specified package context.
   * Syntax is as follows:
   *
   * <pre>{@code
   *   environment_group(
   *       name = "sample_group",
   *       environments = [":env1", ":env2", ...],
   *       defaults = [":env1", ...]
   *   )
   * }</pre>
   *
   * <p>Where ":env1", "env2", ... are all environment rules declared in the same package. All
   * parameters are mandatory.
   */
  private static Function newEnvironmentGroupFunction(final PackageContext context) {
    List<String> params = ImmutableList.of("name", "environments", "defaults");
    return new MixedModeFunction("environment_group", params, params.size(), true) {
        @Override
        public Object call(Object[] namedArgs, FuncallExpression ast)
            throws EvalException, ConversionException {
          Preconditions.checkState(namedArgs[0] != null);
          String name = Type.STRING.convert(namedArgs[0], "'environment_group' argument");
          Preconditions.checkState(namedArgs[1] != null);
          List<Label> environments = Type.LABEL_LIST.convert(
              namedArgs[1], "'environment_group argument'", context.pkgBuilder.getBuildFileLabel());
          Preconditions.checkState(namedArgs[2] != null);
          List<Label> defaults = Type.LABEL_LIST.convert(
              namedArgs[2], "'environment_group argument'", context.pkgBuilder.getBuildFileLabel());

          try {
            context.pkgBuilder.addEnvironmentGroup(name, environments, defaults,
                context.eventHandler, ast.getLocation());
            return Environment.NONE;
          } catch (Label.SyntaxException e) {
            throw new EvalException(ast.getLocation(),
                "environment group has invalid name: " + name + ": " + e.getMessage());
          } catch (Package.NameConflictException e) {
            throw new EvalException(ast.getLocation(), e.getMessage());
          }
        }
      };
  }

  /**
   * Returns a function-value implementing "exports_files" in the specified
   * package context.
   */
  private static Function newExportsFilesFunction() {
    List<String> params = ImmutableList.of("srcs", "visibility", "licenses");
    return new MixedModeFunction("exports_files", params, 1, false) {
      @Override
      public Object call(Object[] namedArgs, FuncallExpression ast, Environment env)
          throws EvalException, ConversionException {
        return callExportsFiles(ast, env, namedArgs);
      }
    };
  }

  static Object callExportsFiles(FuncallExpression ast, Environment env, Object[] namedArgs)
      throws EvalException, ConversionException {
    Package.LegacyBuilder pkgBuilder = getContext(env, ast).pkgBuilder;
    List<String> files = Type.STRING_LIST.convert(namedArgs[0], "'exports_files' operand");

    RuleVisibility visibility = namedArgs[1] == null
        ? ConstantRuleVisibility.PUBLIC
        : getVisibility(Type.LABEL_LIST.convert(
            namedArgs[1],
            "'exports_files' operand",
            pkgBuilder.getBuildFileLabel()));
    License license = namedArgs[2] == null
        ? null
        : Type.LICENSE.convert(namedArgs[2], "'exports_files' operand");

    for (String file : files) {
      String errorMessage = LabelValidator.validateTargetName(file);
      if (errorMessage != null) {
        throw new EvalException(ast.getLocation(), errorMessage);
      }
      try {
        InputFile inputFile = pkgBuilder.createInputFile(file, ast.getLocation());
        if (inputFile.isVisibilitySpecified()
            && inputFile.getVisibility() != visibility) {
          throw new EvalException(ast.getLocation(),
              String.format("visibility for exported file '%s' declared twice",
                  inputFile.getName()));
        }
        if (license != null && inputFile.isLicenseSpecified()) {
          throw new EvalException(ast.getLocation(),
              String.format("licenses for exported file '%s' declared twice",
                  inputFile.getName()));
        }
        if (license == null && pkgBuilder.getDefaultLicense() == License.NO_LICENSE
            && pkgBuilder.getBuildFileLabel().toString().startsWith("//third_party/")) {
          throw new EvalException(ast.getLocation(),
              "third-party file '" + inputFile.getName() + "' lacks a license declaration "
              + "with one of the following types: notice, reciprocal, permissive, "
              + "restricted, unencumbered, by_exception_only");
        }

        pkgBuilder.setVisibilityAndLicense(inputFile, visibility, license);
      } catch (Package.Builder.GeneratedLabelConflict e) {
        throw new EvalException(ast.getLocation(), e.getMessage());
      }
    }
    return Environment.NONE;
  }

  /**
   * Returns a function-value implementing "licenses" in the specified package
   * context.
   * TODO(bazel-team): Remove in favor of package.licenses.
   */
  private static Function newLicensesFunction(final PackageContext context) {
    return new MixedModeFunction("licenses", ImmutableList.of("object"), 1, false) {
        @Override
        public Object call(Object[] args, FuncallExpression ast) {
          try {
            License license = Type.LICENSE.convert(args[0], "'licenses' operand");
            context.pkgBuilder.setDefaultLicense(license);
          } catch (ConversionException e) {
            context.eventHandler.handle(Event.error(ast.getLocation(), e.getMessage()));
            context.pkgBuilder.setContainsErrors();
          }
          return Environment.NONE;
        }
      };
  }

  /**
   * Returns a function-value implementing "distribs" in the specified package
   * context.
   * TODO(bazel-team): Remove in favor of package.distribs.
   */
  private static Function newDistribsFunction(final PackageContext context) {
    return new MixedModeFunction("distribs", ImmutableList.of("object"), 1, false) {
        @Override
        public Object call(Object[] args, FuncallExpression ast) {
          try {
            Set<DistributionType> distribs = Type.DISTRIBUTIONS.convert(args[0],
                "'distribs' operand");
            context.pkgBuilder.setDefaultDistribs(distribs);
          } catch (ConversionException e) {
            context.eventHandler.handle(Event.error(ast.getLocation(), e.getMessage()));
            context.pkgBuilder.setContainsErrors();
          }
          return Environment.NONE;
        }
      };
  }

  private static Function newPackageGroupFunction() {
    List<String> params = ImmutableList.of("name", "packages", "includes");
    return new MixedModeFunction("package_group", params, 1, true) {
        @Override
        public Object call(Object[] namedArgs, FuncallExpression ast, Environment env)
            throws EvalException, ConversionException {
          return callPackageFunction(ast, env, namedArgs);
        }
      };
  }

  static Object callPackageFunction(FuncallExpression ast, Environment env, Object[] namedArgs)
      throws EvalException, ConversionException {
    PackageContext context = getContext(env, ast);
    Preconditions.checkState(namedArgs[0] != null);
    String name = Type.STRING.convert(namedArgs[0], "'package_group' argument");
    List<String> packages = namedArgs[1] == null
        ? Collections.<String>emptyList()
        : Type.STRING_LIST.convert(namedArgs[1], "'package_group' argument");
    List<Label> includes = namedArgs[2] == null
        ? Collections.<Label>emptyList()
        : Type.LABEL_LIST.convert(namedArgs[2], "'package_group argument'",
                                  context.pkgBuilder.getBuildFileLabel());

    try {
      context.pkgBuilder.addPackageGroup(name, packages, includes, context.eventHandler,
          ast.getLocation());
      return Environment.NONE;
    } catch (Label.SyntaxException e) {
      throw new EvalException(ast.getLocation(),
          "package group has invalid name: " + name + ": " + e.getMessage());
    } catch (Package.NameConflictException e) {
      throw new EvalException(ast.getLocation(), e.getMessage());
    }
  }

  public static RuleVisibility getVisibility(List<Label> original) {
    RuleVisibility result;

    result = ConstantRuleVisibility.tryParse(original);
    if (result != null) {
      return result;
    }

    result = PackageGroupsRuleVisibility.tryParse(original);
    return result;
  }

  /**
   * Returns a function-value implementing "package" in the specified package
   * context.
   */
  private static Function newPackageFunction(
      final Map<String, PackageArgument<?>> packageArguments) {
    return new MixedModeFunction("package", packageArguments.keySet(), 0, true) {
      @Override
      public Object call(Object[] namedArguments, FuncallExpression ast, Environment env)
          throws EvalException, ConversionException {

        Package.LegacyBuilder pkgBuilder = getContext(env, ast).pkgBuilder;

        // Validate parameter list
        if (pkgBuilder.isPackageFunctionUsed()) {
          throw new EvalException(ast.getLocation(),
              "'package' can only be used once per BUILD file");
        }
        pkgBuilder.setPackageFunctionUsed();

        // Parse params
        boolean foundParameter = false;

        int argNumber = 0;
        for (Map.Entry<String, PackageArgument<?>> entry : packageArguments.entrySet()) {
          Object arg = namedArguments[argNumber];
          argNumber += 1;
          if (arg == null) {
            continue;
          }

          foundParameter = true;
          entry.getValue().convertAndProcess(pkgBuilder, ast.getLocation(), arg);
        }

        if (!foundParameter) {
          throw new EvalException(ast.getLocation(),
              "at least one argument must be given to the 'package' function");
        }

        return Environment.NONE;
      }
    };
  }

  // Helper function for createRuleFunction.
  private static void addRule(RuleFactory ruleFactory,
                              String ruleClassName,
                              PackageContext context,
                              Map<String, Object> kwargs,
                              FuncallExpression ast)
      throws RuleFactory.InvalidRuleException, Package.NameConflictException {
    RuleClass ruleClass = getBuiltInRuleClass(ruleClassName, ruleFactory);
    RuleFactory.createAndAddRule(context, ruleClass, kwargs, ast);
  }

  private static RuleClass getBuiltInRuleClass(String ruleClassName, RuleFactory ruleFactory) {
    if (ruleFactory.getRuleClassNames().contains(ruleClassName)) {
      return ruleFactory.getRuleClass(ruleClassName);
    }
    throw new IllegalArgumentException("no such rule class: "  + ruleClassName);
  }

  /**
   * Get the PackageContext by looking up in the environment.
   */
  public static PackageContext getContext(Environment env, FuncallExpression ast)
      throws EvalException {
    try {
      return (PackageContext) env.lookup(PKG_CONTEXT);
    } catch (NoSuchVariableException e) {
      // if PKG_CONTEXT is missing, we're not called from a BUILD file. This happens if someone
      // uses native.some_func() in the wrong place.
      throw new EvalException(ast.getLocation(),
          "The native module cannot be accessed from here. "
          + "Wrap the function in a macro and call it from a BUILD file");
    }
  }

  /**
   * Returns a function-value implementing the build rule "ruleClass" (e.g. cc_library) in the
   * specified package context.
   */
  private static Function newRuleFunction(final RuleFactory ruleFactory,
                                          final String ruleClass) {
    return new AbstractFunction(ruleClass) {
      @Override
      public Object call(List<Object> args, Map<String, Object> kwargs, FuncallExpression ast,
          Environment env)
          throws EvalException {
        if (!args.isEmpty()) {
          throw new EvalException(ast.getLocation(),
              "build rules do not accept positional parameters");
        }

        try {
          addRule(ruleFactory, ruleClass, getContext(env, ast), kwargs, ast);
        } catch (RuleFactory.InvalidRuleException | Package.NameConflictException e) {
          throw new EvalException(ast.getLocation(), e.getMessage());
        }
        return Environment.NONE;
      }
    };
  }

  /**
   * Returns a new environment populated with common entries that can be shared
   * across packages and that don't require the context.
   */
  private static Environment newGlobalEnvironment() {
    Environment env = new Environment();
    MethodLibrary.setupMethodEnvironment(env);
    return env;
  }

  /****************************************************************************
   * Package creation.
   */

  /**
   * Loads, scans parses and evaluates the build file at "buildFile", and
   * creates and returns a Package builder instance capable of building a package identified by
   * "packageId".
   *
   * <p>This method returns a builder to allow the caller to do additional work, if necessary.
   *
   * <p>This method assumes "packageId" is a valid package name according to the
   * {@link LabelValidator#validatePackageName} heuristic.
   *
   * <p>See {@link #evaluateBuildFile} for information on AST retention.
   *
   * <p>Executes {@code globber.onCompletion()} on completion and executes
   * {@code globber.onInterrupt()} on an {@link InterruptedException}.
   */
  private Package.LegacyBuilder createPackage(ExternalPackage externalPkg,
      PackageIdentifier packageId, Path buildFile, List<Statement> preludeStatements,
      ParserInputSource inputSource, Map<PathFragment, SkylarkEnvironment> imports,
      ImmutableList<Label> skylarkFileDependencies, CachingPackageLocator locator,
      RuleVisibility defaultVisibility, Globber globber)
          throws InterruptedException {
    StoredEventHandler localReporter = new StoredEventHandler();
    Preprocessor.Result preprocessingResult = preprocess(packageId, buildFile, inputSource, globber,
        localReporter);
    return createPackageFromPreprocessingResult(externalPkg, packageId, buildFile,
        preprocessingResult, localReporter.getEvents(), preludeStatements, imports,
        skylarkFileDependencies, locator, defaultVisibility, globber);
  }

  /**
   * Same as {@link #createPackage}, but using a {@link Preprocessor.Result} from
   * {@link #preprocess}.
   *
   * <p>Executes {@code globber.onCompletion()} on completion and executes
   * {@code globber.onInterrupt()} on an {@link InterruptedException}.
   */
  // Used outside of bazel!
  public Package.LegacyBuilder createPackageFromPreprocessingResult(Package externalPkg,
      PackageIdentifier packageId,
      Path buildFile,
      Preprocessor.Result preprocessingResult,
      Iterable<Event> preprocessingEvents,
      List<Statement> preludeStatements,
      Map<PathFragment, SkylarkEnvironment> imports,
      ImmutableList<Label> skylarkFileDependencies,
      CachingPackageLocator locator,
      RuleVisibility defaultVisibility,
      Globber globber) throws InterruptedException {
    StoredEventHandler localReporter = new StoredEventHandler();
    // Run the lexer and parser with a local reporter, so that errors from other threads do not
    // show up below. Merge the local and global reporters afterwards.
    // Logged messages are used as a testability hook tracing the parsing progress
    LOG.fine("Starting to parse " + packageId);
    BuildFileAST buildFileAST = BuildFileAST.parseBuildFile(
        preprocessingResult.result, preludeStatements, localReporter, locator, false);
    LOG.fine("Finished parsing of " + packageId);

    MakeEnvironment.Builder makeEnv = new MakeEnvironment.Builder();
    if (platformSetRegexps != null) {
      makeEnv.setPlatformSetRegexps(platformSetRegexps);
    }
    try {
      // At this point the package is guaranteed to exist.  It may have parse or
      // evaluation errors, resulting in a diminished number of rules.
      prefetchGlobs(packageId, buildFileAST, preprocessingResult.preprocessed,
          buildFile, globber, defaultVisibility, makeEnv);
      return evaluateBuildFile(
          externalPkg, packageId, buildFileAST, buildFile, globber,
          Iterables.concat(preprocessingEvents, localReporter.getEvents()),
          defaultVisibility, preprocessingResult.containsErrors,
          preprocessingResult.containsTransientErrors, makeEnv, imports, skylarkFileDependencies);
    } catch (InterruptedException e) {
      globber.onInterrupt();
      throw e;
    } finally {
      globber.onCompletion();
    }
  }

  /**
   * Same as {@link #createPackage}, but does the required validation of "packageName" first,
   * throwing a {@link NoSuchPackageException} if the name is invalid.
   */
  @VisibleForTesting
  public Package createPackageForTesting(PackageIdentifier packageId,
      Path buildFile, CachingPackageLocator locator, EventHandler eventHandler)
          throws NoSuchPackageException, InterruptedException {
    String error = LabelValidator.validatePackageName(
        packageId.getPackageFragment().getPathString());
    if (error != null) {
      throw new BuildFileNotFoundException(
          packageId.toString(), "illegal package name: '" + packageId + "' (" + error + ")");
    }
    ParserInputSource inputSource = maybeGetParserInputSource(buildFile, eventHandler);
    if (inputSource == null) {
      throw new BuildFileContainsErrorsException(packageId.toString(), "IOException occured");
    }

    Package result = createPackage((new ExternalPackage.Builder(
        buildFile.getRelative("WORKSPACE"))).build(), packageId, buildFile,
        ImmutableList.<Statement>of(), inputSource, ImmutableMap.<PathFragment,
        SkylarkEnvironment>of(), ImmutableList.<Label>of(), locator, ConstantRuleVisibility.PUBLIC,
        createLegacyGlobber(buildFile.getParentDirectory(), packageId, locator)).build();
    Event.replayEventsOn(eventHandler, result.getEvents());
    return result;
  }

  /** Preprocesses the given BUILD file. */
  // Used outside of bazel!
  public Preprocessor.Result preprocess(
      PackageIdentifier packageId,
      Path buildFile,
      CachingPackageLocator locator,
      EventHandler eventHandler) throws InterruptedException {
    ParserInputSource inputSource = maybeGetParserInputSource(buildFile, eventHandler);
    if (inputSource == null) {
      return Preprocessor.Result.transientError(buildFile);
    }
    Globber globber = createLegacyGlobber(buildFile.getParentDirectory(), packageId, locator);
    try {
      return preprocess(packageId, buildFile, inputSource, globber, eventHandler);
    } finally {
      globber.onCompletion();
    }
  }

  /**
   * Preprocesses the given BUILD file, executing {@code globber.onInterrupt()} on an
   * {@link InterruptedException}.
   */
  // Used outside of bazel!
  public Preprocessor.Result preprocess(
      PackageIdentifier packageId,
      Path buildFile,
      ParserInputSource inputSource,
      Globber globber,
      EventHandler eventHandler) throws InterruptedException {
    Preprocessor preprocessor = preprocessorFactory.getPreprocessor();
    if (preprocessor == null) {
      return Preprocessor.Result.noPreprocessing(inputSource);
    }
    try {
      return preprocessor.preprocess(inputSource, packageId.toString(), globber, eventHandler,
          globalEnv, ruleFactory.getRuleClassNames());
    } catch (IOException e) {
      eventHandler.handle(Event.error(Location.fromFile(buildFile),
                     "preprocessing failed: " + e.getMessage()));
      return Preprocessor.Result.transientError(buildFile);
    } catch (InterruptedException e) {
      globber.onInterrupt();
      throw e;
    }
  }

  // Used outside of bazel!
  public LegacyGlobber createLegacyGlobber(Path packageDirectory, PackageIdentifier packageId,
      CachingPackageLocator locator) {
    return new LegacyGlobber(new GlobCache(packageDirectory, packageId, locator, syscalls,
        threadPool));
  }

  @Nullable
  private ParserInputSource maybeGetParserInputSource(Path buildFile, EventHandler eventHandler) {
    try {
      return ParserInputSource.create(buildFile);
    } catch (IOException e) {
      eventHandler.handle(Event.error(Location.fromFile(buildFile), e.getMessage()));
      return null;
    }
  }

  /**
   * This tuple holds the current package builder, current lexer, etc, for the
   * duration of the evaluation of one BUILD file. (We use a PackageContext
   * object in preference to storing these values in mutable fields of the
   * PackageFactory.)
   *
   * <p>PLEASE NOTE: references to PackageContext objects are held by many
   * Function closures, but should become unreachable once the Environment is
   * discarded at the end of evaluation.  Please be aware of your memory
   * footprint when making changes here!
   */
  public static class PackageContext {

    final Package.LegacyBuilder pkgBuilder;
    final Globber globber;
    final EventHandler eventHandler;

    @VisibleForTesting
    public PackageContext(Package.LegacyBuilder pkgBuilder, Globber globber,
        EventHandler eventHandler) {
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
     * Returns the MakeEnvironment Builder of this Package.
     */
    public MakeEnvironment.Builder getMakeEnvironment() {
      return pkgBuilder.getMakeEnvironment();
    }
  }

  /**
   * Returns the list of native rule functions created using the {@link RuleClassProvider}
   * of this {@link PackageFactory}.
   */
  public ImmutableList<Function> collectNativeRuleFunctions() {
    ImmutableList.Builder<Function> builder = ImmutableList.builder();
    for (String ruleClass : ruleFactory.getRuleClassNames()) {
      builder.add(newRuleFunction(ruleFactory, ruleClass));
    }
    builder.add(newPackageFunction(packageArguments));
    for (EnvironmentExtension extension : environmentExtensions) {
      builder.addAll(extension.nativeModuleFunctions());
    }
    return builder.build();
  }

  private void buildPkgEnv(Environment pkgEnv, String packageName,
      PackageContext context, RuleFactory ruleFactory) {
    pkgEnv.update("distribs", newDistribsFunction(context));
    pkgEnv.update("glob", newGlobFunction(context, /*async=*/false));
    pkgEnv.update("mocksubinclude", newMockSubincludeFunction(context));
    pkgEnv.update("licenses", newLicensesFunction(context));
    pkgEnv.update("exports_files", newExportsFilesFunction());
    pkgEnv.update("package_group", newPackageGroupFunction());
    pkgEnv.update("package", newPackageFunction(packageArguments));
    pkgEnv.update("subinclude", newSubincludeFunction());
    pkgEnv.update("environment_group", newEnvironmentGroupFunction(context));

    pkgEnv.update("PACKAGE_NAME", packageName);

    for (String ruleClass : ruleFactory.getRuleClassNames()) {
      Function ruleFunction = newRuleFunction(ruleFactory, ruleClass);
      pkgEnv.update(ruleClass, ruleFunction);
    }

    for (EnvironmentExtension extension : environmentExtensions) {
      extension.update(pkgEnv, context.pkgBuilder.getBuildFileLabel());
    }
  }

  /**
   * Constructs a Package instance, evaluates the BUILD-file AST inside the
   * build environment, and populates the package with Rule instances as it
   * goes.  As with most programming languages, evaluation stops when an
   * exception is encountered: no further rules after the point of failure will
   * be constructed.  We assume that rules constructed before the point of
   * failure are valid; this assumption is not entirely correct, since a
   * "vardef" after a rule declaration can affect the behavior of that rule.
   *
   * <p>Rule attribute checking is performed during evaluation. Each attribute
   * must conform to the type specified for that <i>(rule class, attribute
   * name)</i> pair.  Errors reported at this stage include: missing value for
   * mandatory attribute, value of wrong type.  Such error cause Rule
   * construction to be aborted, so the resulting package will have missing
   * members.
   *
   * @see PackageFactory#PackageFactory
   */
  @VisibleForTesting // used by PackageFactoryApparatus
  public Package.LegacyBuilder evaluateBuildFile(Package externalPkg,
      PackageIdentifier packageId, BuildFileAST buildFileAST, Path buildFilePath, Globber globber,
      Iterable<Event> pastEvents, RuleVisibility defaultVisibility, boolean containsError,
      boolean containsTransientError, MakeEnvironment.Builder pkgMakeEnv,
      Map<PathFragment, SkylarkEnvironment> imports,
      ImmutableList<Label> skylarkFileDependencies) throws InterruptedException {
    // Important: Environment should be unreachable by the end of this method!
    StoredEventHandler eventHandler = new StoredEventHandler();
    Environment pkgEnv = new Environment(globalEnv, eventHandler);

    Package.LegacyBuilder pkgBuilder =
        new Package.LegacyBuilder(packageId)
        .setGlobber(globber)
        .setFilename(buildFilePath)
        .setMakeEnv(pkgMakeEnv)
        .setDefaultVisibility(defaultVisibility)
        // "defaultVisibility" comes from the command line. Let's give the BUILD file a chance to
        // set default_visibility once, be reseting the PackageBuilder.defaultVisibilitySet flag.
        .setDefaultVisibilitySet(false)
        .setSkylarkFileDependencies(skylarkFileDependencies)
        .setWorkspaceName(externalPkg.getWorkspaceName());

    Event.replayEventsOn(eventHandler, pastEvents);

    // Stuff that closes over the package context:`
    PackageContext context = new PackageContext(pkgBuilder, globber, eventHandler);
    buildPkgEnv(pkgEnv, packageId.toString(), context, ruleFactory);

    if (containsError) {
      pkgBuilder.setContainsErrors();
    }

    if (containsTransientError) {
      pkgBuilder.setContainsTemporaryErrors();
    }

    if (!validatePackageIdentifier(packageId, buildFileAST.getLocation(), eventHandler)) {
      pkgBuilder.setContainsErrors();
    }

    pkgEnv.setImportedExtensions(imports);
    pkgEnv.updateAndPropagate(PKG_CONTEXT, context);
    pkgEnv.updateAndPropagate(Environment.PKG_NAME, packageId.toString());

    if (!validateAssignmentStatements(pkgEnv, buildFileAST, eventHandler)) {
      pkgBuilder.setContainsErrors();
    }

    if (buildFileAST.containsErrors()) {
      pkgBuilder.setContainsErrors();
    }

    // TODO(bazel-team): (2009) the invariant "if errors are reported, mark the package
    // as containing errors" is strewn all over this class.  Refactor to use an
    // event sensor--and see if we can simplify the calling code in
    // createPackage().
    if (!buildFileAST.exec(pkgEnv, eventHandler)) {
      pkgBuilder.setContainsErrors();
    }

    pkgBuilder.addEvents(eventHandler.getEvents());
    return pkgBuilder;
  }

  /**
   * Visit all targets and expand the globs in parallel.
   */
  private void prefetchGlobs(PackageIdentifier packageId, BuildFileAST buildFileAST,
      boolean wasPreprocessed, Path buildFilePath, Globber globber,
      RuleVisibility defaultVisibility, MakeEnvironment.Builder pkgMakeEnv)
      throws InterruptedException {
    if (wasPreprocessed) {
      // No point in prefetching globs here: preprocessing implies eager evaluation
      // of all globs.
      return;
    }
    // Important: Environment should be unreachable by the end of this method!
    Environment pkgEnv = new Environment();

    Package.LegacyBuilder pkgBuilder =
        new Package.LegacyBuilder(packageId)
        .setFilename(buildFilePath)
        .setMakeEnv(pkgMakeEnv)
        .setDefaultVisibility(defaultVisibility)
        // "defaultVisibility" comes from the command line. Let's give the BUILD file a chance to
        // set default_visibility once, be reseting the PackageBuilder.defaultVisibilitySet flag.
        .setDefaultVisibilitySet(false);

    // Stuff that closes over the package context:
    PackageContext context = new PackageContext(pkgBuilder, globber, NullEventHandler.INSTANCE);
    buildPkgEnv(pkgEnv, packageId.toString(), context, ruleFactory);
    pkgEnv.update("glob", newGlobFunction(context, /*async=*/true));
    // The Fileset function is heavyweight in that it can run glob(). Avoid this during the
    // preloading phase.
    pkgEnv.remove("FilesetEntry");

    buildFileAST.exec(pkgEnv, NullEventHandler.INSTANCE);
  }


  /**
   * Tests a build AST to ensure that it contains no assignment statements that
   * redefine built-in build rules.
   *
   * @param pkgEnv a package environment initialized with all of the built-in
   *        build rules
   * @param ast the build file AST to be tested
   * @param eventHandler a eventHandler where any errors should be logged
   * @return true if the build file contains no redefinitions of built-in
   *         functions
   */
  private static boolean validateAssignmentStatements(Environment pkgEnv,
                                                      BuildFileAST ast,
                                                      EventHandler eventHandler) {
    for (Statement stmt : ast.getStatements()) {
      if (stmt instanceof AssignmentStatement) {
        Expression lvalue = ((AssignmentStatement) stmt).getLValue().getExpression();
        if (!(lvalue instanceof Ident)) {
          continue;
        }
        String target = ((Ident) lvalue).getName();
        if (pkgEnv.lookup(target, null) != null) {
          eventHandler.handle(Event.error(stmt.getLocation(), "Reassignment of builtin build "
              + "function '" + target + "' not permitted"));
          return false;
        }
      }
    }
    return true;
  }

  // Reports an error and returns false iff package identifier was illegal.
  private static boolean validatePackageIdentifier(PackageIdentifier packageId, Location location,
      EventHandler eventHandler) {
    String error = LabelValidator.validatePackageName(packageId.getPackageFragment().toString());
    if (error != null) {
      eventHandler.handle(Event.error(location, error));
      return false; // Invalid package name 'foo'
    }
    return true;
  }
}
