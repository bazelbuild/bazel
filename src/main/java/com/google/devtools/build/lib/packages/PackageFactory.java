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
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.actions.ThreadStateReceiver;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.concurrent.NamedForkJoinPool;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.Globber.BadGlobException;
import com.google.devtools.build.lib.packages.Package.Builder.PackageSettings;
import com.google.devtools.build.lib.packages.Package.ConfigSettingVisibilityPolicy;
import com.google.devtools.build.lib.packages.PackageValidator.InvalidPackageException;
import com.google.devtools.build.lib.packages.WorkspaceFileValue.WorkspaceFileKey;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.PackageLoading;
import com.google.devtools.build.lib.server.FailureDetails.PackageLoading.Code;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.Collection;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.Semaphore;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Module;
import net.starlark.java.eval.Mutability;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkCallable;
import net.starlark.java.eval.StarlarkFunction;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.syntax.Argument;
import net.starlark.java.syntax.CallExpression;
import net.starlark.java.syntax.Expression;
import net.starlark.java.syntax.Identifier;
import net.starlark.java.syntax.IntLiteral;
import net.starlark.java.syntax.ListExpression;
import net.starlark.java.syntax.Location;
import net.starlark.java.syntax.Program;
import net.starlark.java.syntax.StarlarkFile;
import net.starlark.java.syntax.StringLiteral;
import net.starlark.java.syntax.SyntaxError;

/**
 * The package factory is responsible for constructing Package instances from a BUILD file's
 * abstract syntax tree (AST).
 *
 * <p>A PackageFactory is a heavy-weight object; create them sparingly. Typically only one is needed
 * per client application.
 */
public final class PackageFactory {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private final RuleClassProvider ruleClassProvider;

  private SyscallCache syscallCache;

  private ForkJoinPool executor;

  private int maxDirectoriesToEagerlyVisitInGlobbing;

  private final PackageSettings packageSettings;
  private final PackageValidator packageValidator;
  private final PackageOverheadEstimator packageOverheadEstimator;
  private final PackageLoadingListener packageLoadingListener;

  /** Builder for {@link PackageFactory} instances. Intended to only be used by unit tests. */
  @VisibleForTesting
  public abstract static class BuilderForTesting {
    protected PackageValidator packageValidator = PackageValidator.NOOP_VALIDATOR;
    protected PackageOverheadEstimator packageOverheadEstimator =
        PackageOverheadEstimator.NOOP_ESTIMATOR;

    protected boolean doChecksForTesting = true;

    @CanIgnoreReturnValue
    public BuilderForTesting disableChecks() {
      this.doChecksForTesting = false;
      return this;
    }

    @CanIgnoreReturnValue
    public BuilderForTesting setPackageValidator(PackageValidator packageValidator) {
      this.packageValidator = packageValidator;
      return this;
    }

    @CanIgnoreReturnValue
    public BuilderForTesting setPackageOverheadEstimator(
        PackageOverheadEstimator packageOverheadEstimator) {
      this.packageOverheadEstimator = packageOverheadEstimator;
      return this;
    }

    public abstract PackageFactory build(RuleClassProvider ruleClassProvider, FileSystem fs);
  }

  @VisibleForTesting
  public PackageSettings getPackageSettingsForTesting() {
    return packageSettings;
  }

  /**
   * Constructs a {@code PackageFactory} instance with a specific glob path translator and rule
   * factory.
   *
   * <p>Only intended to be called by BlazeRuntime or {@link BuilderForTesting#build}.
   *
   * <p>Do not call this constructor directly in tests; please use
   * TestConstants#PACKAGE_FACTORY_BUILDER_FACTORY_FOR_TESTING instead.
   */
  // TODO(bazel-team): Maybe store `version` in the RuleClassProvider rather than passing it in
  // here? It's an extra constructor parameter that all the tests have to give, and it's only needed
  // so WorkspaceFactory can add an extra top-level builtin.
  public PackageFactory(
      RuleClassProvider ruleClassProvider,
      ForkJoinPool executorForGlobbing,
      PackageSettings packageSettings,
      PackageValidator packageValidator,
      PackageOverheadEstimator packageOverheadEstimator,
      PackageLoadingListener packageLoadingListener) {
    this.ruleClassProvider = ruleClassProvider;
    this.executor = executorForGlobbing;
    this.packageSettings = packageSettings;
    this.packageValidator = packageValidator;
    this.packageOverheadEstimator = packageOverheadEstimator;
    this.packageLoadingListener = packageLoadingListener;
  }

  /** Sets the syscalls cache used in filesystem access. */
  public void setSyscallCache(SyscallCache syscallCache) {
    this.syscallCache = Preconditions.checkNotNull(syscallCache);
  }

  /**
   * Sets the max number of threads to use for globbing.
   *
   * <p>Internally there is a {@link ForkJoinPool} used for globbing. If the specified {@code
   * globbingThreads} does not match the previous value (initial value is 100), then we {@link
   * ForkJoinPool#shutdown()} the old {@link ForkJoinPool} instance and make a new one.
   */
  public void setGlobbingThreads(int globbingThreads) {
    if (executor == null) {
      executor = makeForkJoinPool(globbingThreads);
      return;
    }
    if (executor.getParallelism() == globbingThreads) {
      return;
    }
    // We don't use ForkJoinPool#shutdownNow since it has a performance bug. See
    // http://b/33482341#comment13.
    executor.shutdown();
    executor = makeForkJoinPool(globbingThreads);
  }

  public static ForkJoinPool makeDefaultSizedForkJoinPoolForGlobbing() {
    return makeForkJoinPool(/*globbingThreads=*/ 100);
  }

  private static ForkJoinPool makeForkJoinPool(int globbingThreads) {
    return NamedForkJoinPool.newNamedPool("globbing pool", globbingThreads);
  }

  /**
   * Sets the number of directories to eagerly traverse on the first glob for a given package, in
   * order to warm the filesystem. -1 means do no eager traversal. See {@link
   * com.google.devtools.build.lib.pkgcache.PackageOptions#maxDirectoriesToEagerlyVisitInGlobbing}.
   * -2 means do the eager traversal using the regular globbing infrastructure, i.e. sharing the
   * globbing threads and caching the actual glob results.
   */
  public void setMaxDirectoriesToEagerlyVisitInGlobbing(
      int maxDirectoriesToEagerlyVisitInGlobbing) {
    this.maxDirectoriesToEagerlyVisitInGlobbing = maxDirectoriesToEagerlyVisitInGlobbing;
  }

  /** Returns the {@link RuleClassProvider} of this {@link PackageFactory}. */
  public RuleClassProvider getRuleClassProvider() {
    return ruleClassProvider;
  }

  public Package.Builder newExternalPackageBuilder(
      WorkspaceFileKey workspaceFileKey,
      String workspaceName,
      RepositoryMapping mainRepoMapping,
      StarlarkSemantics starlarkSemantics) {
    return Package.newExternalPackageBuilder(
        packageSettings,
        workspaceFileKey,
        workspaceName,
        mainRepoMapping,
        starlarkSemantics.getBool(BuildLanguageOptions.INCOMPATIBLE_NO_IMPLICIT_FILE_EXPORT),
        starlarkSemantics.getBool(
            BuildLanguageOptions.INCOMPATIBLE_SIMPLIFY_UNCONDITIONAL_SELECTS_IN_RULE_ATTRS),
        packageOverheadEstimator);
  }

  // This function is public only for the benefit of skyframe.PackageFunction,
  // which is morally part of lib.packages, so that it can create empty packages
  // in case of error before BUILD execution. Do not call it from anywhere else.
  // TODO(adonovan): refactor Rule{Class,Factory}Test not to need this.
  public Package.Builder newPackageBuilder(
      PackageIdentifier packageId,
      RootedPath filename,
      String workspaceName,
      Optional<String> associatedModuleName,
      Optional<String> associatedModuleVersion,
      StarlarkSemantics starlarkSemantics,
      RepositoryMapping repositoryMapping,
      RepositoryMapping mainRepositoryMapping,
      @Nullable Semaphore cpuBoundSemaphore,
      @Nullable ImmutableMap<Location, String> generatorMap,
      @Nullable ConfigSettingVisibilityPolicy configSettingVisibilityPolicy,
      @Nullable Globber globber) {
    return Package.newPackageBuilder(
        packageSettings,
        packageId,
        filename,
        workspaceName,
        associatedModuleName,
        associatedModuleVersion,
        starlarkSemantics.getBool(BuildLanguageOptions.INCOMPATIBLE_NO_IMPLICIT_FILE_EXPORT),
        starlarkSemantics.getBool(
            BuildLanguageOptions.INCOMPATIBLE_SIMPLIFY_UNCONDITIONAL_SELECTS_IN_RULE_ATTRS),
        repositoryMapping,
        mainRepositoryMapping,
        cpuBoundSemaphore,
        packageOverheadEstimator,
        generatorMap,
        configSettingVisibilityPolicy,
        globber,
        /* enableNameConflictChecking= */ true);
  }

  /** Returns a new {@link NonSkyframeGlobber}. */
  // Exposed to skyframe.PackageFunction.
  public NonSkyframeGlobber createNonSkyframeGlobber(
      Path packageDirectory,
      PackageIdentifier packageId,
      ImmutableSet<PathFragment> ignoredGlobPrefixes,
      CachingPackageLocator locator,
      ThreadStateReceiver threadStateReceiverForMetrics) {
    return new NonSkyframeGlobber(
        new GlobCache(
            packageDirectory,
            packageId,
            ignoredGlobPrefixes,
            locator,
            syscallCache,
            executor,
            maxDirectoriesToEagerlyVisitInGlobbing,
            threadStateReceiverForMetrics));
  }

  /**
   * Runs final validation and administrative tasks on newly loaded package. Called by a caller of
   * {@link #executeBuildFile} after this caller has fully loaded the package.
   *
   * @throws InvalidPackageException if the package is determined to be invalid
   */
  public void afterDoneLoadingPackage(
      Package pkg,
      StarlarkSemantics starlarkSemantics,
      long loadTimeNanos,
      ExtendedEventHandler eventHandler)
      throws InvalidPackageException {

    packageValidator.validate(pkg, eventHandler);

    // Enforce limit on number of compute steps in BUILD file (b/151622307).
    long maxSteps = starlarkSemantics.get(BuildLanguageOptions.MAX_COMPUTATION_STEPS);
    long steps = pkg.getComputationSteps();
    if (maxSteps > 0 && steps > maxSteps) {
      String message =
          String.format(
              "BUILD file computation took %d steps, but --max_computation_steps=%d",
              steps, maxSteps);
      throw new InvalidPackageException(
          pkg.getPackageIdentifier(),
          message,
          DetailedExitCode.of(
              FailureDetail.newBuilder()
                  .setMessage(message)
                  .setPackageLoading(
                      PackageLoading.newBuilder()
                          .setCode(PackageLoading.Code.MAX_COMPUTATION_STEPS_EXCEEDED)
                          .build())
                  .build()));
    }

    packageLoadingListener.onLoadingCompleteAndSuccessful(pkg, starlarkSemantics, loadTimeNanos);
  }

  /**
   * Populates the Package.Builder by executing the specified BUILD file.
   *
   * <p>The package exists---we have parsed its BUILD file---but it may contain errors, either
   * arising from Starlark evaluation (such as an array index error, or a call to a built-in
   * function that fails), or reported as a side effect of a built-in function, such as rule
   * instantiation, that returns normally. A partial package is nonetheless returned in both cases,
   * although it may have fewer rules than expected.
   *
   * <p>TODO(adonovan): do not return a partial package in case of BUILD evaluation errors. Errors
   * during .bzl execution are already fatal.
   *
   * <p><b>Do not call it from elsewhere! It is not in any meaningful sense a public API.</b><br>
   * In tests, use BuildViewTestCase or PackageLoadingTestCase instead.
   *
   * <p>TODO(adonovan): move PackageFunction into this package and develop a rational API.
   */
  // This function is the sole entry point for package creation in production and tests. Do not add
  // others! It changes often, and is exposed only for the benefit of skyframe.PackageFunction,
  // which is logically part of the loading phase and should in due course be moved to lib.packages,
  // but that cannot happen until Skyframe's core interfaces have been separated.
  public void executeBuildFile(
      Package.Builder pkgBuilder,
      Program buildFileProgram,
      ImmutableList<String> globs,
      ImmutableList<String> globsWithDirs,
      ImmutableList<String> subpackages,
      ImmutableMap<String, Object> predeclared,
      ImmutableMap<String, Module> loadedModules,
      StarlarkSemantics starlarkSemantics)
      throws InterruptedException {
    Globber globber = pkgBuilder.getGlobber();

    // Prefetch glob patterns asynchronously.
    if (maxDirectoriesToEagerlyVisitInGlobbing == -2) {
      try {
        boolean allowEmpty = true;
        globber.runAsync(globs, ImmutableList.of(), Globber.Operation.FILES, allowEmpty);
        globber.runAsync(
            globsWithDirs, ImmutableList.of(), Globber.Operation.FILES_AND_DIRS, allowEmpty);
        globber.runAsync(
            subpackages, ImmutableList.of(), Globber.Operation.SUBPACKAGES, allowEmpty);
      } catch (BadGlobException ex) {
        logger.atWarning().withCause(ex).log(
            "Suppressing exception for globs=%s, globsWithDirs=%s", globs, globsWithDirs);
        // Ignore exceptions. Errors will be properly reported when the actual globbing is done.
      }
    }

    Semaphore cpuSemaphore = pkgBuilder.getCpuBoundSemaphore();
    try {
      if (cpuSemaphore != null) {
        cpuSemaphore.acquire();
      }
      executeBuildFileImpl(
          pkgBuilder, buildFileProgram, predeclared, loadedModules, starlarkSemantics);
    } catch (InterruptedException e) {
      globber.onInterrupt();
      throw e;
    } finally {
      globber.onCompletion();
      if (cpuSemaphore != null) {
        cpuSemaphore.release();
      }
    }
  }

  private void executeBuildFileImpl(
      Package.Builder pkgBuilder,
      Program buildFileProgram,
      ImmutableMap<String, Object> predeclared,
      ImmutableMap<String, Module> loadedModules,
      StarlarkSemantics semantics)
      throws InterruptedException {
    pkgBuilder.setLoads(loadedModules.values());

    try (Mutability mu = Mutability.create("package", pkgBuilder.getFilename())) {
      Module module = Module.withPredeclared(semantics, predeclared);
      StarlarkThread thread =
          StarlarkThread.create(
              mu, semantics, /* contextDescription= */ "", pkgBuilder.getSymbolGenerator());
      thread.setLoader(loadedModules::get);
      thread.setPrintHandler(Event.makeDebugPrintHandler(pkgBuilder.getLocalEventHandler()));
      pkgBuilder.storeInThread(thread);

      // TODO(b/291752414): The rule definition environment shouldn't be needed at BUILD evaluation
      // time EXCEPT for analysis_test, which needs the tools repository for use in
      // StarlarkRuleClassFunctions#createRule. So we set it here as a thread-local to be retrieved
      // by StarlarkTestingModule#analysisTest.
      thread.setThreadLocal(RuleDefinitionEnvironment.class, ruleClassProvider);
      packageValidator.configureThreadWhileLoading(thread);

      try {
        Starlark.execFileProgram(buildFileProgram, module, thread);
      } catch (EvalException ex) {
        pkgBuilder
            .getLocalEventHandler()
            .handle(Package.error(null, ex.getMessageWithStack(), Code.STARLARK_EVAL_ERROR));
        pkgBuilder.setContainsErrors();
      } catch (InterruptedException ex) {
        if (pkgBuilder.containsErrors()) {
          // Suppress the interrupted exception: we have an error of our own to return.
          Thread.currentThread().interrupt();
          logger.atInfo().withCause(ex).log(
              "Suppressing InterruptedException for Package %s because an error was also found",
              pkgBuilder.getPackageIdentifier().getCanonicalForm());
        } else {
          throw ex;
        }
      }
      pkgBuilder.setComputationSteps(thread.getExecutedSteps());
    }
  }

  /**
   * checkBuildSyntax is a static pass over the syntax tree of a BUILD (not .bzl) file.
   *
   * <p>It throws a {@link SyntaxError.Exception} if it discovers disallowed elements (see {@link
   * DotBazelFileSyntaxChecker}).
   *
   * <p>It extracts literal {@code glob(include="pattern")} patterns and adds them to {@code globs},
   * or to {@code globsWithDirs} if the call had a {@code exclude_directories=0} argument.
   *
   * <p>It records in {@code generatorNameByLocation} all calls of the form {@code f(name="foo",
   * ...)} so that any rules instantiated during the call to {@code f} can be ascribed a "generator
   * name" of {@code "foo"}.
   */
  // TODO(adonovan): restructure so that this is called from the sole place that executes BUILD
  // files. Also, make private; there's no reason for tests to call this directly.
  public static void checkBuildSyntax(
      StarlarkFile file,
      Collection<String> globs,
      Collection<String> globsWithDirs,
      Collection<String> subpackages,
      Map<Location, String> generatorNameByLocation)
      throws SyntaxError.Exception {
    new DotBazelFileSyntaxChecker("BUILD files", /* canLoadBzl= */ true) {
      // Extract literal glob patterns from calls of the form:
      //   glob(include = ["pattern"])
      //   glob(["pattern"])
      //   subpackages(include = ["pattern"])
      // This may spuriously match user-defined functions named glob or
      // subpackages; that's ok, it's only a heuristic.
      void extractGlobPatterns(CallExpression call) {
        if (call.getFunction() instanceof Identifier) {
          String functionName = ((Identifier) call.getFunction()).getName();
          if (!functionName.equals("glob") && !functionName.equals("subpackages")) {
            return;
          }

          Expression excludeDirectories = null;
          Expression include = null;
          ImmutableList<Argument> arguments = call.getArguments();
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
                if (excludeDirectories instanceof IntLiteral) {
                  Number v = ((IntLiteral) excludeDirectories).getValue();
                  if (v instanceof Integer && (Integer) v == 0) {
                    exclude = false;
                  }
                }
                if (functionName.equals("glob")) {
                  (exclude ? globs : globsWithDirs).add(pattern);
                } else {
                  subpackages.add(pattern);
                }
              }
            }
          }
        }
      }

      // Record calls of the form f(name="foo", ...)
      // so that we can later ascribe "foo" as the "generator name"
      // of any rules instantiated during the call of f.
      void recordGeneratorName(CallExpression call) {
        for (Argument arg : call.getArguments()) {
          if (arg instanceof Argument.Keyword
              && arg.getName().equals("name")
              && arg.getValue() instanceof StringLiteral) {
            generatorNameByLocation.put(
                call.getLparenLocation(), ((StringLiteral) arg.getValue()).getValue());
          }
        }
      }

      @Override
      public void visit(CallExpression node) {
        extractGlobPatterns(node);
        recordGeneratorName(node);
        // Continue traversal so as not to miss nested calls
        // like cc_binary(..., f(**kwargs), srcs=glob(...), ...).
        super.visit(node);
      }
    }.check(file);
  }

  // Install profiler hooks into Starlark interpreter.
  static {
    // parser profiler
    StarlarkFile.setParseProfiler(
        new StarlarkFile.ParseProfiler() {
          @Override
          public long start() {
            return Profiler.nanoTimeMaybe();
          }

          @Override
          public void end(long startTimeNanos, String filename) {
            Profiler.instance()
                .completeTask(startTimeNanos, ProfilerTask.STARLARK_PARSER, filename);
          }
        });

    // call profiler
    StarlarkThread.setCallProfiler(
        new StarlarkThread.CallProfiler() {
          @Override
          public long start() {
            return Profiler.nanoTimeMaybe();
          }

          @Override
          public void end(long startTimeNanos, StarlarkCallable fn) {
            Profiler.instance()
                .completeTask(
                    startTimeNanos,
                    fn instanceof StarlarkFunction
                        ? ProfilerTask.STARLARK_USER_FN
                        : ProfilerTask.STARLARK_BUILTIN_FN,
                    fn.getName());
          }
        });
  }
}
