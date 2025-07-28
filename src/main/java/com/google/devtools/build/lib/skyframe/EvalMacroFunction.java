// Copyright 2025 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skyframe;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.collect.ImmutableMap.toImmutableMap;
import static java.lang.Math.max;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.packages.MacroClass;
import com.google.devtools.build.lib.packages.MacroInstance;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchPackagePieceException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.PackageLoadingListener.Metrics;
import com.google.devtools.build.lib.packages.PackagePiece;
import com.google.devtools.build.lib.packages.PackagePieceIdentifier;
import com.google.devtools.build.lib.packages.PackageValidator.InvalidPackagePieceException;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.server.FailureDetails.PackageLoading.Code;
import com.google.devtools.build.lib.skyframe.MacroInstanceFunction.NoSuchMacroInstanceException;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.SkyframeLookupResult;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicReference;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;

/**
 * A SkyFunction that evaluates a symbolic macro instance, identified by a
 * {PackagePieceIdentifier.ForMacro}, and produces a {@link PackagePieceValue.ForMacro}.
 */
final class EvalMacroFunction implements SkyFunction {
  private final PackageFactory packageFactory;
  private final AtomicReference<Semaphore> cpuBoundSemaphore;

  public EvalMacroFunction(
      PackageFactory packageFactory, AtomicReference<Semaphore> cpuBoundSemaphore) {
    this.packageFactory = packageFactory;
    this.cpuBoundSemaphore = cpuBoundSemaphore;
  }

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws EvalMacroFunctionException, InterruptedException {
    PackagePieceIdentifier.ForMacro key = (PackagePieceIdentifier.ForMacro) skyKey.argument();
    // Get the common metadata and declarations shared by all package pieces of the package.
    PackageDeclarationsValue packageDeclarationsValue;
    try {
      packageDeclarationsValue =
          (PackageDeclarationsValue)
              env.getValueOrThrow(
                  new PackageDeclarationsValue.Key(key.getPackageIdentifier()),
                  NoSuchPackageException.class,
                  NoSuchPackagePieceException.class);
    } catch (NoSuchPackageException e) {
      throw new EvalMacroFunctionException(e);
    } catch (NoSuchPackagePieceException e) {
      throw new EvalMacroFunctionException(e);
    }
    if (packageDeclarationsValue == null) {
      return null;
    }

    // Get the macro instance (owned by the parent package piece) which we will be expanding to
    // produce this package piece.
    MacroInstanceValue macroInstanceValue;
    try {
      macroInstanceValue =
          (MacroInstanceValue)
              env.getValueOrThrow(
                  new MacroInstanceValue.Key(key.getParentIdentifier(), key.getInstanceName()),
                  NoSuchPackageException.class,
                  NoSuchPackagePieceException.class,
                  NoSuchMacroInstanceException.class);
    } catch (NoSuchPackageException e) {
      throw new EvalMacroFunctionException(e);
    } catch (NoSuchPackagePieceException e) {
      throw new EvalMacroFunctionException(e);
    } catch (NoSuchMacroInstanceException e) {
      throw new EvalMacroFunctionException(e);
    }
    if (macroInstanceValue == null) {
      return null;
    }
    MacroInstance macroInstance = macroInstanceValue.macroInstance();

    // Non-null iff the macro is a finalizer.
    NonFinalizerPackagePiecesValue nonFinalizerPackagePiecesValue = null;
    // Non-null iff the macro is a finalizer and finalizer dependencies were computed without error.
    @Nullable ImmutableMap<String, Rule> existingRulesMapForFinalizer = null;

    if (macroInstance.getMacroClass().isFinalizer()) {
      try {
        nonFinalizerPackagePiecesValue =
            (NonFinalizerPackagePiecesValue)
                env.getValueOrThrow(
                    new NonFinalizerPackagePiecesValue.Key(key.getPackageIdentifier()),
                    NoSuchPackageException.class,
                    NoSuchPackagePieceException.class,
                    NoSuchMacroInstanceException.class);
      } catch (NoSuchPackageException e) {
        throw new EvalMacroFunctionException(e);
      } catch (NoSuchPackagePieceException e) {
        throw new EvalMacroFunctionException(e);
      } catch (NoSuchMacroInstanceException e) {
        throw new EvalMacroFunctionException(e);
      }
      if (nonFinalizerPackagePiecesValue == null) {
        // Restart
        return null;
      } else if (!nonFinalizerPackagePiecesValue.containsErrors()) {
        existingRulesMapForFinalizer =
            nonFinalizerPackagePiecesValue.targets().entrySet().stream()
                .filter(e -> e.getValue() instanceof Rule)
                .collect(toImmutableMap(Map.Entry::getKey, e -> (Rule) e.getValue()));
      }
    }

    // Expand the macro.
    long startTimeNanos = BlazeClock.nanoTime();
    PackagePiece.ForMacro.Builder packagePieceBuilder =
        packageFactory.newPackagePieceForMacroBuilder(
            packageDeclarationsValue.metadata(),
            packageDeclarationsValue.declarations(),
            macroInstance,
            key.getParentIdentifier(),
            packageDeclarationsValue.starlarkSemantics(),
            packageDeclarationsValue.mainRepositoryMapping(),
            cpuBoundSemaphore.get(),
            existingRulesMapForFinalizer);
    if (nonFinalizerPackagePiecesValue != null && nonFinalizerPackagePiecesValue.containsErrors()) {
      // Error within one non-finalizer package piece or a name conflict between package pieces. It
      // was already reported as an event with stack trace by the computation of the
      // PackagePieceValue or NonFinalizerPackagePiecesValue, so we don't need to repeat the stack
      // trace - just a brief summary.
      if (!nonFinalizerPackagePiecesValue.getErrorKeys().isEmpty()) {
        PackagePieceIdentifier errorKey = nonFinalizerPackagePiecesValue.getErrorKeys().getFirst();
        PackagePiece errorPiece = nonFinalizerPackagePiecesValue.getPackagePieces().get(errorKey);
        handleFinalizerDependencyError(
            packagePieceBuilder, "error in " + errorPiece.getShortDescription());
      } else {
        handleFinalizerDependencyError(
            packagePieceBuilder,
            nonFinalizerPackagePiecesValue
                .nameConflictBetweenPackagePiecesException()
                .getMessage());
      }
      packagePieceBuilder.setContainsErrors();
    } else {
      try {
        MacroClass.executeMacroImplementation(
            macroInstance, packagePieceBuilder, packageDeclarationsValue.starlarkSemantics());
      } catch (EvalException e) {
        packagePieceBuilder
            .getLocalEventHandler()
            .handle(
                Package.error(
                    e.getInnermostLocation(), e.getMessageWithStack(), Code.STARLARK_EVAL_ERROR));
        packagePieceBuilder.setContainsErrors();
      }
    }
    long loadTimeNanos = max(BlazeClock.nanoTime() - startTimeNanos, 0L);

    try {
      packagePieceBuilder.buildPartial();
      // TODO(https://github.com/bazelbuild/bazel/issues/23852): verify labels using
      // PackageFunction#handleLabelsCrossingSubpackagesAndPropagateInconsistentFilesystemExceptions
    } catch (NoSuchPackageException e) {
      throw new EvalMacroFunctionException(e);
    }
    PackagePiece.ForMacro packagePiece = packagePieceBuilder.finishBuild();
    packagePieceBuilder.getLocalEventHandler().replayOn(env.getListener());

    try {
      packageFactory.afterDoneLoadingPackagePiece(
          packagePiece,
          packageDeclarationsValue.starlarkSemantics(),
          new Metrics(
              loadTimeNanos,
              // Symbolic macros don't use `native.glob`.
              /* globFilesystemOperationCost= */ 0L),
          env.getListener());
    } catch (InvalidPackagePieceException e) {
      throw new EvalMacroFunctionException(e);
    }

    return new PackagePieceValue.ForMacro(packagePiece);
  }

  private static void handleFinalizerDependencyError(
      PackagePiece.ForMacro.Builder packagePieceBuilder, String message) {
    packagePieceBuilder
        .getLocalEventHandler()
        .handle(
            Package.error(
                packagePieceBuilder.getPackagePiece().getEvaluatedMacro().getBuildFileLocation(),
                String.format(
                    "cannot compute %s: %s",
                    packagePieceBuilder.getPackagePiece().getShortDescription(), message),
                Code.STARLARK_EVAL_ERROR));
  }

  /**
   * A mutable {@link PackagePieces} implementation which produces its collection of package pieces
   * by recursively expanding a starting collection of package piece identifiers.
   *
   * <p>Intended to be used as part of a skyfunction compute() implementation. The {@link
   * RecursiveExpander} lacks any kind of invalidation of already-expanded package pieces, so it
   * cannot be reused across multiple skyframe evaluations.
   */
  static class RecursiveExpander implements PackagePieces {
    private final LinkedHashMap<PackagePieceIdentifier, PackagePiece> packagePieces =
        new LinkedHashMap<>();
    private final LinkedHashSet<PackagePieceIdentifier> errorKeys = new LinkedHashSet<>();

    @Override
    public ImmutableMap<PackagePieceIdentifier, PackagePiece> getPackagePieces() {
      return ImmutableMap.copyOf(packagePieces);
    }

    @Override
    public PackagePiece.ForBuildFile getPackagePieceForBuildFile() {
      return (PackagePiece.ForBuildFile) packagePieces.values().iterator().next();
    }

    @Override
    public ImmutableList<PackagePieceIdentifier> getErrorKeys() {
      return ImmutableList.copyOf(errorKeys);
    }

    /**
     * Recursively expands the pieces of a package. Intended for inlining into skyfunction
     * implementations.
     *
     * @param pkgId the package whose pieces are being expanded
     * @param env the skyframe environment
     * @return this expander on success, or null to signal a skyframe restart.
     */
    @Nullable
    RecursiveExpander expand(PackageIdentifier pkgId, Environment env, boolean expandFinalizers)
        throws NoSuchPackageException,
            NoSuchPackagePieceException,
            NoSuchMacroInstanceException,
            InterruptedException {
      return expand(
          ImmutableList.of(new PackagePieceIdentifier.ForBuildFile(pkgId)), env, expandFinalizers);
    }

    /**
     * Performs "opportunistic BFS" recursive expansion of the given keys: expands in BFS order
     * (siblings ordered by name) as far as possible, skipping missing values, and then signals a
     * skyframe restart if any values were missing. Once all missing values have been obtained, the
     * final evaluation of this function - one which does not trigger a restart - will collect
     * package pieces in BFS order.
     *
     * @param keys set of keys to expand. If the expander is empty, must contain a single {@link
     *     PackagePieceIdentifier.ForBuildFile}. Otherwise, must contain package piece keys of the
     *     same depth, with siblings ordered by name.
     * @return this expander on success, or null to signal a skyframe restart.
     */
    // TODO(https://github.com/bazelbuild/bazel/issues/23852) - use state machine to reduce restart
    // cost?
    @Nullable
    private RecursiveExpander expand(
        Collection<? extends PackagePieceIdentifier> keys,
        Environment env,
        boolean expandFinalizers)
        throws NoSuchPackageException,
            NoSuchPackagePieceException,
            NoSuchMacroInstanceException,
            InterruptedException {
      if (keys.isEmpty()) {
        return this;
      }
      if (packagePieces.isEmpty()) {
        checkArgument(
            keys.size() == 1
                && keys.iterator().next() instanceof PackagePieceIdentifier.ForBuildFile,
            "expansion must start from a PackagePieceIdentifier.ForBuildFile");
      }
      boolean valuesMissing = false;
      SkyframeLookupResult lookupResult = env.getValuesAndExceptions(keys);
      ImmutableList.Builder<PackagePieceIdentifier.ForMacro> childKeys = ImmutableList.builder();
      for (PackagePieceIdentifier key : keys) {
        PackagePieceValue packagePieceValue =
            (PackagePieceValue)
                lookupResult.getOrThrow(
                    key,
                    NoSuchPackageException.class,
                    NoSuchPackagePieceException.class,
                    NoSuchMacroInstanceException.class);
        if (packagePieceValue == null) {
          valuesMissing = true;
          continue;
        }
        packagePieces.put(key, packagePieceValue.getPackagePiece());
        if (packagePieceValue.getPackagePiece().containsErrors()) {
          errorKeys.add(key);
        } else {
          for (MacroInstance childMacroInstance : packagePieceValue.getPackagePiece().getMacros()) {
            PackagePieceIdentifier.ForMacro childKey =
                new PackagePieceIdentifier.ForMacro(
                    key.getPackageIdentifier(), key, childMacroInstance.getName());
            if (packagePieces.containsKey(childKey)) {
              // Already expanded.
              continue;
            }
            if (expandFinalizers || !childMacroInstance.getMacroClass().isFinalizer()) {
              childKeys.add(childKey);
            }
          }
        }
      }
      if (expand(childKeys.build(), env, expandFinalizers) == null) {
        valuesMissing = true;
      }
      return valuesMissing ? null : this;
    }
  }

  public static final class EvalMacroFunctionException extends SkyFunctionException {
    EvalMacroFunctionException(NoSuchPackageException cause) {
      super(cause, Transience.PERSISTENT);
    }

    EvalMacroFunctionException(NoSuchPackagePieceException cause) {
      super(cause, Transience.PERSISTENT);
    }

    EvalMacroFunctionException(NoSuchMacroInstanceException cause) {
      super(cause, Transience.PERSISTENT);
    }
  }
}
