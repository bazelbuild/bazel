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

import static java.lang.Math.max;

import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.packages.MacroClass;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.PackagePiece;
import com.google.devtools.build.lib.packages.PackagePieceIdentifier;
import com.google.devtools.build.lib.packages.PackageValidator.InvalidPackagePieceException;
import com.google.devtools.build.lib.server.FailureDetails.PackageLoading.Code;
import com.google.devtools.build.lib.skyframe.MacroInstanceFunction.MacroInstanceFunctionException;
import com.google.devtools.build.lib.skyframe.PackageFunction.PackageFunctionException;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
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
      throws EvalMacroFunctionException,
          PackageFunctionException,
          MacroInstanceFunctionException,
          InterruptedException {
    PackagePieceIdentifier.ForMacro key = (PackagePieceIdentifier.ForMacro) skyKey.argument();

    // Get this package's top-level package piece, for common metadata and declarations shared by
    // all package pieces of the package.
    PackagePieceValue.ForBuildFile buildFileValue =
        (PackagePieceValue.ForBuildFile)
            env.getValueOrThrow(getBuildFileKey(key), PackageFunctionException.class);
    if (buildFileValue == null) {
      return null;
    }

    // Get the macro instance (owned by the parent package piece) which we will be expanding to
    // produce this package piece.
    MacroInstanceValue macroInstanceValue =
        (MacroInstanceValue)
            env.getValueOrThrow(
                new MacroInstanceValue.Key(key.getParentIdentifier(), key.getInstanceName()),
                EvalMacroFunctionException.class,
                PackageFunctionException.class,
                MacroInstanceFunctionException.class);
    if (macroInstanceValue == null) {
      return null;
    }

    // TODO(https://github.com/bazelbuild/bazel/issues/23852): support finalizers. Requires
    // recursively expanding all non-finalizer macros in current package and creating a map of all
    // non-finalizer-defined rules for native.existing_rules().
    if (macroInstanceValue.macroInstance().getMacroClass().isFinalizer()) {
      throw new EvalMacroFunctionException(
          new InvalidPackagePieceException(
              key, "finalizers not yet supported under lazy macro expansion"),
          Transience.PERSISTENT);
    }

    // Expand the macro.
    long startTimeNanos = BlazeClock.nanoTime();
    PackagePiece.ForMacro.Builder packagePieceBuilder =
        packageFactory.newPackagePieceForMacroBuilder(
            macroInstanceValue.macroInstance(),
            key.getParentIdentifier(),
            buildFileValue.getPackagePiece(),
            buildFileValue.starlarkSemantics(),
            buildFileValue.mainRepositoryMapping(),
            cpuBoundSemaphore.get(),
            buildFileValue.generatorMap());
    try {
      MacroClass.executeMacroImplementation(
          macroInstanceValue.macroInstance(),
          packagePieceBuilder,
          buildFileValue.starlarkSemantics());
    } catch (EvalException e) {
      packagePieceBuilder
          .getLocalEventHandler()
          .handle(Package.error(null, e.getMessageWithStack(), Code.STARLARK_EVAL_ERROR));
      packagePieceBuilder.setContainsErrors();
    }
    long loadTimeNanos = max(BlazeClock.nanoTime() - startTimeNanos, 0L);

    try {
      packagePieceBuilder.buildPartial();
      // TODO(https://github.com/bazelbuild/bazel/issues/23852): verify labels using
      // PackageFunction#handleLabelsCrossingSubpackagesAndPropagateInconsistentFilesystemExceptions
    } catch (NoSuchPackageException e) {
      throw new EvalMacroFunctionException(e, Transience.PERSISTENT);
    }
    PackagePiece.ForMacro packagePiece = packagePieceBuilder.finishBuild();

    try {
      packageFactory.afterDoneLoadingPackagePiece(
          packagePiece, buildFileValue.starlarkSemantics(), loadTimeNanos, env.getListener());
    } catch (InvalidPackagePieceException e) {
      throw new EvalMacroFunctionException(e, Transience.PERSISTENT);
    }

    return new PackagePieceValue.ForMacro(packagePiece);
  }

  private static PackagePieceIdentifier.ForBuildFile getBuildFileKey(
      PackagePieceIdentifier.ForMacro key) {
    do {
      PackagePieceIdentifier parent = key.getParentIdentifier();
      if (parent instanceof PackagePieceIdentifier.ForBuildFile buildFileKey) {
        return buildFileKey;
      } else {
        key = (PackagePieceIdentifier.ForMacro) parent;
      }
    } while (true);
  }

  public static final class EvalMacroFunctionException extends SkyFunctionException {
    EvalMacroFunctionException(Exception cause, Transience transience) {
      super(cause, transience);
    }
  }
}
