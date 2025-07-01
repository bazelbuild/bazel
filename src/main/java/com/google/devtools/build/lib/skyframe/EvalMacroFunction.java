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
import com.google.devtools.build.lib.packages.MacroInstance;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchPackagePieceException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.PackagePiece;
import com.google.devtools.build.lib.packages.PackagePieceIdentifier;
import com.google.devtools.build.lib.packages.PackageValidator.InvalidPackagePieceException;
import com.google.devtools.build.lib.server.FailureDetails.PackageLoading.Code;
import com.google.devtools.build.lib.skyframe.MacroInstanceFunction.NoSuchMacroInstanceException;
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

    // TODO(https://github.com/bazelbuild/bazel/issues/23852): support finalizers. Requires
    // recursively expanding all non-finalizer macros in current package and creating a map of all
    // non-finalizer-defined rules for native.existing_rules().
    if (macroInstance.getMacroClass().isFinalizer()) {
      throw new EvalMacroFunctionException(
          new InvalidPackagePieceException(
              key, "finalizers not yet supported under lazy macro expansion"));
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
            cpuBoundSemaphore.get());
    try {
      MacroClass.executeMacroImplementation(
          macroInstance, packagePieceBuilder, packageDeclarationsValue.starlarkSemantics());
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
      throw new EvalMacroFunctionException(e);
    }
    PackagePiece.ForMacro packagePiece = packagePieceBuilder.finishBuild();

    try {
      packageFactory.afterDoneLoadingPackagePiece(
          packagePiece,
          packageDeclarationsValue.starlarkSemantics(),
          loadTimeNanos,
          env.getListener());
    } catch (InvalidPackagePieceException e) {
      throw new EvalMacroFunctionException(e);
    }

    return new PackagePieceValue.ForMacro(packagePiece);
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
