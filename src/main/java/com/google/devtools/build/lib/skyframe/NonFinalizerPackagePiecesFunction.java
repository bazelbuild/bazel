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

import com.google.common.collect.ImmutableSortedMap;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchPackagePieceException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.TargetRecorder;
import com.google.devtools.build.lib.server.FailureDetails.PackageLoading.Code;
import com.google.devtools.build.lib.skyframe.MacroInstanceFunction.NoSuchMacroInstanceException;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;

/**
 * A SkyFunction that collects the non-finalizer-defined {@link
 * com.google.devtools.build.lib.packages.PackagePiece}s of a package, producing a {@link
 * NonFinalizerPackagePiecesValue}.
 */
final class NonFinalizerPackagePiecesFunction implements SkyFunction {
  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws NonFinalizerPackagePiecesFunctionException, InterruptedException {
    NonFinalizerPackagePiecesValue.Key key = (NonFinalizerPackagePiecesValue.Key) skyKey.argument();
    EvalMacroFunction.RecursiveExpander expander = new EvalMacroFunction.RecursiveExpander();
    try {
      if (expander.expand(key.pkgId(), env, /* expandFinalizers= */ false) == null) {
        // Restart
        return null;
      }
    } catch (NoSuchPackageException e) {
      throw new NonFinalizerPackagePiecesFunctionException(e);
    } catch (NoSuchPackagePieceException e) {
      throw new NonFinalizerPackagePiecesFunctionException(e);
    } catch (NoSuchMacroInstanceException e) {
      throw new NonFinalizerPackagePiecesFunctionException(e);
    }

    if (expander.getPackagePieces().size() == 1) {
      // Trivial case - BUILD file only; name conflicts were already checked by
      // PackagePiece.ForBuildFile construction.
      return new NonFinalizerPackagePiecesValue(
          expander.getPackagePieces(),
          expander.getErrorKeys(),
          /* nameConflictBetweenPackagePiecesException= */ null,
          // All targets are top-level; no non-finalizer macros.
          expander.getPackagePieceForBuildFile().getTargets(),
          ImmutableSortedMap.of());
    }

    TargetRecorder targetRecorder =
        new TargetRecorder(
            /* enableNameConflictChecking= */ true,
            /* trackFullMacroInformation= */ false,
            /* enableTargetMapSnapshotting= */ false);
    @Nullable EvalException nameConflictException = null;
    try {
      expander.recordTargetsAndMacros(targetRecorder);
    } catch (EvalException e) {
      nameConflictException = e;
      env.getListener()
          .handle(
              Package.error(
                  e.getInnermostLocation(), e.getMessageWithStack(), Code.STARLARK_EVAL_ERROR));
    }

    return new NonFinalizerPackagePiecesValue(
        expander.getPackagePieces(),
        expander.getErrorKeys(),
        nameConflictException,
        ImmutableSortedMap.copyOf(targetRecorder.getTargetMap()),
        ImmutableSortedMap.copyOf(targetRecorder.getMacroMap()));
  }

  public static final class NonFinalizerPackagePiecesFunctionException
      extends SkyFunctionException {
    NonFinalizerPackagePiecesFunctionException(NoSuchPackageException cause) {
      super(cause, Transience.PERSISTENT);
    }

    NonFinalizerPackagePiecesFunctionException(NoSuchPackagePieceException cause) {
      super(cause, Transience.PERSISTENT);
    }

    NonFinalizerPackagePiecesFunctionException(NoSuchMacroInstanceException cause) {
      super(cause, Transience.PERSISTENT);
    }
  }
}
