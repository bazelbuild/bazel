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

import static com.google.common.base.Preconditions.checkState;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.packages.MacroInstance;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.PackagePiece;
import com.google.devtools.build.lib.packages.PackagePieceIdentifier;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.TargetRecorder;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.StarlarkThread;

/** A subtree of package pieces belonging to one package. */
public interface PackagePieces {
  /**
   * Returns the package pieces, with parents ordered before children, and with siblings ordered by
   * name.
   */
  ImmutableMap<PackagePieceIdentifier, PackagePiece> getPackagePieces();

  /** Returns the package piece for the BUILD file. */
  PackagePiece.ForBuildFile getPackagePieceForBuildFile();

  /** Returns the identifiers of package pieces which contain errors. */
  ImmutableList<PackagePieceIdentifier> getErrorKeys();

  @Nullable
  default PackagePiece getFirstPieceContainingErrors() {
    if (getErrorKeys().isEmpty()) {
      return null;
    }
    PackagePiece firstPieceContainingErrors = getPackagePieces().get(getErrorKeys().getFirst());
    checkState(firstPieceContainingErrors.containsErrors());
    return firstPieceContainingErrors;
  }

  /**
   * Records the targets and macros of the package pieces in this collection, verifying that there
   * are no name conflicts between package pieces.
   *
   * @throws EvalException with a reconstructed Starlark call stack if there is a name conflict.
   */
  default void recordTargetsAndMacros(TargetRecorder recorder) throws EvalException {
    try {
      for (PackagePiece packagePiece : getPackagePieces().values()) {
        recorder.addAllFromPackagePiece(packagePiece, /* skipBuildFile= */ false);
      }
    } catch (TargetRecorder.NameConflictException e) {
      throw wrapNameConflictException(e);
    }
  }

  /**
   * Records the targets and macros of the package pieces in this collection, verifying that there
   * are no name conflicts between package pieces.
   *
   * @throws EvalException with a reconstructed Starlark call stack if there is a name conflict.
   */
  default void recordTargetsAndMacros(Package.Builder pkgBuilder) throws EvalException {
    try {
      for (PackagePiece packagePiece : getPackagePieces().values()) {
        pkgBuilder.addAllFromPackagePiece(packagePiece);
      }
    } catch (TargetRecorder.NameConflictException e) {
      throw wrapNameConflictException(e);
    }
  }

  private static EvalException wrapNameConflictException(TargetRecorder.NameConflictException e) {
    return new EvalException(e)
        .withCallStack(
            e.getMacro() != null
                ? e.getMacro().reconstructParentCallStack()
                : reconstructCallStack(e.getTarget()));
  }

  private static ImmutableList<StarlarkThread.CallStackEntry> reconstructCallStack(Target target) {
    @Nullable Rule rule = target.getAssociatedRule();
    if (rule != null) {
      return rule.reconstructCallStack();
    }
    @Nullable MacroInstance declaringMacro = target.getDeclaringMacro();
    if (declaringMacro != null) {
      return declaringMacro.reconstructParentCallStack();
    }
    // Top-level non-rule target
    return ImmutableList.of(
        StarlarkThread.callStackEntry(StarlarkThread.TOP_LEVEL, target.getLocation()));
  }
}
