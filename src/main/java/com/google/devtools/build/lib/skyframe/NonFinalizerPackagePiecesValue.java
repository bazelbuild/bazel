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
import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSortedMap;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.packages.MacroInstance;
import com.google.devtools.build.lib.packages.PackagePiece;
import com.google.devtools.build.lib.packages.PackagePieceIdentifier;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;

/**
 * A Skyframe value representing all package pieces of a package that are not defined by finalizer
 * macros.
 *
 * <p>The corresponding {@link com.google.devtools.build.skyframe.SkyKey} is {@link
 * NonFinalizerPackagePiecesValue.Key}.
 */
@AutoCodec
public record NonFinalizerPackagePiecesValue(
    /**
     * The package pieces, ordered by BFS traversal, with siblings ordered by name. The first
     * element is always the package's {@link PackagePiece.ForBuildFile}. Any package pieces with
     * errors are not further expanded.
     */
    ImmutableMap<PackagePieceIdentifier, PackagePiece> packagePieces,
    /** Identifiers of package pieces which contain errors. */
    ImmutableList<PackagePieceIdentifier> errorKeys,
    /**
     * An {@link EvalException} indicating a name conflict between targets or macros in different
     * package pieces.
     */
    @Nullable EvalException nameConflictBetweenPackagePiecesException,
    /**
     * The targets in the package pieces, ordered by name. May be incomplete if either
     * nameConflictBetweenPackagePiecesException is non-null or errorKeys is non-empty.
     */
    ImmutableSortedMap<String, Target> targets,
    /**
     * The macros in the package pieces, keyed (and ordered) by id. May be incomplete if either
     * nameConflictBetweenPackagePiecesException is non-null or errorKeys is non-empty.
     */
    ImmutableSortedMap<String, MacroInstance> macroInstances)
    implements PackagePieces, SkyValue {
  public NonFinalizerPackagePiecesValue {
    checkNotNull(packagePieces);
    checkArgument(!packagePieces.isEmpty());
    checkArgument(packagePieces.values().iterator().next() instanceof PackagePiece.ForBuildFile);
    checkNotNull(errorKeys);
    checkNotNull(targets);
    checkNotNull(macroInstances);
  }

  @Override
  public ImmutableMap<PackagePieceIdentifier, PackagePiece> getPackagePieces() {
    return packagePieces;
  }

  @Override
  public PackagePiece.ForBuildFile getPackagePieceForBuildFile() {
    return (PackagePiece.ForBuildFile) packagePieces.values().iterator().next();
  }

  @Override
  public ImmutableList<PackagePieceIdentifier> getErrorKeys() {
    return errorKeys;
  }

  public boolean containsErrors() {
    return !errorKeys.isEmpty() || nameConflictBetweenPackagePiecesException() != null;
  }

  /** A SkyKey for a {@link NonFinalizerPackagePiecesValue}. */
  @AutoCodec
  public static record Key(PackageIdentifier pkgId) implements SkyKey {
    public Key {
      checkNotNull(pkgId);
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.NON_FINALIZER_PACKAGE_PIECES;
    }
  }
}
