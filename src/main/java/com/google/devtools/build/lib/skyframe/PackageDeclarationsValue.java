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

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import net.starlark.java.eval.StarlarkSemantics;

/**
 * A Skyframe value representing {@link Package.Declarations} and associated data (e.g. package
 * metadata, Starlark builtins, etc.) shared by all package pieces of a package.
 *
 * <p>The corresponding {@link SkyKey} is {@link PackageDeclarationsValue.Key}.
 *
 * <p>The purpose of this value is to allow change-pruning on the transitive dependency from a
 * {@link PackagePieceValue.ForMacro} to its {@link PackagePieceValue.ForBuildFile} - since package
 * pieces are not comparable.
 */
@AutoCodec
public record PackageDeclarationsValue(
    Package.Metadata metadata,
    Package.Declarations declarations,
    StarlarkSemantics starlarkSemantics,
    RepositoryMapping mainRepositoryMapping)
    implements SkyValue {

  public PackageDeclarationsValue {
    checkNotNull(metadata);
    checkNotNull(declarations);
    checkNotNull(starlarkSemantics);
    checkNotNull(mainRepositoryMapping);
  }

  /** The {@link SkyKey} for a {@link PackageDeclarationsValue}. */
  @AutoCodec
  public record Key(PackageIdentifier packageId) implements SkyKey {
    public Key {
      checkNotNull(packageId);
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.PACKAGE_DECLARATIONS;
    }
  }
}
