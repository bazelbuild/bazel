// Copyright 2021 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.actionsketch;

import static com.google.common.base.Preconditions.checkState;
import static java.lang.Math.max;

import com.google.auto.value.AutoValue;
import com.google.devtools.build.lib.util.BigIntegerFingerprintUtils;
import java.math.BigInteger;
import javax.annotation.Nullable;

/** Container holding a version and a {@link BigInteger} hash that has the version and more. */
@AutoValue
public abstract class HashAndVersion {
  public static final HashAndVersion ZERO = create(BigInteger.ZERO, 0L);

  public abstract BigInteger hash();

  /**
   * -1 indicates that accurate version information could not be obtained (because all the files in
   * a symlink chain were not examined, for instance). The hash may only be valid at the currently
   * evaluating version.
   */
  public abstract long version();

  @Nullable
  public static HashAndVersion create(BigInteger hash, long version) {
    if (hash == null) {
      checkState(version == Long.MAX_VALUE, "no hash with valid version %s", version);
      return null;
    }
    return new AutoValue_HashAndVersion(hash, version);
  }

  public static HashAndVersion createNoVersion(BigInteger hash) {
    return create(hash, 0L);
  }

  @Nullable
  public static HashAndVersion composeNullable(HashAndVersion first, HashAndVersion second) {
    if (first == null || second == null) {
      return null;
    }
    return create(
        BigIntegerFingerprintUtils.compose(first.hash(), second.hash()),
        first.version() == -1 || second.version() == -1
            ? -1
            : max(first.version(), second.version()));
  }
}
