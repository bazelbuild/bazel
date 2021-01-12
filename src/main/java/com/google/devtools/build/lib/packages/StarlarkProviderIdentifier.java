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

import com.google.common.base.Preconditions;
import com.google.common.collect.Interner;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.util.Fingerprint;
import java.util.Objects;
import javax.annotation.Nullable;

/**
 * A wrapper around Starlark provider identifier, representing either a declared provider ({@see
 * StarlarkProvider}) or a "legacy" string identifier.
 */
public final class StarlarkProviderIdentifier {
  private static final Interner<StarlarkProviderIdentifier> interner =
      BlazeInterners.newWeakInterner();

  @Nullable
  private final String legacyId;
  @Nullable private final Provider.Key key;

  /** Creates an id for a declared provider with a given key ({@see StarlarkProvider}). */
  public static StarlarkProviderIdentifier forKey(Provider.Key key) {
    return interner.intern(new StarlarkProviderIdentifier(key));
  }

  /** Creates an id for a provider with a given name. */
  public static StarlarkProviderIdentifier forLegacy(String legacyId) {
    return interner.intern(new StarlarkProviderIdentifier(legacyId));
  }

  private StarlarkProviderIdentifier(String legacyId) {
    this.legacyId = legacyId;
    this.key = null;
  }

  private StarlarkProviderIdentifier(Provider.Key key) {
    this.legacyId = null;
    this.key = key;
  }

  /**
   * Returns true if this {@link StarlarkProviderIdentifier} identifies
   * a legacy provider (with a string name).
   */
  public boolean isLegacy() {
    return legacyId != null;
  }

  /**
   * Returns a string identifying the provider (only for legacy providers).
   */
  public String getLegacyId() {
    Preconditions.checkState(isLegacy(), "Check isLegacy() first");
    return legacyId;
  }

  /** Returns a key identifying the declared provider (only for non-legacy providers). */
  public Provider.Key getKey() {
    Preconditions.checkState(!isLegacy(), "Check !isLegacy() first");
    return key;
  }

  void fingerprint(Fingerprint fp) {
    if (isLegacy()) {
      fp.addBoolean(true);
      fp.addString(legacyId);
    } else {
      fp.addBoolean(false);
      key.fingerprint(fp);
    }
  }

  @Override
  public String toString() {
    if (isLegacy()) {
      return legacyId;
    }
    return key.toString();
  }

  @Override
  public int hashCode() {
    return legacyId != null ? legacyId.hashCode() * 2 : key.hashCode();
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof StarlarkProviderIdentifier)) {
      return false;
    }
    StarlarkProviderIdentifier other = (StarlarkProviderIdentifier) obj;
    return Objects.equals(legacyId, other.legacyId)
        && Objects.equals(key, other.key);
  }
}
