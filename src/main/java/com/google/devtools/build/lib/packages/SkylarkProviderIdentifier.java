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
import java.util.Objects;
import javax.annotation.Nullable;

/**
 * A wrapper around Skylark provider identifier, representing either a declared provider ({@see
 * SkylarkProvider}) or a "legacy" string identifier.
 */
public final class SkylarkProviderIdentifier {

  @Nullable
  private final String legacyId;
  @Nullable private final Provider.Key key;

  /** Creates an id for a declared provider with a given key ({@see SkylarkProvider}). */
  public static SkylarkProviderIdentifier forKey(Provider.Key key) {
    return new SkylarkProviderIdentifier(key);
  }

  /**
   * Creates an id for a provider with a given name.
   */
  public static SkylarkProviderIdentifier forLegacy(String legacyId) {
    return new SkylarkProviderIdentifier(legacyId);
  }

  private SkylarkProviderIdentifier(String legacyId) {
    this.legacyId = legacyId;
    this.key = null;
  }

  private SkylarkProviderIdentifier(Provider.Key key) {
    this.legacyId = null;
    this.key = key;
  }

  /**
   * Returns true if this {@link SkylarkProviderIdentifier} identifies
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
    if (!(obj instanceof SkylarkProviderIdentifier)) {
      return false;
    }
    SkylarkProviderIdentifier other = (SkylarkProviderIdentifier) obj;
    return Objects.equals(legacyId, other.legacyId)
        && Objects.equals(key, other.key);
  }
}
