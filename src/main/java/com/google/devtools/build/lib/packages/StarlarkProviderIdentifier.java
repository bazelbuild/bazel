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

import com.google.common.collect.Interner;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.util.Fingerprint;
import java.util.Objects;

/**
 * A wrapper around Starlark provider identifier, representing either a declared provider ({@see
 * StarlarkProvider}) or a "legacy" string identifier.
 */
public abstract class StarlarkProviderIdentifier {
  private static final Interner<StarlarkProviderIdentifier> interner =
      BlazeInterners.newWeakInterner();

  /** Creates an id for a declared provider with a given key ({@see StarlarkProvider}). */
  public static StarlarkProviderIdentifier forKey(Provider.Key key) {
    return interner.intern(new KeyedIdentifier(key));
  }

  /** Returns a key identifying the declared provider (only for non-legacy providers). */
  public abstract Provider.Key getKey();

  abstract void fingerprint(Fingerprint fp);

  /**
   * Returns the provider key name for a declared provider, or the legacy ID for a legacy provider.
   *
   * <p>Used for rendering human-readable descriptions, such as for a rule attribute's set of
   * required providers.
   */
  @Override
  public abstract String toString();

  @AutoCodec
  static final class KeyedIdentifier extends StarlarkProviderIdentifier {
    private final Provider.Key key;

    private KeyedIdentifier(Provider.Key key) {
      this.key = key;
    }

    @Override
    public Provider.Key getKey() {
      return key;
    }

    @Override
    void fingerprint(Fingerprint fp) {
      fp.addBoolean(false);
      key.fingerprint(fp);
    }

    @Override
    public int hashCode() {
      return key.hashCode();
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      }
      if (!(obj instanceof KeyedIdentifier)) {
        return false;
      }
      return Objects.equals(key, ((KeyedIdentifier) obj).key);
    }

    @Override
    public String toString() {
      return key.toString();
    }

    @AutoCodec.Interner
    static KeyedIdentifier intern(KeyedIdentifier id) {
      return (KeyedIdentifier) interner.intern(id);
    }
  }
}
