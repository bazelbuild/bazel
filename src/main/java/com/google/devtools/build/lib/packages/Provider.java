// Copyright 2017 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.starlarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.util.Fingerprint;
import net.starlark.java.syntax.Location;

/**
 * Declared Provider (a constructor for {@link Info}).
 *
 * <p>Declared providers can be declared either natively ({@link BuiltinProvider} or in Starlark
 * {@link StarlarkProvider}.
 *
 * <p>{@link Provider} serves both as "type identifier" for declared provider instances and as a
 * function that can be called to construct a provider. To the Starlark user, there are "providers"
 * and "provider instances"; the former is a Java instance of this class, and the latter is a Java
 * instance of {@link Info}.
 *
 * <p>Prefer to use {@link Key} as a serializable identifier of {@link Provider}. In particular,
 * {@link Key} should be used in all data structures exposed to Skyframe.
 */
@Immutable
public interface Provider extends ProviderApi {

  /**
   * Has this {@link Provider} been exported? All built-in providers are always exported. Starlark
   * providers are exported if they are assigned to top-level name in a Starlark module.
   */
  boolean isExported();

  /** Returns a serializable representation of this {@link Provider}. */
  Key getKey();

  /** Returns a name of this {@link Provider} that should be used in error messages. */
  String getPrintableName();

  /**
   * Returns an error message for instances to use for their {@link
   * net.starlark.java.eval.Structure#getErrorMessageForUnknownField(String)}.
   */
  default String getErrorMessageForUnknownField(String name) {
    return String.format("'%s' value has no field or method '%s'", getPrintableName(), name);
  }

  /**
   * Returns the location at which provider was defined.
   */
  Location getLocation();

  /** A serializable and fingerprintable representation of {@link Provider}. */
  abstract class Key {
    abstract void fingerprint(Fingerprint fp);
  }
}
