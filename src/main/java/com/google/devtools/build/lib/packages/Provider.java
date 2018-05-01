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
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkValue;
import com.google.devtools.build.lib.syntax.ClassObject;

/**
 * Declared Provider (a constructor for {@link Info}).
 *
 * <p>Declared providers can be declared either natively ({@link NativeProvider} or in Skylark
 * {@link SkylarkProvider}.
 *
 * <p>{@link Provider} serves both as "type identifier" for declared provider instances and as a
 * function that can be called to construct a provider. To the Skylark user, there are "providers"
 * and "provider instances"; the former is a Java instance of this class, and the latter is a Java
 * instance of {@link Info}.
 *
 * <p>Prefer to use {@link Key} as a serializable identifier of {@link Provider}. In particular,
 * {@link Key} should be used in all data structures exposed to Skyframe.
 */
@SkylarkModule(
  name = "Provider",
  doc =
      "A constructor for simple value objects, known as provider instances."
          + "<br>"
          + "This value has a dual purpose:"
          + "  <ul>"
          + "     <li>It is a function that can be called to construct 'struct'-like values:"
          + "<pre class=\"language-python\">DataInfo = provider()\n"
          + "d = DataInfo(x = 2, y = 3)\n"
          + "print(d.x + d.y) # prints 5</pre>"
          + "     Note: Some providers, defined internally, do not allow instance creation"
          + "     </li>"
          + "     <li>It is a <i>key</i> to access a provider instance on a"
          + "        <a href=\"Target.html\">Target</a>"
          + "<pre class=\"language-python\">DataInfo = provider()\n"
          + "def _rule_impl(ctx)\n"
          + "  ... ctx.attr.dep[DataInfo]</pre>"
          + "     </li>"
          + "  </ul>"
          + "Create a new <code>Provider</code> using the "
          + "<a href=\"globals.html#provider\">provider</a> function."
)
@Immutable
public interface Provider extends SkylarkValue {

  /**
   * Has this {@link Provider} been exported? All native providers are always exported. Skylark
   * providers are exported if they are assigned to top-level name in a Skylark module.
   */
  boolean isExported();

  /** Returns a serializable representation of this {@link Provider}. */
  Key getKey();

  /** Returns a name of this {@link Provider} that should be used in error messages. */
  String getPrintableName();

  /**
   * Returns an error message format string for instances to use for their {@link
   * ClassObject#getErrorMessageForUnknownField(String)}.
   *
   * <p>The format string must contain one {@code '%s'} placeholder for the field name.
   */
  default String getErrorMessageFormatForUnknownField() {
    return String.format("'%s' object has no attribute '%%s'", getPrintableName());
  }

  /**
   * Returns the location at which provider was defined.
   */
  Location getLocation();

  /** A serializable representation of {@link Provider}. */
  public abstract static class Key {}
}
