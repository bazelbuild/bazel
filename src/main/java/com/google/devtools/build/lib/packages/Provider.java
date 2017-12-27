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
import com.google.devtools.build.lib.syntax.BaseFunction;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.FunctionSignature;
import com.google.devtools.build.lib.syntax.SkylarkType;
import javax.annotation.Nullable;

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
public abstract class Provider extends BaseFunction {

  /**
   * Constructs a provider.
   *
   * @param name provider name; should be null iff the subclass overrides {@link #getName}
   * @param signature the signature for calling this provider as a Skylark function (to construct an
   *     instance of the provider)
   * @param location the location of this provider's Skylark definition. Use {@link
   *     Location#BUILTIN} if it is a native provider.
   */
  protected Provider(
      @Nullable String name,
      FunctionSignature.WithValues<Object, SkylarkType> signature,
      Location location) {
    super(name, signature, location);
  }

  /**
   * Has this {@link Provider} been exported? All native providers are always exported. Skylark
   * providers are exported if they are assigned to top-level name in a Skylark module.
   */
  public abstract boolean isExported();

  /** Returns a serializable representation of this {@link Provider}. */
  public abstract Key getKey();

  /** Returns a name of this {@link Provider} that should be used in error messages. */
  public abstract String getPrintableName();

  /**
   * Returns an error message format for instances.
   *
   * <p>Must contain one {@code '%s'} placeholder for field name.
   */
  // TODO(bazel-team): Rename to getErrorMessageFormatForUnknownField().
  public abstract String getErrorMessageFormatForInstances();

  public SkylarkProviderIdentifier id() {
    return SkylarkProviderIdentifier.forKey(getKey());
  }

  @Override
  protected Object call(Object[] args, @Nullable FuncallExpression ast, @Nullable Environment env)
      throws EvalException, InterruptedException {
    Location loc = ast != null ? ast.getLocation() : Location.BUILTIN;
    return createInstanceFromSkylark(args, loc);
  }

  /**
   * Override this method to provide logic that is used to instantiate a declared provider from
   * Skylark.
   *
   * <p>This is a method that is called when a constructor {@code c} is invoked as<br>
   * {@code c(arg1 = val1, arg2 = val2, ...)}.
   *
   * @param args an array of argument values sorted as per the signature ({@see BaseFunction#call})
   */
  protected abstract Info createInstanceFromSkylark(Object[] args, Location loc)
      throws EvalException;

  /** A serializable representation of {@link Provider}. */
  public abstract static class Key {}
}
