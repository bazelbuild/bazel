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
package com.google.devtools.build.lib.starlarkbuildapi.core;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.docgen.annot.DocCategory;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.eval.StarlarkTypeValue;
import net.starlark.java.syntax.StarlarkType;
import net.starlark.java.syntax.TypeConstructor;
import net.starlark.java.syntax.TypeContext;
import net.starlark.java.syntax.Types;

/** Interface for provider objects (constructors for {@link StructApi} objects). */
@StarlarkBuiltin(
    name = "Provider",
    category = DocCategory.BUILTIN,
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
            + "        <a href=\"../builtins/Target.html\">Target</a>"
            + "<pre class=\"language-python\">DataInfo = provider()\n"
            + "def _rule_impl(ctx)\n"
            + "  ... ctx.attr.dep[DataInfo]</pre>"
            + "     </li>"
            + "  </ul>"
            + "Create a new <code>Provider</code> using the "
            + "<a href=\"../globals/bzl.html#provider\">provider</a> function.")
public interface ProviderApi extends StarlarkTypeValue {
  public static TypeConstructor getAssociatedTypeConstructor() {
    return ANY_PROVIDER_TYPE_CONSTRUCTOR;
  }

  /**
   * The {@link StarlarkType} of a {@link ProviderApi} value. (In other words, the type of a
   * provider symbol; as contrasted with the type of Info objects which that provider symbol
   * constructs.)
   */
  public abstract static class ProviderType extends StarlarkType {
    /** The callable supertype of the provider symbol. */
    public abstract Types.CallableType asCallableType(TypeContext context);

    /** The type of Info objects constructed by this provider symbol. */
    public abstract StarlarkType getInfoType(TypeContext context);

    @Override
    public ImmutableList<StarlarkType> getSupertypes(TypeContext context) {
      return ImmutableList.of(asCallableType(context), Types.TYPE);
    }

    @Override
    public boolean assignableFromHook(StarlarkType t, TypeContext context) {
      if (t instanceof ProviderType that) {
        return this.equals(ANY_PROVIDER_TYPE)
            || that.equals(ANY_PROVIDER_TYPE)
            || this.equals(that);
      }
      return false;
    }
  }

  /** A provider type allowing arbitrary calls, and assignable to and from any provider type. */
  public static final AnyProviderType ANY_PROVIDER_TYPE = new AnyProviderType();

  // Not parameterized; user code should not be able to instantiate custom ProviderType instances
  // other than as a side effect of creating a new provider symbol by calling `provider()`.
  public static final TypeConstructor ANY_PROVIDER_TYPE_CONSTRUCTOR =
      Types.wrapType("Provider", ANY_PROVIDER_TYPE);

  /** A provider type allowing arbitrary calls, and assignable to and from any provider type. */
  public static final class AnyProviderType extends ProviderType {
    // (*args, **kwargs) -> struct
    private static final Types.CallableType anyProviderCallable =
        Types.simpleCallable(ImmutableList.of(), true, Types.ANY_STRUCT);

    @Override
    public Types.CallableType asCallableType(TypeContext context) {
      return anyProviderCallable;
    }

    @Override
    public StarlarkType getInfoType(TypeContext context) {
      return Types.ANY_STRUCT;
    }

    @Override
    public String typeRepr() {
      return "Provider";
    }

    // singleton
    private AnyProviderType() {}

    @Override
    public int hashCode() {
      return AnyProviderType.class.hashCode();
    }

    @Override
    public boolean equals(Object obj) {
      return obj instanceof AnyProviderType;
    }
  }
}
