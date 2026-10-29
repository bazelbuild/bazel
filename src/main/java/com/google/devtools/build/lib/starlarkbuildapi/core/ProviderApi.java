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

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.docgen.annot.DocCategory;
import javax.annotation.Nullable;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.eval.Starlark;
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
   * The {@link StarlarkType} of a Starlark-callable {@link ProviderApi} value. (In other words, the
   * type of a provider symbol; as contrasted with the type of Info objects which that provider
   * symbol constructs.)
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

  /**
   * The type of a builtin provider symbol (as contrasted with the type of the Info values that the
   * provider symbol constructs).
   *
   * <p>An instance of this class should correspond to a subclass of {@code
   * com.google.devtools.build.lib.packages.BuiltinProvider}.
   *
   * <p>Note that some subclasses of {@code BuiltinProvider} shouldn't be typed as {@link
   * BuiltinProviderType}. Some implementations, such as {@link
   * com.google.devtools.build.lib.analysis.test.AnalysisFailureInfo.AnalysisFailureInfoProvider},
   * aren't callable from Starlark; therefore, they should instead be typed as {@link Types.TYPE}
   * (or a custom subtype of it, if they have Starlark-accessible methods).
   */
  public static final class BuiltinProviderType extends ProviderType {
    private final Class<?> providerClass;
    // The type of the values (usually Infos) that the provider symbol constructs.
    private final StarlarkType valueType;

    public BuiltinProviderType(Class<?> providerClass, Class<?> valueClass) {
      this.providerClass = providerClass;
      this.valueType = Starlark.getStarlarkTypeFromJavaType(valueClass, /* objectIsAny= */ false);
    }

    @Override
    public String typeRepr() {
      // This representation is not unique for the few providers that use an unmarked struct as
      // their value type, but that's unlikely to matter in practice because the user cares about
      // the value type, not the provider symbol's type.
      return String.format("<Provider[%s]>", valueType.typeRepr());
    }

    @Override
    public Types.CallableType asCallableType(TypeContext context) {
      StarlarkType selfCallMethodType =
          checkNotNull(
              context.getSelfCallMethodType(providerClass),
              "Builtin provider class %s is expected to have a self-call method",
              providerClass.getName());
      return checkNotNull(
          Types.toCallableType(selfCallMethodType, context),
          "Could not convert builtin provider class %s self-call method type %s to CallableType",
          providerClass,
          selfCallMethodType);
    }

    @Override
    public StarlarkType getInfoType(TypeContext context) {
      return valueType;
    }

    @Override
    @Nullable
    public StarlarkType getField(String name, TypeContext context) {
      return context.getStarlarkBuiltinFieldType(providerClass, name);
    }

    @Override
    public boolean equals(Object obj) {
      return obj instanceof BuiltinProviderType builtinProviderType
          && providerClass.equals(builtinProviderType.providerClass);
    }

    @Override
    public int hashCode() {
      return providerClass.hashCode();
    }
  }
}
