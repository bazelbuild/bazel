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
package net.starlark.java.annot;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Annotates a Java method that can be called from Starlark.
 *
 * <p>A method annotated with {@code @StarlarkMethod} may not have overloads or hide any static or
 * default methods. Overriding is allowed, but the {@code @StarlarkMethod} annotation itself must
 * not be repeated on the override. This ensures that given a method, we can always determine its
 * corresponding {@code @StarlarkMethod} annotation, if it has one, by scanning all methods of the
 * same name in its class hierarchy, without worrying about complications like overloading or
 * generics. The lookup functionality is implemented by {@link
 * StarlarkAnnotations#getStarlarkMethod}.
 *
 * <p>Methods having this annotation must satisfy the following requirements, which are enforced at
 * compile time by {@link StarlarkMethodProcessor}:
 *
 * <ul>
 *   <li>The method must be public and non-static, and its class must implement StarlarkValue.
 *   <li>The method must declare the following parameters, in order:
 *       <ol>
 *         <li>one for each {@code Param} marked {@link Param#positional}. These parameters may be
 *             specified positionally. Among these, required parameters must precede optional ones.
 *             A suffix of the optional positional parameters may additionally be marked {@link
 *             Param#named}, meaning they may be specified by position or by name.
 *         <li>one for each {@code Param} marked {@link Param#named} but not {@link
 *             Param#positional}. These parameters must be specified by name. Again, required
 *             named-only parameters must precede optional ones.
 *         <li>one for the {@code Tuple<Object>} of extra positional arguments ({@code *args}), if
 *             {@code extraPositionals};
 *         <li>a {@code Dict<String, Object>} of extra keyword arguments ({@code **kwargs}), if
 *             {@code extraKeywords};
 *         <li>a {@code StarlarkThread}, if {@code useStarlarkThread};
 *         <li>a {@code StarlarkSemantics}, if {@code useStarlarkSemantics}.
 *       </ol>
 *       The last three parameters are implicitly supplied by the interpreter when the method is
 *       called from Starlark.
 *   <li>If {@code structField}, there must be no {@code @Param} annotations or parameters, and the
 *       only permitted special parameter is {@code StarlarkSemantics}. Rationale: unlike a method,
 *       which is actively called within in the context of a Starlark thread (which encapsulates a
 *       call stack of locations), a field is a passive thing, part of a data structure, that may be
 *       accessed by a Java caller without a Starlark thread.
 *   <li>Each {@code Param} annotation, if explicitly typed, may use either {@code type} or {@code
 *       allowedTypes}, but not both.
 *   <li>Each {@code Param} annotation must be positional or named, or both.
 *   <li>Noneable parameter variables must be declared with type Object, as the actual value may be
 *       either {@code None} or some other value, which do not share a superclass other than Object
 *       (or StarlarkValue, which is typically no more descriptive than Object).
 *   <li>Parameter variables whose class is generic must be declared using wildcard types. For
 *       example, {@code Sequence<?>} is allowed but {@code Sequence<String>} is forbidden. This is
 *       because the call-time dynamic checks verify the class but cannot verify the type
 *       parameters. Such parameters may require additional validation within the method
 *       implementation.
 *   <li>The class of the declared result type, if final, must be accepted by {@link
 *       Starlark#fromJava}. Rationale: this check helps reject clearly invalid parameter types.
 *   <li>The {@code doc} string must be non-empty, or {@code documented} must be false. Rationale:
 *       Leaving a function undocumented requires an explicit decision.
 *   <li>Each class may have up to one method annotated with {@code selfCall}, which must not be
 *       marked {@code structField=true}.
 * </ul>
 *
 * <p>When an annotated method is called from Starlark, it is a dynamic error if it returns null,
 * unless the method is marked as {@link #allowReturnNones}, in which case {@link Starlark#fromJava}
 * converts the Java null value to {@link Starlark#NONE}. This feature prevents a method whose
 * declared (and documented) result type is T from unexpectedly returning a value of type NoneType.
 */
// TODO(adonovan): rename to StarlarkAttribute and factor Starlark{Method,Field} as subinterfaces.
@Target({ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
public @interface StarlarkMethod {

  /** Name of the method, as exposed to Starlark. */
  String name();

  /**
   * The documentation text in Starlark. It can contain HTML tags for special formatting.
   *
   * <p>It is allowed to be empty only if {@link #documented()} is false.
   */
  String doc() default "";

  /**
   * If true, the function will appear in the Starlark documentation. Set this to false if the
   * function is experimental or an overloading and doesn't need to be documented.
   */
  boolean documented() default true;

  /**
   * If true, this method will be considered as a field of the enclosing Java object. E.g., if set
   * to true on a method {@code foo}, then the callsites of this method will look like {@code
   * bar.foo} instead of {@code bar.foo()}. The annotated method must be parameterless and {@link
   * #parameters()} should be empty.
   */
  boolean structField() default false;

  /**
   * List of parameters this function accepts.
   */
  Param[] parameters() default {};

  /**
   * Defines a catch-all list for additional unspecified positional parameters.
   *
   * <p>If this is left as default, it is an error for the caller to pass more positional arguments
   * than are explicitly allowed by the method signature. If this is defined, all additional
   * positional arguments are passed as elements of a {@link Tuple<Object>} to the method.
   *
   * <p>See Python's <code>*args</code> (http://thepythonguru.com/python-args-and-kwargs/).
   *
   * <p>If defined, the annotated method must declare a corresponding parameter to which a {@code
   * Tuple<Object>} may be assigned. See the interface-level javadoc for details.
   */
  // TODO(adonovan): consider using a simpler type than Param here. All that's needed at run-time
  // is a boolean. The doc tools want a name and doc string, but the rest is irrelevant and
  // distracting.
  // Ditto extraKeywords.
  Param extraPositionals() default @Param(name = "");

  /**
   * Defines a catch-all dictionary for additional unspecified named parameters.
   *
   * <p>If this is left as default, it is an error for the caller to pass any named arguments not
   * explicitly declared by the method signature. If this is defined, all additional named arguments
   * are passed as elements of a {@link Dict<String, Object>} to the method.
   *
   * <p>See Python's <code>**kwargs</code> (http://thepythonguru.com/python-args-and-kwargs/).
   *
   * <p>If defined, the annotated method must declare a corresponding parameter to which a {@code
   * Dict<String, Object>} may be assigned. See the interface-level javadoc for details.
   */
  Param extraKeywords() default @Param(name = "");

  /**
   * If true, indicates that the class containing the annotated method has the ability to be called
   * from Starlark (as if it were a function) and that the annotated method should be invoked when
   * this occurs.
   *
   * <p>A class may only have one method with selfCall set to true.
   *
   * <p>A method with selfCall=true must not be a structField, and must have name specified (used
   * for descriptive errors if, for example, there are missing arguments).
   */
  boolean selfCall() default false;

  /**
   * Permits the Java method to return null, which {@link Starlark#fromJava} then converts to {@link
   * Starlark#NONE}. If false, a null result causes the Starlark call to fail.
   */
  boolean allowReturnNones() default false;

  /**
   * If true, the StarlarkThread will be passed as an argument of the annotated function. (Thus, the
   * annotated method signature must contain StarlarkThread as a parameter. See the interface-level
   * javadoc for details.)
   *
   * <p>This is incompatible with structField=true. If structField is true, this must be false.
   */
  boolean useStarlarkThread() default false;

  /**
   * If true, the Starlark semantics will be passed to the annotated Java method. (Thus, the
   * annotated method signature must contain StarlarkSemantics as a parameter. See the
   * interface-level javadoc for details.)
   *
   * <p>This option is allowed only for fields ({@code structField=true}). For methods, the {@code
   * StarlarkThread} parameter provides access to the semantics, and more.
   */
  boolean useStarlarkSemantics() default false;

  /**
   * If non-empty, the annotated method will only be callable if the given semantic flag is true.
   * Note that at most one of {@link #enableOnlyWithFlag} and {@link #disableWithFlag} can be
   * non-empty.
   */
  String enableOnlyWithFlag() default "";

  /**
   * If non-empty, the annotated method will only be callable if the given semantic flag is false.
   * Note that at most one of {@link #enableOnlyWithFlag} and {@link #disableWithFlag} can be
   * non-empty.
   */
  String disableWithFlag() default "";
}
