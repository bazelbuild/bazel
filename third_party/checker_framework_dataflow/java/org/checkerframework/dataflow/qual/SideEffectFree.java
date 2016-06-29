package org.checkerframework.dataflow.qual;

import java.lang.annotation.Documented;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * A method is called <em>side-effect-free</em> if it has no visible
 * side-effects, such as setting a field of an object that existed before
 * the method was called.
 * <p>
 * Only the visible side-effects are important. The method is allowed to cache
 * the answer to a computationally expensive query, for instance.  It is also
 * allowed to modify newly-created objects, and a constructor is
 * side-effect-free if it does not modify any objects that existed before
 * it was called.
 * <p>
 * This annotation is important to pluggable type-checking because if some
 * fact about an object is known before a call to such a method, then the
 * fact is still known afterwards, even if the fact is about some non-final
 * field.  When any non-{@code @SideEffectFree} method is called, then a
 * pluggable type-checker must assume that any field of any accessible
 * object might have been modified, which annuls the effect of
 * flow-sensitive type refinement and prevents the pluggable type-checker
 * from making conclusions that are obvious to a programmer.
 * <p>
 * Also see {@link Pure}, which means both side-effect-free and {@link
 * Deterministic}.
 * <p>
 * <b>Analysis:</b>
 * The Checker Framework performs a conservative analysis to verify a
 * {@code @SideEffectFree} annotation.
 * The Checker Framework issues a warning
 * if the method uses any of the following Java constructs:
 * <ol>
 * <li>Assignment to any expression, except for local variables and method
 * parameters.
 * <li>A method invocation of a method that is not {@code @SideEffectFree}.
 * <li>Construction of a new object where the constructor is not {@code @SideEffectFree}.
 * </ol>
 * These rules are conservative:  any code that passes the checks is
 * side-effect-free, but the Checker Framework may issue false positive
 * warnings, for code that uses one of the forbidden constructs but is
 * side-effect-free nonetheless.  In particular, a method that caches its
 * result will be rejected.
 * <p>
 *
 * In fact, the rules are so conservative that checking is currently
 * disabled by default, but can be enabled via the
 * {@code -AcheckPurityAnnotations} command-line option.
 * <p>
 *
 * @checker_framework.manual #type-refinement-purity Side effects, determinism, purity, and flow-sensitive analysis
 *
 * @author Stefan Heule
 */
@Documented
@Retention(RetentionPolicy.RUNTIME)
@Target({ ElementType.METHOD, ElementType.CONSTRUCTOR })
public @interface SideEffectFree {
}
