package org.checkerframework.dataflow.qual;

import java.lang.annotation.Documented;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * A method is called <em>deterministic</em> if it returns the same value
 * (according to {@code ==}) every time it is called with the same
 * parameters and in the same environment. The parameters include the
 * receiver, and the environment includes all of the Java heap (that is,
 * all fields of all objects and all static variables).
 * <p>
 * This annotation is important to pluggable type-checking because, after a
 * call to a {@code @Deterministic} method, flow-sensitive type refinement
 * can assume that anything learned about the first invocation is true
 * about subsequent invocations (so long as no non-{@code @}{@link
 * SideEffectFree} method call intervenes).  For example,
 * the following code never suffers a null pointer
 * exception, so the Nullness Checker need not issue a warning:
 * <pre>{@code       if (x.myDeterministicMethod() != null) {
        x.myDeterministicMethod().hashCode();
      }}</pre>
 * <p>
 * Note that {@code @Deterministic} guarantees that the result is
 * identical according to {@code ==}, <b>not</b> equal according to
 * {@code equals}.  This means that writing {@code @Deterministic} on a
 * method that returns a reference is often erroneous unless the
 * returned value is cached or interned.
 * <p>
 * Also see {@link Pure}, which means both deterministic and {@link
 * SideEffectFree}.
 * <p>
 * <b>Analysis:</b>
 * The Checker Framework performs a conservative analysis to verify a
 * {@code @Deterministic} annotation.  The Checker Framework issues a
 * warning if the method uses any of the following Java constructs:
 * <ol>
 * <li>Assignment to any expression, except for local variables (and method
 * parameters).
 * <li>A method invocation of a method that is not {@link Deterministic}.
 * <li>Construction of a new object.
 * <li>Catching any exceptions.  This is to prevent a method to get a hold of
 * newly created objects and using these objects (or some property thereof)
 * to change their return value.  For instance, the following method must be
 * forbidden.
 * <pre>{@code
      &#64;Deterministic
      int f() {
         try {
            int b = 0;
            int a = 1/b;
         } catch (Throwable t) {
            return t.hashCode();
         }
         return 0;
      }
    }</pre>
 * </ol>
 * A constructor can be {@code @Pure}, but a constructor <em>invocation</em> is
 * not deterministic since it returns a different new object each time.
 * TODO: Side-effect-free constructors could be allowed to set their own fields.
 * <p>
 *
 * Note that the rules for checking currently imply that every {@code
 * Deterministic} method is also {@link SideEffectFree}. This might change
 * in the future; in general, a deterministic method does not need to be
 * side-effect-free.
 * <p>
 *
 * These rules are conservative:  any code that passes the checks is
 * deterministic, but the Checker Framework may issue false positive
 * warnings, for code that uses one of the forbidden constructs but is
 * deterministic nonetheless.
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
public @interface Deterministic {
}
