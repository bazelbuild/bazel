package org.checkerframework.dataflow.qual;

import java.lang.annotation.Documented;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Behaves identically to @SideEffectFree when running the Lock Checker.
 * Ignored by all other checkers.
 *
 * @see SideEffectFree
 */
@Documented
@Retention(RetentionPolicy.RUNTIME)
@Target({ ElementType.METHOD, ElementType.CONSTRUCTOR })
public @interface LockingFree {
}
