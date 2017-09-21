/*
 * ProGuard -- shrinking, optimization, obfuscation, and preverification
 *             of Java bytecode.
 */
package proguard.annotation;

import java.lang.annotation.*;

/**
 * This annotation specifies not to optimize or obfuscate the annotated class or
 * class member as an entry point.
 *
 * @author Eric Lafortune
 */
@Target({ ElementType.TYPE, ElementType.FIELD, ElementType.METHOD, ElementType.CONSTRUCTOR })
@Retention(RetentionPolicy.CLASS)
@Documented
public @interface KeepName {}
