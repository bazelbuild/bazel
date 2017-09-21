/*
 * ProGuard -- shrinking, optimization, obfuscation, and preverification
 *             of Java bytecode.
 */
package proguard.annotation;

import java.lang.annotation.*;

/**
 * This annotation specifies to keep the annotated class as an application,
 * together with its a main method.
 *
 * @author Eric Lafortune
 */
@Target({ ElementType.TYPE })
@Retention(RetentionPolicy.CLASS)
@Documented
public @interface KeepApplication {}
