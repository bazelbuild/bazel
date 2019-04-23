/*
 * ProGuard -- shrinking, optimization, obfuscation, and preverification
 *             of Java bytecode.
 */
package proguard.annotation;

import java.lang.annotation.*;

/**
 * This annotation specifies to keep all public implementations or extensions
 * of the annotated class as entry points.
 *
 * @author Eric Lafortune
 */
@Target({ ElementType.TYPE })
@Retention(RetentionPolicy.CLASS)
@Documented
public @interface KeepPublicImplementations {}
