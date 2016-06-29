package org.checkerframework.dataflow.qual;

import java.lang.annotation.Documented;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * {@code TerminatesExecution} is a method annotation that indicates that a
 * method terminates the execution of the program. This can be used to
 * annotate methods such as {@code System.exit()}.
 * <p>

 * The annotation enables flow-sensitive type refinement to be more
 * precise.  For example, after
 * <pre>
 * if (x == null) {
 *   System.err.println("Bad value supplied");
 *   System.exit(1);
 * }
 * </pre>
 * the Nullness Checker can determine that {@code x} is non-null.
 *
 * <p>
 * The annotation is a <em>trusted</em> annotation, meaning that it is not
 * checked whether the annotated method really does terminate the program.
 *
 * @checker_framework.manual #type-refinement Automatic type refinement (flow-sensitive type qualifier inference)
 *
 * @author Stefan Heule
 *
 */
@Documented
@Retention(RetentionPolicy.RUNTIME)
@Target({ ElementType.METHOD, ElementType.CONSTRUCTOR })
public @interface TerminatesExecution {
}
