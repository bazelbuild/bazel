package org.checkerframework.javacutil;

import java.lang.annotation.Annotation;

import javax.lang.model.element.AnnotationMirror;
import javax.lang.model.element.Element;

import com.sun.source.tree.Tree;

/**
 * An implementation of AnnotationProvider returns annotations on
 * Java AST elements.
 */
public interface AnnotationProvider {

    /**
     * Returns the actual annotation mirror used to annotate this type,
     * whose name equals the passed annotationName if one exists, null otherwise.
     *
     * @param anno annotation class
     * @return the annotation mirror for anno
     */
    public AnnotationMirror getDeclAnnotation(Element elt,
            Class<? extends Annotation> anno);

    /**
     * Return the annotation on {@code tree} that has the class
     * {@code target}. If no annotation for the given target class exists,
     * the result is {@code null}
     *
     * @param tree
     *            The tree of which the annotation is returned
     * @param target
     *            The class of the annotation
     */
    public AnnotationMirror getAnnotationMirror(Tree tree,
            Class<? extends Annotation> target);
}
