package org.checkerframework.dataflow.util;

import java.util.ArrayList;
import java.util.List;

import javax.lang.model.element.AnnotationMirror;
import javax.lang.model.element.Element;

import org.checkerframework.dataflow.qual.Deterministic;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import org.checkerframework.dataflow.qual.Pure.Kind;

import org.checkerframework.javacutil.AnnotationProvider;
import org.checkerframework.javacutil.InternalUtils;

import com.sun.source.tree.MethodTree;

/**
 * An utility class for working with the {@link SideEffectFree}, {@link
 * Deterministic}, and {@link Pure} annotations.
 *
 * @see SideEffectFree
 * @see Deterministic
 * @see Pure
 *
 * @author Stefan Heule
 *
 */
public class PurityUtils {

    /** Does the method {@code tree} have any purity annotation? */
    public static boolean hasPurityAnnotation(AnnotationProvider provider,
            MethodTree tree) {
        return !getPurityKinds(provider, tree).isEmpty();
    }

    /** Does the method {@code methodElement} have any purity annotation? */
    public static boolean hasPurityAnnotation(AnnotationProvider provider,
            Element methodElement) {
        return !getPurityKinds(provider, methodElement).isEmpty();
    }

    /** Is the method {@code tree} deterministic? */
    public static boolean isDeterministic(AnnotationProvider provider,
            MethodTree tree) {
        Element methodElement = InternalUtils.symbol(tree);
        return isDeterministic(provider, methodElement);
    }

    /** Is the method {@code methodElement} deterministic? */
    public static boolean isDeterministic(AnnotationProvider provider,
            Element methodElement) {
        List<Kind> kinds = getPurityKinds(provider, methodElement);
        return kinds.contains(Kind.DETERMINISTIC);
    }

    /** Is the method {@code tree} side-effect-free? */
    public static boolean isSideEffectFree(AnnotationProvider provider,
            MethodTree tree) {
        Element methodElement = InternalUtils.symbol(tree);
        return isSideEffectFree(provider, methodElement);
    }

    /** Is the method {@code methodElement} side-effect-free? */
    public static boolean isSideEffectFree(AnnotationProvider provider,
            Element methodElement) {
        List<Kind> kinds = getPurityKinds(provider, methodElement);
        return kinds.contains(Kind.SIDE_EFFECT_FREE);
    }

    /**
     * @return the types of purity of the method {@code tree}.
     */
    public static List<Pure.Kind> getPurityKinds(AnnotationProvider provider,
            MethodTree tree) {
        Element methodElement = InternalUtils.symbol(tree);
        return getPurityKinds(provider, methodElement);
    }

    /**
     * @return the types of purity of the method {@code methodElement}.
     * TODO: should the return type be an EnumSet?
     */
    public static List<Pure.Kind> getPurityKinds(AnnotationProvider provider,
            Element methodElement) {
        AnnotationMirror pureAnnotation = provider.getDeclAnnotation(
                methodElement, Pure.class);
        AnnotationMirror sefAnnotation = provider.getDeclAnnotation(
                methodElement, SideEffectFree.class);
        AnnotationMirror detAnnotation = provider.getDeclAnnotation(
                methodElement, Deterministic.class);

        List<Pure.Kind> kinds = new ArrayList<>();
        if (pureAnnotation != null) {
            kinds.add(Kind.DETERMINISTIC);
            kinds.add(Kind.SIDE_EFFECT_FREE);
        }
        if (sefAnnotation != null) {
            kinds.add(Kind.SIDE_EFFECT_FREE);
        }
        if (detAnnotation != null) {
            kinds.add(Kind.DETERMINISTIC);
        }
        return kinds;
    }
}
