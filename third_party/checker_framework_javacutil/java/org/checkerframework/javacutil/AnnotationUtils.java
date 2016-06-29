package org.checkerframework.javacutil;

/*>>>
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import org.checkerframework.checker.nullness.qual.*;
import org.checkerframework.checker.interning.qual.*;
*/


import java.lang.annotation.Annotation;
import java.lang.annotation.Inherited;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

import javax.lang.model.element.AnnotationMirror;
import javax.lang.model.element.AnnotationValue;
import javax.lang.model.element.ElementKind;
import javax.lang.model.element.ExecutableElement;
import javax.lang.model.element.Name;
import javax.lang.model.element.TypeElement;
import javax.lang.model.type.DeclaredType;
import javax.lang.model.type.TypeMirror;
import javax.lang.model.util.ElementFilter;
import javax.lang.model.util.Elements;

import com.sun.source.tree.AnnotationTree;
import com.sun.source.tree.MethodTree;
import com.sun.source.tree.ModifiersTree;
import com.sun.tools.javac.code.Symbol.VarSymbol;
import com.sun.tools.javac.code.Type;
import com.sun.tools.javac.model.JavacElements;
/**
 * A utility class for working with annotations.
 */
public class AnnotationUtils {

    // Class cannot be instantiated.
    private AnnotationUtils() { throw new AssertionError("Class AnnotationUtils cannot be instantiated."); }

    // TODO: hack to clear out static state.
    // {@link org.checkerframework.qualframework.util.QualifierContext} should
    // handle instantiation of utility classes.
    public static void clear() {
        annotationsFromNames.clear();
        annotationMirrorNames.clear();
        annotationMirrorSimpleNames.clear();
        annotationClassNames.clear();
    }

    // **********************************************************************
    // Factory Methods to create instances of AnnotationMirror
    // **********************************************************************

    /** Caching for annotation creation. */
    private static final Map<CharSequence, AnnotationMirror> annotationsFromNames
        = new HashMap<CharSequence, AnnotationMirror>();


    private static final int ANNOTATION_CACHE_SIZE = 500;

    /**
     * Cache names of AnnotationMirrors for faster access.  Values in
     * the map are interned Strings, so they can be compared with ==.
     */
    private static final Map<AnnotationMirror, /*@Interned*/ String> annotationMirrorNames
        = CollectionUtils.createLRUCache(ANNOTATION_CACHE_SIZE);

    /**
     * Cache simple names of AnnotationMirrors for faster access.  Values in
     * the map are interned Strings, so they can be compared with ==.
     */
    private static final Map<AnnotationMirror, /*@Interned*/ String> annotationMirrorSimpleNames
        = CollectionUtils.createLRUCache(ANNOTATION_CACHE_SIZE);

    /**
     * Cache names of classes representing AnnotationMirrors for
     * faster access.  Values in the map are interned Strings, so they
     * can be compared with ==.
     */
    private static final Map<Class<? extends Annotation>, /*@Interned*/ String> annotationClassNames
        = new HashMap<Class<? extends Annotation>, /*@Interned*/ String>();

    /**
     * Creates an {@link AnnotationMirror} given by a particular
     * fully-qualified name.  getElementValues on the result returns an
     * empty map.
     *
     * @param elements the element utilities to use
     * @param name the name of the annotation to create
     * @return an {@link AnnotationMirror} of type {@code} name
     */
    public static AnnotationMirror fromName(Elements elements, CharSequence name) {
        if (annotationsFromNames.containsKey(name)) {
            return annotationsFromNames.get(name);
        }
        final DeclaredType annoType = typeFromName(elements, name);
        if (annoType == null) {
            return null;
        }
        if (annoType.asElement().getKind() != ElementKind.ANNOTATION_TYPE) {
            ErrorReporter.errorAbort(annoType + " is not an annotation");
            return null; // dead code
        }
        AnnotationMirror result = new AnnotationMirror() {
            String toString = "@" + annoType;

            @Override
            public DeclaredType getAnnotationType() {
                return annoType;
            }
            @Override
            public Map<? extends ExecutableElement, ? extends AnnotationValue>
                getElementValues() {
                return Collections.emptyMap();
            }
            /*@SideEffectFree*/
            @Override
            public String toString() {
                return toString;
            }
        };
        annotationsFromNames.put(name, result);
        return result;
    }

    /**
     * Creates an {@link AnnotationMirror} given by a particular annotation
     * class.
     *
     * @param elements the element utilities to use
     * @param clazz the annotation class
     * @return an {@link AnnotationMirror} of type given type
     */
    public static AnnotationMirror fromClass(Elements elements, Class<? extends Annotation> clazz) {
        return fromName(elements, clazz.getCanonicalName());
    }

    /**
     * A utility method that converts a {@link CharSequence} (usually a {@link
     * String}) into a {@link TypeMirror} named thereby.
     *
     * @param elements the element utilities to use
     * @param name the name of a type
     * @return the {@link TypeMirror} corresponding to that name
     */
    private static DeclaredType typeFromName(Elements elements, CharSequence name) {
        /*@Nullable*/ TypeElement typeElt = elements.getTypeElement(name);
        if (typeElt == null) {
            return null;
        }

        return (DeclaredType) typeElt.asType();
    }


    // **********************************************************************
    // Helper methods to handle annotations.  mainly workaround
    // AnnotationMirror.equals undesired property
    // (I think the undesired property is that it's reference equality.)
    // **********************************************************************

    /**
     * @return the fully-qualified name of an annotation as a String
     */
    public static final /*@Interned*/ String annotationName(AnnotationMirror annotation) {
        if (annotationMirrorNames.containsKey(annotation)) {
            return annotationMirrorNames.get(annotation);
        }

        final DeclaredType annoType = annotation.getAnnotationType();
        final TypeElement elm = (TypeElement) annoType.asElement();
        /*@Interned*/ String name = elm.getQualifiedName().toString().intern();
        annotationMirrorNames.put(annotation, name);
        return name;
    }

    /**
     * @return the simple name of an annotation as a String
     */
    public static String annotationSimpleName(AnnotationMirror annotation) {
        if (annotationMirrorSimpleNames.containsKey(annotation)) {
            return annotationMirrorSimpleNames.get(annotation);
        }

        final DeclaredType annoType = annotation.getAnnotationType();
        final TypeElement elm = (TypeElement) annoType.asElement();
        /*@Interned*/ String name = elm.getSimpleName().toString().intern();
        annotationMirrorSimpleNames.put(annotation, name);
        return name;
    }

    /**
     * Checks if both annotations are the same.
     *
     * Returns true iff both annotations are of the same type and have the
     * same annotation values.  This behavior differs from
     * {@code AnnotationMirror.equals(Object)}.  The equals method returns
     * true iff both annotations are the same and annotate the same annotation
     * target (e.g. field, variable, etc).
     *
     * @return true iff a1 and a2 are the same annotation
     */
    public static boolean areSame(/*@Nullable*/ AnnotationMirror a1, /*@Nullable*/ AnnotationMirror a2) {
        if (a1 != null && a2 != null) {
            if (annotationName(a1) != annotationName(a2)) {
                return false;
            }

            Map<? extends ExecutableElement, ? extends AnnotationValue> elval1 = getElementValuesWithDefaults(a1);
            Map<? extends ExecutableElement, ? extends AnnotationValue> elval2 = getElementValuesWithDefaults(a2);

            return elval1.toString().equals(elval2.toString());
        }

        // only true, iff both are null
        return a1 == a2;
    }

    /**
     * @see #areSame(AnnotationMirror, AnnotationMirror)
     * @return true iff a1 and a2 have the same annotation type
     */
    public static boolean areSameIgnoringValues(AnnotationMirror a1, AnnotationMirror a2) {
        if (a1 != null && a2 != null) {
            return annotationName(a1) == annotationName(a2);
        }
        return a1 == a2;
    }

    /**
     * Checks that the annotation {@code am} has the name {@code aname}. Values
     * are ignored.
     */
    public static boolean areSameByName(AnnotationMirror am, /*@Interned*/ String aname) {
        // Both strings are interned.
        return annotationName(am) == aname;
    }

    /**
     * Checks that the annotation {@code am} has the name of {@code anno}.
     * Values are ignored.
     */
    public static boolean areSameByClass(AnnotationMirror am,
            Class<? extends Annotation> anno) {
        /*@Interned*/ String canonicalName;
        if (annotationClassNames.containsKey(anno)) {
            canonicalName = annotationClassNames.get(anno);
        } else {
            canonicalName = anno.getCanonicalName().intern();
            annotationClassNames.put(anno, canonicalName);
        }
        return areSameByName(am, canonicalName);
    }

    /**
     * Checks that two collections contain the same annotations.
     *
     * @return true iff c1 and c2 contain the same annotations
     */
    public static boolean areSame(Collection<? extends AnnotationMirror> c1, Collection<? extends AnnotationMirror> c2) {
        if (c1.size() != c2.size()) {
            return false;
        }
        if (c1.size() == 1) {
            return areSame(c1.iterator().next(), c2.iterator().next());
        }

        Set<AnnotationMirror> s1 = createAnnotationSet();
        Set<AnnotationMirror> s2 = createAnnotationSet();
        s1.addAll(c1);
        s2.addAll(c2);

        // depend on the fact that Set is an ordered set.
        Iterator<AnnotationMirror> iter1 = s1.iterator();
        Iterator<AnnotationMirror> iter2 = s2.iterator();

        while (iter1.hasNext()) {
            AnnotationMirror anno1 = iter1.next();
            AnnotationMirror anno2 = iter2.next();
            if (!areSame(anno1, anno2)) {
                return false;
            }
        }
        return true;
    }

    /**
     * Checks that the collection contains the annotation.
     * Using Collection.contains does not always work, because it
     * does not use areSame for comparison.
     *
     * @return true iff c contains anno, according to areSame
     */
    public static boolean containsSame(Collection<? extends AnnotationMirror> c, AnnotationMirror anno) {
        for (AnnotationMirror an : c) {
            if (AnnotationUtils.areSame(an, anno)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Checks that the collection contains the annotation.
     * Using Collection.contains does not always work, because it
     * does not use areSame for comparison.
     *
     * @return true iff c contains anno, according to areSameByClass
     */
    public static boolean containsSameByClass(Collection<? extends AnnotationMirror> c, Class<? extends Annotation> anno) {
        for (AnnotationMirror an : c) {
            if (AnnotationUtils.areSameByClass(an, anno)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Checks that the collection contains the annotation ignoring values.
     * Using Collection.contains does not always work, because it
     * does not use areSameIgnoringValues for comparison.
     *
     * @return true iff c contains anno, according to areSameIgnoringValues
     */
    public static boolean containsSameIgnoringValues(Collection<? extends AnnotationMirror> c, AnnotationMirror anno) {
        for (AnnotationMirror an : c) {
            if (AnnotationUtils.areSameIgnoringValues(an, anno)) {
                return true;
            }
        }
        return false;
    }

    private static final Comparator<AnnotationMirror> ANNOTATION_ORDERING
    = new Comparator<AnnotationMirror>() {
        @Override
        public int compare(AnnotationMirror a1, AnnotationMirror a2) {
            String n1 = a1.toString();
            String n2 = a2.toString();

            return n1.compareTo(n2);
        }
    };

    /**
     * provide ordering for {@link AnnotationMirror} based on their fully
     * qualified name.  The ordering ignores annotation values when ordering.
     *
     * The ordering is meant to be used as {@link TreeSet} or {@link TreeMap}
     * ordering.  A {@link Set} should not contain two annotations that only
     * differ in values.
     */
    public static Comparator<AnnotationMirror> annotationOrdering() {
        return ANNOTATION_ORDERING;
    }

    /**
     * Create a map suitable for storing {@link AnnotationMirror} as keys.
     *
     * It can store one instance of {@link AnnotationMirror} of a given
     * declared type, regardless of the annotation element values.
     *
     * @param <V> the value of the map
     * @return a new map with {@link AnnotationMirror} as key
     */
    public static <V> Map<AnnotationMirror, V> createAnnotationMap() {
        return new TreeMap<AnnotationMirror, V>(annotationOrdering());
    }

    /**
     * Constructs a {@link Set} suitable for storing {@link AnnotationMirror}s.
     *
     * It stores at most once instance of {@link AnnotationMirror} of a given
     * type, regardless of the annotation element values.
     *
     * @return a new set to store {@link AnnotationMirror} as element
     */
    public static Set<AnnotationMirror> createAnnotationSet() {
        return new TreeSet<AnnotationMirror>(annotationOrdering());
    }

    /** Returns true if the given annotation has a @Inherited meta-annotation. */
    public static boolean hasInheritedMeta(AnnotationMirror anno) {
        return anno.getAnnotationType().asElement().getAnnotation(Inherited.class) != null;
    }


    // **********************************************************************
    // Extractors for annotation values
    // **********************************************************************

    /**
     * Returns the values of an annotation's attributes, including defaults.
     * The method with the same name in JavacElements cannot be used directly,
     * because it includes a cast to Attribute.Compound, which doesn't hold
     * for annotations generated by the Checker Framework.
     *
     * @see AnnotationMirror#getElementValues()
     * @see JavacElements#getElementValuesWithDefaults(AnnotationMirror)
     *
     * @param ad  annotation to examine
     * @return the values of the annotation's elements, including defaults
     */
    public static Map<? extends ExecutableElement, ? extends AnnotationValue>
    getElementValuesWithDefaults(AnnotationMirror ad) {
        Map<ExecutableElement, AnnotationValue> valMap
            = new HashMap<ExecutableElement, AnnotationValue>();
        if (ad.getElementValues() != null) {
            valMap.putAll(ad.getElementValues());
        }
        for (ExecutableElement meth :
            ElementFilter.methodsIn(ad.getAnnotationType().asElement().getEnclosedElements())) {
            AnnotationValue defaultValue = meth.getDefaultValue();
            if (defaultValue != null && !valMap.containsKey(meth)) {
                valMap.put(meth, defaultValue);
            }
        }
        return valMap;
    }


    /**
     * Verify whether the attribute with the name {@code name} exists in
     * the annotation {@code anno}.
     *
     * @param anno the annotation to examine
     * @param name the name of the attribute
     * @return whether the attribute exists in anno
     */
    public static <T> boolean hasElementValue(AnnotationMirror anno, CharSequence name) {
        Map<? extends ExecutableElement, ? extends AnnotationValue> valmap = anno.getElementValues();
        for (ExecutableElement elem : valmap.keySet()) {
            if (elem.getSimpleName().contentEquals(name)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Get the attribute with the name {@code name} of the annotation
     * {@code anno}. The result is expected to have type {@code expectedType}.
     *
     * <p>
     * <em>Note 1</em>: The method does not work well for attributes of an array
     * type (as it would return a list of {@link AnnotationValue}s). Use
     * {@code getElementValueArray} instead.
     *
     * <p>
     * <em>Note 2</em>: The method does not work for attributes of an enum type,
     * as the AnnotationValue is a VarSymbol and would be cast to the enum type,
     * which doesn't work. Use {@code getElementValueEnum} instead.
     *
     *
     * @param anno the annotation to disassemble
     * @param name the name of the attribute to access
     * @param expectedType the expected type used to cast the return type
     * @param useDefaults whether to apply default values to the attribute
     * @return the value of the attribute with the given name
     */
    public static <T> T getElementValue(AnnotationMirror anno, CharSequence name,
            Class<T> expectedType, boolean useDefaults) {
        Map<? extends ExecutableElement, ? extends AnnotationValue> valmap;
        if (useDefaults) {
            valmap = getElementValuesWithDefaults(anno);
        } else {
            valmap = anno.getElementValues();
        }
        for (ExecutableElement elem : valmap.keySet()) {
            if (elem.getSimpleName().contentEquals(name)) {
                AnnotationValue val = valmap.get(elem);
                return expectedType.cast(val.getValue());
            }
        }
        ErrorReporter.errorAbort("No element with name \'" + name + "\' in annotation " + anno);
        return null; // dead code
    }

    /**
     * Version that is suitable for Enum elements.
     */
    public static <T extends Enum<T>> T getElementValueEnum(
            AnnotationMirror anno, CharSequence name, Class<T> t,
            boolean useDefaults) {
        VarSymbol vs = getElementValue(anno, name, VarSymbol.class, useDefaults);
        T value = Enum.valueOf(t, vs.getSimpleName().toString());
        return value;
    }

    /**
     * Get the attribute with the name {@code name} of the annotation
     * {@code anno}, where the attribute has an array type. One element of the
     * result is expected to have type {@code expectedType}.
     *
     * Parameter useDefaults is used to determine whether default values
     * should be used for annotation values. Finding defaults requires
     * more computation, so should be false when no defaulting is needed.
     *
     * @param anno the annotation to disassemble
     * @param name the name of the attribute to access
     * @param expectedType the expected type used to cast the return type
     * @param useDefaults whether to apply default values to the attribute
     * @return the value of the attribute with the given name
     */
    public static <T> List<T> getElementValueArray(AnnotationMirror anno,
            CharSequence name, Class<T> expectedType, boolean useDefaults) {
        @SuppressWarnings("unchecked")
        List<AnnotationValue> la = getElementValue(anno, name, List.class, useDefaults);
        List<T> result = new ArrayList<T>(la.size());
        for (AnnotationValue a : la) {
            result.add(expectedType.cast(a.getValue()));
        }
        return result;
    }

    /**
     * Get the attribute with the name {@code name} of the annotation
     * {@code anno}, or the default value if no attribute is present explicitly,
     * where the attribute has an array type and the elements are {@code Enum}s.
     * One element of the result is expected to have type {@code expectedType}.
     */
    public static <T extends Enum<T>> List<T> getElementValueEnumArray(
            AnnotationMirror anno, CharSequence name, Class<T> t,
            boolean useDefaults) {
        @SuppressWarnings("unchecked")
        List<AnnotationValue> la = getElementValue(anno, name, List.class, useDefaults);
        List<T> result = new ArrayList<T>(la.size());
        for (AnnotationValue a : la) {
            T value = Enum.valueOf(t, a.getValue().toString());
            result.add(value);
        }
        return result;
    }

    /**
     * Get the Name of the class that is referenced by attribute 'name'.
     * This is a convenience method for the most common use-case.
     * Like getElementValue(anno, name, ClassType.class).getQualifiedName(), but
     * this method ensures consistent use of the qualified name.
     */
    public static Name getElementValueClassName(AnnotationMirror anno, CharSequence name,
            boolean useDefaults) {
        Type.ClassType ct = getElementValue(anno, name, Type.ClassType.class, useDefaults);
        // TODO:  Is it a problem that this returns the type parameters too?  Should I cut them off?
        return ct.asElement().getQualifiedName();
    }

    /**
     * Get the Class that is referenced by attribute 'name'.
     * This method uses Class.forName to load the class. It returns
     * null if the class wasn't found.
     */
    public static Class<?> getElementValueClass(AnnotationMirror anno, CharSequence name,
            boolean useDefaults) {
        Name cn = getElementValueClassName(anno, name, useDefaults);
        try {
            Class<?> cls =  Class.forName(cn.toString());
            return cls;
        } catch (ClassNotFoundException e) {
            ErrorReporter.errorAbort("Could not load class '" + cn + "' for field '" + name +
                    "' in annotation " + anno, e);
            return null; // dead code
        }
    }

    /**
     * See checkers.types.QualifierHierarchy#updateMappingToMutableSet(QualifierHierarchy, Map, Object, AnnotationMirror)
     * (Not linked because it is in an independent project.
     */
    public static <T> void updateMappingToImmutableSet(Map<T, Set<AnnotationMirror>> map,
            T key, Set<AnnotationMirror> newQual) {

        Set<AnnotationMirror> result = AnnotationUtils.createAnnotationSet();
        // TODO: if T is also an AnnotationMirror, should we use areSame?
        if (!map.containsKey(key)) {
            result.addAll(newQual);
        } else {
            result.addAll(map.get(key));
            result.addAll(newQual);
        }
        map.put(key, Collections.unmodifiableSet(result));
    }

    /**
     * Returns the annotations explicitly written on a constructor result.
     * Callers should check that {@code constructorDeclaration} is in fact a declaration
     * of a constructor.
     *
     * @param constructorDeclaration declaration tree of constructor
     * @return set of annotations explicit on the resulting type of the constructor
     */
    public static Set<AnnotationMirror> getExplicitAnnotationsOnConstructorResult(MethodTree constructorDeclaration) {
        Set<AnnotationMirror> annotationSet = AnnotationUtils.createAnnotationSet();
        ModifiersTree modifiersTree = constructorDeclaration.getModifiers();
        if (modifiersTree != null) {
            List<? extends AnnotationTree> annotationTrees = modifiersTree.getAnnotations();
            annotationSet.addAll(InternalUtils.annotationsFromTypeAnnotationTrees(annotationTrees));
        }
        return annotationSet;
    }
}
