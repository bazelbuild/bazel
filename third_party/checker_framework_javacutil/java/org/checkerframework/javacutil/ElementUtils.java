package org.checkerframework.javacutil;

/*>>>
import org.checkerframework.checker.nullness.qual.Nullable;
*/

import static com.sun.tools.javac.code.Flags.ABSTRACT;
import static com.sun.tools.javac.code.Flags.EFFECTIVELY_FINAL;
import static com.sun.tools.javac.code.Flags.FINAL;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Deque;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import javax.lang.model.element.Element;
import javax.lang.model.element.ElementKind;
import javax.lang.model.element.ExecutableElement;
import javax.lang.model.element.Modifier;
import javax.lang.model.element.Name;
import javax.lang.model.element.PackageElement;
import javax.lang.model.element.TypeElement;
import javax.lang.model.element.VariableElement;
import javax.lang.model.type.DeclaredType;
import javax.lang.model.type.TypeKind;
import javax.lang.model.type.TypeMirror;
import javax.lang.model.util.ElementFilter;
import javax.lang.model.util.Elements;

import com.sun.tools.javac.code.Symbol;

/**
 * A Utility class for analyzing {@code Element}s.
 */
public class ElementUtils {

    // Class cannot be instantiated.
    private ElementUtils() { throw new AssertionError("Class ElementUtils cannot be instantiated."); }

    /**
     * Returns the innermost type element enclosing the given element
     *
     * @param elem the enclosed element of a class
     * @return  the innermost type element
     */
    public static TypeElement enclosingClass(final Element elem) {
        Element result = elem;
        while (result != null && !result.getKind().isClass()
                && !result.getKind().isInterface()) {
            /*@Nullable*/ Element encl = result.getEnclosingElement();
            result = encl;
        }
        return (TypeElement) result;
    }

    /**
     * Returns the innermost package element enclosing the given element.
     * The same effect as {@link javax.lang.model.util.Elements#getPackageOf(Element)}.
     * Returns the element itself if it is a package.
     *
     * @param elem the enclosed element of a package
     * @return the innermost package element
     *
     */
    public static PackageElement enclosingPackage(final Element elem) {
        Element result = elem;
        while (result != null && result.getKind() != ElementKind.PACKAGE) {
            /*@Nullable*/ Element encl = result.getEnclosingElement();
            result = encl;
        }
        return (PackageElement) result;
    }

    /**
     * Returns the "parent" package element for the given package element.
     * For package "A.B" it gives "A".
     * For package "A" it gives the default package.
     * For the default package it returns null;
     *
     * Note that packages are not enclosed within each other, we have to manually climb
     * the namespaces. Calling "enclosingPackage" on a package element returns the
     * package element itself again.
     *
     * @param elem the package to start from
     * @return the parent package element
     *
     */
    public static PackageElement parentPackage(final Elements e, final PackageElement elem) {
        // The following might do the same thing:
        //   ((Symbol) elt).owner;
        // TODO: verify and see whether the change is worth it.
        String fqnstart = elem.getQualifiedName().toString();
        String fqn = fqnstart;
        if (fqn != null && !fqn.isEmpty() && fqn.contains(".")) {
            fqn = fqn.substring(0, fqn.lastIndexOf('.'));
            return e.getPackageElement(fqn);
        }
        return null;
    }

    /**
     * Returns true if the element is a static element: whether it is a static
     * field, static method, or static class
     *
     * @return true if element is static
     */
    public static boolean isStatic(Element element) {
        return element.getModifiers().contains(Modifier.STATIC);
    }

    /**
     * Returns true if the element is a final element: a final field, final
     * method, or final class
     *
     * @return true if the element is final
     */
    public static boolean isFinal(Element element) {
        return element.getModifiers().contains(Modifier.FINAL);
    }

    /**
     * Returns true if the element is a effectively final element.
     *
     * @return true if the element is effectively final
     */
    public static boolean isEffectivelyFinal(Element element) {
        Symbol sym = (Symbol) element;
        if (sym.getEnclosingElement().getKind() == ElementKind.METHOD &&
                (sym.getEnclosingElement().flags() & ABSTRACT) != 0) {
            return true;
        }
        return (sym.flags() & (FINAL | EFFECTIVELY_FINAL)) != 0;
    }

    /**
     * Returns the {@code TypeMirror} for usage of Element as a value. It
     * returns the return type of a method element, the class type of a
     * constructor, or simply the type mirror of the element itself.
     *
     * @return  the type for the element used as a value
     */
    public static TypeMirror getType(Element element) {
        if (element.getKind() == ElementKind.METHOD) {
            return ((ExecutableElement)element).getReturnType();
        } else if (element.getKind() == ElementKind.CONSTRUCTOR) {
            return enclosingClass(element).asType();
        } else {
            return element.asType();
        }
    }

    /**
     * Returns the qualified name of the inner most class enclosing
     * the provided {@code Element}
     *
     * @param element
     *            an element enclosed by a class, or a
     *            {@code TypeElement}
     * @return the qualified {@code Name} of the innermost class
     *         enclosing the element
     */
    public static /*@Nullable*/ Name getQualifiedClassName(Element element) {
        if (element.getKind() == ElementKind.PACKAGE) {
            PackageElement elem = (PackageElement) element;
            return elem.getQualifiedName();
        }

        TypeElement elem = enclosingClass(element);
        if (elem == null) {
            return null;
        }

        return elem.getQualifiedName();
    }

    /**
     * Returns a verbose name that identifies the element.
     */
    public static String getVerboseName(Element elt) {
        if (elt.getKind() == ElementKind.PACKAGE ||
                elt.getKind().isClass() ||
                elt.getKind().isInterface()) {
            return getQualifiedClassName(elt).toString();
        } else {
            return getQualifiedClassName(elt) + "." + elt.toString();
        }
    }

    /**
     * Check if the element is an element for 'java.lang.Object'
     *
     * @param element   the type element
     * @return true iff the element is java.lang.Object element
     */
    public static boolean isObject(TypeElement element) {
        return element.getQualifiedName().contentEquals("java.lang.Object");
    }

    /**
     * Returns true if the element is a constant time reference
     */
    public static boolean isCompileTimeConstant(Element elt) {
        return elt != null
            && (elt.getKind() == ElementKind.FIELD || elt.getKind() == ElementKind.LOCAL_VARIABLE)
            && ((VariableElement)elt).getConstantValue() != null;
    }

    /**
     * Returns true if the element is declared in ByteCode.
     * Always return false if elt is a package.
     */
    public static boolean isElementFromByteCode(Element elt) {
        if (elt == null) {
            return false;
        }

        if (elt instanceof Symbol.ClassSymbol) {
            Symbol.ClassSymbol clss = (Symbol.ClassSymbol) elt;
            if (null != clss.classfile) {
                // The class file could be a .java file
                return clss.classfile.getName().endsWith(".class");
            } else {
                return false;
            }
        }
        return isElementFromByteCode(elt.getEnclosingElement(), elt);
    }

    /**
     * Returns true if the element is declared in ByteCode.
     * Always return false if elt is a package.
     */
    private static boolean isElementFromByteCode(Element elt, Element orig) {
        if (elt == null) {
            return false;
        }
        if (elt instanceof Symbol.ClassSymbol) {
            Symbol.ClassSymbol clss = (Symbol.ClassSymbol) elt;
            if (null != clss.classfile) {
                // The class file could be a .java file
                return (clss.classfile.getName().endsWith(".class") ||
                        clss.classfile.getName().endsWith(".class)") ||
                        clss.classfile.getName().endsWith(".class)]"));
            } else {
                return false;
            }
        }
        return isElementFromByteCode(elt.getEnclosingElement(), elt);
    }

    /**
     * Returns the field of the class
     */
    public static VariableElement findFieldInType(TypeElement type, String name) {
        for (VariableElement field: ElementFilter.fieldsIn(type.getEnclosedElements())) {
            if (field.getSimpleName().toString().equals(name)) {
                return field;
            }
        }
        return null;
    }

    public static Set<VariableElement> findFieldsInType(TypeElement type, Collection<String> names) {
        Set<VariableElement> results = new HashSet<VariableElement>();
        for (VariableElement field: ElementFilter.fieldsIn(type.getEnclosedElements())) {
            if (names.contains(field.getSimpleName().toString())) {
                results.add(field);
            }
        }
        return results;
    }

    public static boolean isError(Element element) {
        return element.getClass().getName().equals("com.sun.tools.javac.comp.Resolve$SymbolNotFoundError");
    }

    /**
     * Does the given element need a receiver for accesses?
     * For example, an access to a local variable does not require a receiver.
     *
     * @param element the element to test
     * @return whether the element requires a receiver for accesses
     */
    public static boolean hasReceiver(Element element) {
        return (element.getKind().isField() ||
                element.getKind() == ElementKind.METHOD ||
                element.getKind() == ElementKind.CONSTRUCTOR)
                && !ElementUtils.isStatic(element);
    }

    /**
     * Determine all type elements for the classes and interfaces referenced
     * in the extends/implements clauses of the given type element.
     * TODO: can we learn from the implementation of
     * com.sun.tools.javac.model.JavacElements.getAllMembers(TypeElement)?
     */
    public static List<TypeElement> getSuperTypes(Elements elements, TypeElement type) {

        List<TypeElement> superelems = new ArrayList<TypeElement>();
        if (type == null) {
            return superelems;
        }

        // Set up a stack containing type, which is our starting point.
        Deque<TypeElement> stack = new ArrayDeque<TypeElement>();
        stack.push(type);

        while (!stack.isEmpty()) {
            TypeElement current = stack.pop();

            // For each direct supertype of the current type element, if it
            // hasn't already been visited, push it onto the stack and
            // add it to our superelems set.
            TypeMirror supertypecls = current.getSuperclass();
            if (supertypecls.getKind() != TypeKind.NONE) {
                TypeElement supercls = (TypeElement) ((DeclaredType)supertypecls).asElement();
                if (!superelems.contains(supercls)) {
                    stack.push(supercls);
                    superelems.add(supercls);
                }
            }
            for (TypeMirror supertypeitf : current.getInterfaces()) {
                TypeElement superitf = (TypeElement) ((DeclaredType)supertypeitf).asElement();
                if (!superelems.contains(superitf)) {
                    stack.push(superitf);
                    superelems.add(superitf);
                }
            }
        }

        // Include java.lang.Object as implicit superclass for all classes and interfaces.
        TypeElement jlobject = elements.getTypeElement("java.lang.Object");
        if (!superelems.contains(jlobject)) {
            superelems.add(jlobject);
        }

        return Collections.<TypeElement>unmodifiableList(superelems);
    }

    /**
     * Return all fields declared in the given type or any superclass/interface.
     * TODO: should this use javax.lang.model.util.Elements.getAllMembers(TypeElement)
     * instead of our own getSuperTypes?
     */
    public static List<VariableElement> getAllFieldsIn(Elements elements, TypeElement type) {
        List<VariableElement> fields = new ArrayList<VariableElement>();
        fields.addAll(ElementFilter.fieldsIn(type.getEnclosedElements()));
        List<TypeElement> alltypes = getSuperTypes(elements, type);
        for (TypeElement atype : alltypes) {
            fields.addAll(ElementFilter.fieldsIn(atype.getEnclosedElements()));
        }
        return Collections.<VariableElement>unmodifiableList(fields);
    }

    /**
     * Return all methods declared in the given type or any superclass/interface.
     * Note that no constructors will be returned.
     * TODO: should this use javax.lang.model.util.Elements.getAllMembers(TypeElement)
     * instead of our own getSuperTypes?
     */
    public static List<ExecutableElement> getAllMethodsIn(Elements elements, TypeElement type) {
        List<ExecutableElement> meths = new ArrayList<ExecutableElement>();
        meths.addAll(ElementFilter.methodsIn(type.getEnclosedElements()));

        List<TypeElement> alltypes = getSuperTypes(elements, type);
        for (TypeElement atype : alltypes) {
            meths.addAll(ElementFilter.methodsIn(atype.getEnclosedElements()));
        }
        return Collections.<ExecutableElement>unmodifiableList(meths);
    }

    public static boolean isTypeDeclaration(Element elt) {
        switch (elt.getKind()) {
            // These tree kinds are always declarations.  Uses of the declared
            // types have tree kind IDENTIFIER.
            case ANNOTATION_TYPE:
            case CLASS:
            case ENUM:
            case INTERFACE:
            case TYPE_PARAMETER:
                return true;

            default:
                return false;
        }
    }

    /**
     * Check that a method Element matches a signature.
     *
     * Note: Matching the receiver type must be done elsewhere as
     * the Element receiver type is only populated when annotated.
     *
     * @param method the method Element
     * @param methodName the name of the method
     * @param parameters the formal parameters' Classes
     * @return true if the method matches
     */
    public static boolean matchesElement(ExecutableElement method,
            String methodName,
            Class<?> ... parameters) {

        if (!method.getSimpleName().toString().equals(methodName)) {
            return false;
        }

        if (method.getParameters().size() != parameters.length) {
            return false;
        } else {
            for (int i = 0; i < method.getParameters().size(); i++) {
                if (!method.getParameters().get(i).asType().toString().equals(
                        parameters[i].getName())) {

                    return false;
                }
            }
        }

        return true;
    }
}
