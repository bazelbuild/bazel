package org.checkerframework.javacutil;

import com.sun.source.tree.AnnotatedTypeTree;
import com.sun.source.tree.AnnotationTree;
import com.sun.source.tree.ArrayAccessTree;
import com.sun.source.tree.AssignmentTree;
import com.sun.source.tree.ExpressionTree;
import com.sun.source.tree.MethodTree;
import com.sun.source.tree.NewArrayTree;
import com.sun.source.tree.NewClassTree;
import com.sun.source.tree.Tree;
import com.sun.source.tree.TypeParameterTree;
import com.sun.source.util.TreePath;
import com.sun.tools.javac.code.Flags;
import com.sun.tools.javac.code.Symbol;
import com.sun.tools.javac.code.Symbol.TypeSymbol;
import com.sun.tools.javac.code.Type;
import com.sun.tools.javac.code.Type.CapturedType;
import com.sun.tools.javac.code.Types;
import com.sun.tools.javac.processing.JavacProcessingEnvironment;
import com.sun.tools.javac.tree.JCTree;
import com.sun.tools.javac.tree.JCTree.JCAnnotatedType;
import com.sun.tools.javac.tree.JCTree.JCAnnotation;
import com.sun.tools.javac.tree.JCTree.JCExpressionStatement;
import com.sun.tools.javac.tree.JCTree.JCMemberReference;
import com.sun.tools.javac.tree.JCTree.JCMethodDecl;
import com.sun.tools.javac.tree.JCTree.JCMethodInvocation;
import com.sun.tools.javac.tree.JCTree.JCNewArray;
import com.sun.tools.javac.tree.JCTree.JCNewClass;
import com.sun.tools.javac.tree.JCTree.JCTypeParameter;
import com.sun.tools.javac.tree.TreeInfo;
import com.sun.tools.javac.util.Context;
import com.sun.tools.javac.util.JCDiagnostic.DiagnosticPosition;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import javax.annotation.processing.ProcessingEnvironment;
import javax.lang.model.element.AnnotationMirror;
import javax.lang.model.element.Element;
import javax.lang.model.element.ExecutableElement;
import javax.lang.model.element.TypeElement;
import javax.lang.model.type.TypeKind;
import javax.lang.model.type.TypeMirror;
import javax.lang.model.type.TypeVariable;
import javax.lang.model.type.WildcardType;
import javax.lang.model.util.Elements;

/*>>>
import org.checkerframework.checker.nullness.qual.*;
*/

/**
 * Static utility methods used by annotation abstractions in this package. Some methods in this
 * class depend on the use of Sun javac internals; any procedure in the Checker Framework that uses
 * a non-public API should be placed here.
 */
public class InternalUtils {

    // Class cannot be instantiated.
    private InternalUtils() {
        throw new AssertionError("Class InternalUtils cannot be instantiated.");
    }

    /**
     * Gets the {@link Element} ("symbol") for the given Tree API node.
     *
     * @param tree the {@link Tree} node to get the symbol for
     * @throws IllegalArgumentException if {@code tree} is null or is not a valid javac-internal
     *     tree (JCTree)
     * @return the {@link Symbol} for the given tree, or null if one could not be found
     */
    public static /*@Nullable*/ Element symbol(Tree tree) {
        if (tree == null) {
            ErrorReporter.errorAbort("InternalUtils.symbol: tree is null");
            return null; // dead code
        }

        if (!(tree instanceof JCTree)) {
            ErrorReporter.errorAbort("InternalUtils.symbol: tree is not a valid Javac tree");
            return null; // dead code
        }

        if (TreeUtils.isExpressionTree(tree)) {
            tree = TreeUtils.skipParens((ExpressionTree) tree);
        }

        switch (tree.getKind()) {
            case VARIABLE:
            case METHOD:
            case CLASS:
            case ENUM:
            case INTERFACE:
            case ANNOTATION_TYPE:
            case TYPE_PARAMETER:
                return TreeInfo.symbolFor((JCTree) tree);

                // symbol() only works on MethodSelects, so we need to get it manually
                // for method invocations.
            case METHOD_INVOCATION:
                return TreeInfo.symbol(((JCMethodInvocation) tree).getMethodSelect());

            case ASSIGNMENT:
                return TreeInfo.symbol((JCTree) ((AssignmentTree) tree).getVariable());

            case ARRAY_ACCESS:
                return symbol(((ArrayAccessTree) tree).getExpression());

            case NEW_CLASS:
                return ((JCNewClass) tree).constructor;

            case MEMBER_REFERENCE:
                // TreeInfo.symbol, which is used in the default case, didn't handle
                // member references until JDK8u20. So handle it here.
                return ((JCMemberReference) tree).sym;

            default:
                return TreeInfo.symbol((JCTree) tree);
        }
    }

    /**
     * Determines whether or not the node referred to by the given {@link TreePath} is an anonymous
     * constructor (the constructor for an anonymous class.
     *
     * @param method the {@link TreePath} for a node that may be an anonymous constructor
     * @return true if the given path points to an anonymous constructor, false if it does not
     */
    public static boolean isAnonymousConstructor(final MethodTree method) {
        /*@Nullable*/ Element e = InternalUtils.symbol(method);
        if (e == null || !(e instanceof Symbol)) {
            return false;
        }

        if ((((/*@NonNull*/ Symbol) e).flags() & Flags.ANONCONSTR) != 0) {
            return true;
        }

        return false;
    }

    /**
     * indicates whether it should return the constructor that gets invoked in cases of anonymous
     * classes
     */
    private static final boolean RETURN_INVOKE_CONSTRUCTOR = true;

    /**
     * Determines the symbol for a constructor given an invocation via {@code new}.
     *
     * <p>If the tree is a declaration of an anonymous class, then method returns constructor that
     * gets invoked in the extended class, rather than the anonymous constructor implicitly added by
     * the constructor (JLS 15.9.5.1)
     *
     * @param tree the constructor invocation
     * @return the {@link ExecutableElement} corresponding to the constructor call in {@code tree}
     */
    public static ExecutableElement constructor(NewClassTree tree) {

        if (!(tree instanceof JCTree.JCNewClass)) {
            ErrorReporter.errorAbort("InternalUtils.constructor: not a javac internal tree");
            return null; // dead code
        }

        JCNewClass newClassTree = (JCNewClass) tree;

        if (RETURN_INVOKE_CONSTRUCTOR && tree.getClassBody() != null) {
            // anonymous constructor bodies should contain exactly one statement
            // in the form:
            //    super(arg1, ...)
            // or
            //    o.super(arg1, ...)
            //
            // which is a method invocation (!) to the actual constructor

            // the method call is guaranteed to return nonnull
            JCMethodDecl anonConstructor =
                    (JCMethodDecl) TreeInfo.declarationFor(newClassTree.constructor, newClassTree);
            assert anonConstructor != null;
            assert anonConstructor.body.stats.size() == 1;
            JCExpressionStatement stmt = (JCExpressionStatement) anonConstructor.body.stats.head;
            JCTree.JCMethodInvocation superInvok = (JCMethodInvocation) stmt.expr;
            return (ExecutableElement) TreeInfo.symbol(superInvok.meth);
        }

        Element e = newClassTree.constructor;

        assert e instanceof ExecutableElement;

        return (ExecutableElement) e;
    }

    public static final List<AnnotationMirror> annotationsFromTypeAnnotationTrees(
            List<? extends AnnotationTree> annos) {
        List<AnnotationMirror> annotations = new ArrayList<AnnotationMirror>(annos.size());
        for (AnnotationTree anno : annos) {
            annotations.add(annotationFromAnnotationTree(anno));
        }
        return annotations;
    }

    public static AnnotationMirror annotationFromAnnotationTree(AnnotationTree tree) {
        return ((JCAnnotation) tree).attribute;
    }

    public static final List<? extends AnnotationMirror> annotationsFromTree(
            AnnotatedTypeTree node) {
        return annotationsFromTypeAnnotationTrees(((JCAnnotatedType) node).annotations);
    }

    public static final List<? extends AnnotationMirror> annotationsFromTree(
            TypeParameterTree node) {
        return annotationsFromTypeAnnotationTrees(((JCTypeParameter) node).annotations);
    }

    public static final List<? extends AnnotationMirror> annotationsFromArrayCreation(
            NewArrayTree node, int level) {

        assert node instanceof JCNewArray;
        final JCNewArray newArray = ((JCNewArray) node);

        if (level == -1) {
            return annotationsFromTypeAnnotationTrees(newArray.annotations);
        }

        if (newArray.dimAnnotations.length() > 0
                && (level >= 0)
                && (level < newArray.dimAnnotations.size()))
            return annotationsFromTypeAnnotationTrees(newArray.dimAnnotations.get(level));

        return Collections.emptyList();
    }

    public static TypeMirror typeOf(Tree tree) {
        return ((JCTree) tree).type;
    }

    /** Returns whether a TypeVariable represents a captured type. */
    public static boolean isCaptured(TypeVariable typeVar) {
        return ((Type.TypeVar) TypeAnnotationUtils.unannotatedType(typeVar)).isCaptured();
    }

    /** If typeVar is a captured wildcard, returns that wildcard; otherwise returns null. */
    public static WildcardType getCapturedWildcard(TypeVariable typeVar) {
        if (isCaptured(typeVar)) {
            return ((CapturedType) TypeAnnotationUtils.unannotatedType(typeVar)).wildcard;
        }
        return null;
    }

    /** Returns whether a TypeMirror represents a class type. */
    public static boolean isClassType(TypeMirror type) {
        return (type instanceof Type.ClassType);
    }

    /**
     * Returns the least upper bound of two {@link TypeMirror}s, ignoring any annotations on the
     * types.
     *
     * <p>Wrapper around Types.lub to add special handling for null types, primitives, and
     * wildcards.
     *
     * @param processingEnv the {@link ProcessingEnvironment} to use
     * @param tm1 a {@link TypeMirror}
     * @param tm2 a {@link TypeMirror}
     * @return the least upper bound of {@code tm1} and {@code tm2}.
     */
    public static TypeMirror leastUpperBound(
            ProcessingEnvironment processingEnv, TypeMirror tm1, TypeMirror tm2) {
        Type t1 = TypeAnnotationUtils.unannotatedType(tm1);
        Type t2 = TypeAnnotationUtils.unannotatedType(tm2);
        JavacProcessingEnvironment javacEnv = (JavacProcessingEnvironment) processingEnv;
        Types types = Types.instance(javacEnv.getContext());
        if (types.isSameType(t1, t2)) {
            // Special case if the two types are equal.
            return t1;
        }
        // Handle the 'null' type manually (not done by types.lub).
        if (t1.getKind() == TypeKind.NULL) {
            return t2;
        }
        if (t2.getKind() == TypeKind.NULL) {
            return t1;
        }
        if (t1.getKind() == TypeKind.WILDCARD) {
            WildcardType wc1 = (WildcardType) t1;
            Type bound = (Type) wc1.getExtendsBound();
            if (bound == null) {
                // Implicit upper bound of java.lang.Object
                Elements elements = processingEnv.getElementUtils();
                return elements.getTypeElement("java.lang.Object").asType();
            }
            t1 = bound;
        }
        if (t2.getKind() == TypeKind.WILDCARD) {
            WildcardType wc2 = (WildcardType) t2;
            Type bound = (Type) wc2.getExtendsBound();
            if (bound == null) {
                // Implicit upper bound of java.lang.Object
                Elements elements = processingEnv.getElementUtils();
                return elements.getTypeElement("java.lang.Object").asType();
            }
            t2 = bound;
        }
        // Special case for primitives.
        if (TypesUtils.isPrimitive(t1) || TypesUtils.isPrimitive(t2)) {
            if (types.isAssignable(t1, t2)) {
                return t2;
            } else if (types.isAssignable(t2, t1)) {
                return t1;
            } else {
                Elements elements = processingEnv.getElementUtils();
                return elements.getTypeElement("java.lang.Object").asType();
            }
        }
        return types.lub(t1, t2);
    }

    /**
     * Returns the greatest lower bound of two {@link TypeMirror}s, ignoring any annotations on the
     * types.
     *
     * <p>Wrapper around Types.glb to add special handling for null types, primitives, and
     * wildcards.
     *
     * @param processingEnv the {@link ProcessingEnvironment} to use
     * @param tm1 a {@link TypeMirror}
     * @param tm2 a {@link TypeMirror}
     * @return the greatest lower bound of {@code tm1} and {@code tm2}.
     */
    public static TypeMirror greatestLowerBound(
            ProcessingEnvironment processingEnv, TypeMirror tm1, TypeMirror tm2) {
        Type t1 = TypeAnnotationUtils.unannotatedType(tm1);
        Type t2 = TypeAnnotationUtils.unannotatedType(tm2);
        JavacProcessingEnvironment javacEnv = (JavacProcessingEnvironment) processingEnv;
        Types types = Types.instance(javacEnv.getContext());
        if (types.isSameType(t1, t2)) {
            // Special case if the two types are equal.
            return t1;
        }
        // Handle the 'null' type manually.
        if (t1.getKind() == TypeKind.NULL) {
            return t1;
        }
        if (t2.getKind() == TypeKind.NULL) {
            return t2;
        }
        // Special case for primitives.
        if (TypesUtils.isPrimitive(t1) || TypesUtils.isPrimitive(t2)) {
            if (types.isAssignable(t1, t2)) {
                return t1;
            } else if (types.isAssignable(t2, t1)) {
                return t2;
            } else {
                // Javac types.glb returns TypeKind.Error when the GLB does
                // not exist, but we can't create one.  Use TypeKind.NONE
                // instead.
                return processingEnv.getTypeUtils().getNoType(TypeKind.NONE);
            }
        }
        if (t1.getKind() == TypeKind.WILDCARD) {
            return t2;
        }
        if (t2.getKind() == TypeKind.WILDCARD) {
            return t1;
        }

        // If neither type is a primitive type, null type, or wildcard
        // and if the types are not the same, use javac types.glb
        return types.glb(t1, t2);
    }

    /**
     * Returns the return type of a method, where the "raw" return type of that method is given
     * (i.e., the return type might still contain unsubstituted type variables), given the receiver
     * of the method call.
     */
    public static TypeMirror substituteMethodReturnType(
            TypeMirror methodType, TypeMirror substitutedReceiverType) {
        if (methodType.getKind() != TypeKind.TYPEVAR) {
            return methodType;
        }
        // TODO: find a nicer way to substitute type variables
        String t = TypeAnnotationUtils.unannotatedType(methodType).toString();
        Type finalReceiverType = (Type) substitutedReceiverType;
        int i = 0;
        for (TypeSymbol typeParam : finalReceiverType.tsym.getTypeParameters()) {
            if (t.equals(typeParam.toString())) {
                return finalReceiverType.getTypeArguments().get(i);
            }
            i++;
        }
        assert false;
        return null;
    }

    /**
     * Helper function to extract the javac Context from the javac processing environment.
     *
     * @param env the processing environment
     * @return the javac Context
     */
    public static Context getJavacContext(ProcessingEnvironment env) {
        return ((JavacProcessingEnvironment) env).getContext();
    }

    /**
     * Returns the type element for {@code type} if {@code type} is a class, interface, annotation
     * type, or enum. Otherwise, returns null.
     *
     * @param type whose element is returned
     * @return the type element for {@code type} if {@code type} is a class, interface, annotation
     *     type, or enum; otherwise, returns null
     */
    public static TypeElement getTypeElement(TypeMirror type) {
        Element element = ((Type) type).asElement();
        switch (element.getKind()) {
            case ANNOTATION_TYPE:
            case CLASS:
            case ENUM:
            case INTERFACE:
                return (TypeElement) element;
            default:
                return null;
        }
    }

    /**
     * Obtain the class loader for {@code clazz}. If that is not available, return the system class
     * loader.
     *
     * @param clazz the class whose class loader to find
     * @return the class loader used to {@code clazz}, or the system class loader, or null if both
     *     are unavailable
     */
    public static ClassLoader getClassLoaderForClass(Class<? extends Object> clazz) {
        ClassLoader classLoader = clazz.getClassLoader();
        return classLoader == null ? ClassLoader.getSystemClassLoader() : classLoader;
    }

    /**
     * Compares tree1 to tree2 by the position at which a diagnostic (e.g., an error message) for
     * the tree should be printed.
     */
    public static int compareDiagnosticPosition(Tree tree1, Tree tree2) {
        DiagnosticPosition pos1 = (DiagnosticPosition) tree1;
        DiagnosticPosition pos2 = (DiagnosticPosition) tree2;

        int preferred = Integer.compare(pos1.getPreferredPosition(), pos2.getPreferredPosition());
        if (preferred != 0) {
            return preferred;
        }

        return Integer.compare(pos1.getStartPosition(), pos2.getStartPosition());
    }

    /**
     * Returns whether or not {@code type} is a functional interface type (as defined in JLS 9.8).
     *
     * @param type possible functional interface type
     * @param env ProcessingEnvironment
     * @return whether or not {@code type} is a functional interface type (as defined in JLS 9.8)
     */
    public static boolean isFunctionalInterface(TypeMirror type, ProcessingEnvironment env) {
        Context ctx = ((JavacProcessingEnvironment) env).getContext();
        com.sun.tools.javac.code.Types javacTypes = com.sun.tools.javac.code.Types.instance(ctx);
        return javacTypes.isFunctionalInterface((Type) type);
    }

    /**
     * The type of the lambda or method reference tree is a functional interface type. This method
     * returns the single abstract method declared by that functional interface. (The type of this
     * method is referred to as the function type.)
     *
     * @param tree lambda or member reference tree
     * @param env ProcessingEnvironment
     * @return the single abstract method declared by the type of the tree
     */
    public static Symbol findFunction(Tree tree, ProcessingEnvironment env) {
        Context ctx = ((JavacProcessingEnvironment) env).getContext();
        com.sun.tools.javac.code.Types javacTypes = com.sun.tools.javac.code.Types.instance(ctx);
        return javacTypes.findDescriptorSymbol(((Type) typeOf(tree)).asElement());
    }
}
