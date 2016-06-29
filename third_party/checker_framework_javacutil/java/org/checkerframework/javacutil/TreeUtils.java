package org.checkerframework.javacutil;

/*>>>
import org.checkerframework.checker.nullness.qual.*;
*/

import java.util.EnumSet;
import java.util.Set;

import javax.annotation.processing.ProcessingEnvironment;
import javax.lang.model.element.Element;
import javax.lang.model.element.ElementKind;
import javax.lang.model.element.ExecutableElement;
import javax.lang.model.element.Name;
import javax.lang.model.element.TypeElement;
import javax.lang.model.element.VariableElement;
import javax.lang.model.util.ElementFilter;

import com.sun.source.tree.AnnotatedTypeTree;
import com.sun.source.tree.ArrayAccessTree;
import com.sun.source.tree.BinaryTree;
import com.sun.source.tree.BlockTree;
import com.sun.source.tree.ClassTree;
import com.sun.source.tree.CompoundAssignmentTree;
import com.sun.source.tree.ExpressionStatementTree;
import com.sun.source.tree.ExpressionTree;
import com.sun.source.tree.IdentifierTree;
import com.sun.source.tree.LiteralTree;
import com.sun.source.tree.MemberSelectTree;
import com.sun.source.tree.MethodInvocationTree;
import com.sun.source.tree.MethodTree;
import com.sun.source.tree.NewClassTree;
import com.sun.source.tree.ParameterizedTypeTree;
import com.sun.source.tree.ParenthesizedTree;
import com.sun.source.tree.PrimitiveTypeTree;
import com.sun.source.tree.StatementTree;
import com.sun.source.tree.Tree;
import com.sun.source.tree.TypeCastTree;
import com.sun.source.tree.VariableTree;
import com.sun.source.util.TreePath;
import com.sun.source.util.Trees;
import com.sun.tools.javac.code.Flags;
import com.sun.tools.javac.code.Symbol.MethodSymbol;
import com.sun.tools.javac.tree.JCTree;

/**
 * A utility class made for helping to analyze a given {@code Tree}.
 */
// TODO: This class needs significant restructuring
public final class TreeUtils {

    // Class cannot be instantiated.
    private TreeUtils() { throw new AssertionError("Class TreeUtils cannot be instantiated."); }

    /**
     * Checks if the provided method is a constructor method or no.
     *
     * @param tree
     *            a tree defining the method
     * @return true iff tree describes a constructor
     */
    public static boolean isConstructor(final MethodTree tree) {
        return tree.getName().contentEquals("<init>");
    }

    /**
     * Checks if the method invocation is a call to super.
     *
     * @param tree
     *            a tree defining a method invocation
     *
     * @return true iff tree describes a call to super
     */
    public static boolean isSuperCall(MethodInvocationTree tree) {
        return isNamedMethodCall("super", tree);
    }

    /**
     * Checks if the method invocation is a call to this.
     *
     * @param tree
     *            a tree defining a method invocation
     *
     * @return true iff tree describes a call to this
     */
    public static boolean isThisCall(MethodInvocationTree tree) {
        return isNamedMethodCall("this", tree);

    }

    protected static boolean isNamedMethodCall(String name, MethodInvocationTree tree) {
        /*@Nullable*/ ExpressionTree mst = tree.getMethodSelect();
        assert mst != null; /*nninvariant*/

        if (mst.getKind() == Tree.Kind.IDENTIFIER ) {
            return ((IdentifierTree)mst).getName().contentEquals(name);
        }

        if (mst.getKind() == Tree.Kind.MEMBER_SELECT) {
            MemberSelectTree selectTree = (MemberSelectTree)mst;

            if (selectTree.getExpression().getKind() != Tree.Kind.IDENTIFIER) {
                return false;
            }

            return ((IdentifierTree) selectTree.getExpression()).getName()
                    .contentEquals(name);
        }

        return false;
    }

    /**
     * Returns true if the tree is a tree that 'looks like' either an access
     * of a field or an invocation of a method that are owned by the same
     * accessing instance.
     *
     * It would only return true if the access tree is of the form:
     * <pre>
     *   field
     *   this.field
     *
     *   method()
     *   this.method()
     * </pre>
     *
     * It does not perform any semantical check to differentiate between
     * fields and local variables; local methods or imported static methods.
     *
     * @param tree  expression tree representing an access to object member
     * @return {@code true} iff the member is a member of {@code this} instance
     */
    public static boolean isSelfAccess(final ExpressionTree tree) {
        ExpressionTree tr = TreeUtils.skipParens(tree);
        // If method invocation check the method select
        if (tr.getKind() == Tree.Kind.ARRAY_ACCESS) {
            return false;
        }

        if (tree.getKind() == Tree.Kind.METHOD_INVOCATION) {
            tr = ((MethodInvocationTree)tree).getMethodSelect();
        }
        tr = TreeUtils.skipParens(tr);
        if (tr.getKind() == Tree.Kind.TYPE_CAST) {
            tr = ((TypeCastTree)tr).getExpression();
        }
        tr = TreeUtils.skipParens(tr);

        if (tr.getKind() == Tree.Kind.IDENTIFIER) {
            return true;
        }

        if (tr.getKind() == Tree.Kind.MEMBER_SELECT) {
            tr = ((MemberSelectTree)tr).getExpression();
            if (tr.getKind() == Tree.Kind.IDENTIFIER) {
                Name ident = ((IdentifierTree)tr).getName();
                return ident.contentEquals("this") ||
                        ident.contentEquals("super");
            }
        }

        return false;
    }

    /**
     * Gets the first enclosing tree in path, of the specified kind.
     *
     * @param path  the path defining the tree node
     * @param kind  the kind of the desired tree
     * @return the enclosing tree of the given type as given by the path
     */
    public static Tree enclosingOfKind(final TreePath path, final Tree.Kind kind) {
        return enclosingOfKind(path, EnumSet.of(kind));
    }

    /**
     * Gets the first enclosing tree in path, with any one of the specified kinds.
     *
     * @param path  the path defining the tree node
     * @param kinds  the set of kinds of the desired tree
     * @return the enclosing tree of the given type as given by the path
     */
    public static Tree enclosingOfKind(final TreePath path, final Set<Tree.Kind> kinds) {
        TreePath p = path;

        while (p != null) {
            Tree leaf = p.getLeaf();
            assert leaf != null; /*nninvariant*/
            if (kinds.contains(leaf.getKind())) {
                return leaf;
            }
            p = p.getParentPath();
        }

        return null;
    }

    /**
     * Gets path to the first enclosing class tree, where class is
     * defined by the classTreeKinds method.
     *
     * @param path  the path defining the tree node
     * @return the path to the enclosing class tree
     */
    public static TreePath pathTillClass(final TreePath path) {
        return pathTillOfKind(path, classTreeKinds());
    }

    /**
     * Gets path to the first enclosing tree of the specified kind.
     *
     * @param path  the path defining the tree node
     * @param kind  the kind of the desired tree
     * @return the path to the enclosing tree of the given type
     */
    public static TreePath pathTillOfKind(final TreePath path, final Tree.Kind kind) {
        return pathTillOfKind(path, EnumSet.of(kind));
    }

    /**
     * Gets path to the first enclosing tree with any one of the specified kinds.
     *
     * @param path  the path defining the tree node
     * @param kinds  the set of kinds of the desired tree
     * @return the path to the enclosing tree of the given type
     */
    public static TreePath pathTillOfKind(final TreePath path, final Set<Tree.Kind> kinds) {
        TreePath p = path;

        while (p != null) {
            Tree leaf = p.getLeaf();
            assert leaf != null; /*nninvariant*/
            if (kinds.contains(leaf.getKind())) {
                return p;
            }
            p = p.getParentPath();
        }

        return null;
    }

    /**
     * Gets the first enclosing tree in path, of the specified class
     *
     * @param path  the path defining the tree node
     * @param treeClass the class of the desired tree
     * @return the enclosing tree of the given type as given by the path
     */
    public static <T extends Tree> T enclosingOfClass(final TreePath path, final Class<T> treeClass) {
        TreePath p = path;

        while (p != null) {
            Tree leaf = p.getLeaf();
            if (treeClass.isInstance(leaf)) {
                return treeClass.cast(leaf);
            }
            p = p.getParentPath();
        }

        return null;
    }

    /**
     * Gets the enclosing class of the tree node defined by the given
     * {@code {@link TreePath}}. It returns a {@link Tree}, from which
     * {@code checkers.types.AnnotatedTypeMirror} or {@link Element} can be
     * obtained.
     *
     * @param path the path defining the tree node
     * @return the enclosing class (or interface) as given by the path, or null
     *         if one does not exist
     */
    public static /*@Nullable*/ ClassTree enclosingClass(final /*@Nullable*/ TreePath path) {
        return (ClassTree) enclosingOfKind(path, classTreeKinds());
    }

    /**
     * Gets the enclosing variable of a tree node defined by the given
     * {@link TreePath}.
     *
     * @param path the path defining the tree node
     * @return the enclosing variable as given by the path, or null if one does not exist
     */
    public static VariableTree enclosingVariable(final TreePath path) {
        return (VariableTree) enclosingOfKind(path, Tree.Kind.VARIABLE);
    }

    /**
     * Gets the enclosing method of the tree node defined by the given
     * {@code {@link TreePath}}. It returns a {@link Tree}, from which an
     * {@code checkers.types.AnnotatedTypeMirror} or {@link Element} can be
     * obtained.
     *
     * @param path the path defining the tree node
     * @return the enclosing method as given by the path, or null if one does
     *         not exist
     */
    public static /*@Nullable*/ MethodTree enclosingMethod(final /*@Nullable*/ TreePath path) {
        return (MethodTree) enclosingOfKind(path, Tree.Kind.METHOD);
    }

    public static /*@Nullable*/ BlockTree enclosingTopLevelBlock(TreePath path) {
        TreePath parpath = path.getParentPath();
        while (parpath!=null && parpath.getLeaf().getKind() != Tree.Kind.CLASS) {
            path = parpath;
            parpath = parpath.getParentPath();
        }
        if (path.getLeaf().getKind() == Tree.Kind.BLOCK) {
            return (BlockTree) path.getLeaf();
        }
        return null;
    }


    /**
     * If the given tree is a parenthesized tree, it returns the enclosed
     * non-parenthesized tree. Otherwise, it returns the same tree.
     *
     * @param tree  an expression tree
     * @return  the outermost non-parenthesized tree enclosed by the given tree
     */
    public static ExpressionTree skipParens(final ExpressionTree tree) {
        ExpressionTree t = tree;
        while (t.getKind() == Tree.Kind.PARENTHESIZED)
            t = ((ParenthesizedTree)t).getExpression();
        return t;
    }

    /**
     * Returns the tree with the assignment context for the treePath
     * leaf node.  (Does not handle pseudo-assignment of an argument to
     * a parameter or a receiver expression to a receiver.)
     *
     * The assignment context for the {@code treePath} is the leaf of its parent,
     * if the leaf is one of the following trees:
     * <ul>
     *   <li>AssignmentTree </li>
     *   <li>CompoundAssignmentTree </li>
     *   <li>MethodInvocationTree</li>
     *   <li>NewArrayTree</li>
     *   <li>NewClassTree</li>
     *   <li>ReturnTree</li>
     *   <li>VariableTree</li>
     * </ul>
     *
     * If the leaf is a ConditionalExpressionTree or ParenthesizedTree, then recur on the leaf.
     *
     * Otherwise, null is returned.
     *
     * @return  the assignment context as described
     */
    public static Tree getAssignmentContext(final TreePath treePath) {
        TreePath parentPath = treePath.getParentPath();

        if (parentPath == null) {
            return null;
        }

        Tree parent = parentPath.getLeaf();
        switch (parent.getKind()) {
        case PARENTHESIZED:
        case CONDITIONAL_EXPRESSION:
            return getAssignmentContext(parentPath);
        case ASSIGNMENT:
        case METHOD_INVOCATION:
        case NEW_ARRAY:
        case NEW_CLASS:
        case RETURN:
        case VARIABLE:
            return parent;
        default:
            // 11 Tree.Kinds are CompoundAssignmentTrees,
            // so use instanceof rather than listing all 11.
            if (parent instanceof CompoundAssignmentTree) {
                return parent;
            }
            return null;
        }
    }

    /**
     * Gets the element for a class corresponding to a declaration.
     *
     * @return the element for the given class
     */
    public static final TypeElement elementFromDeclaration(ClassTree node) {
        TypeElement elt = (TypeElement) InternalUtils.symbol(node);
        return elt;
    }

    /**
     * Gets the element for a method corresponding to a declaration.
     *
     * @return the element for the given method
     */
    public static final ExecutableElement elementFromDeclaration(MethodTree node) {
        ExecutableElement elt = (ExecutableElement) InternalUtils.symbol(node);
        return elt;
    }

    /**
     * Gets the element for a variable corresponding to its declaration.
     *
     * @return the element for the given variable
     */
    public static final VariableElement elementFromDeclaration(VariableTree node) {
        VariableElement elt = (VariableElement) InternalUtils.symbol(node);
        return elt;
    }

    /**
     * Gets the element for the declaration corresponding to this use of an element.
     * To get the element for a declaration, use {@link
     * Trees#getElement(TreePath)} instead.
     *
     * TODO: remove this method, as it really doesn't do anything.
     *
     * @param node the tree corresponding to a use of an element
     * @return the element for the corresponding declaration
     */
    public static final Element elementFromUse(ExpressionTree node) {
        return InternalUtils.symbol(node);
    }

    // Specialization for return type.
    public static final ExecutableElement elementFromUse(MethodInvocationTree node) {
        return (ExecutableElement) elementFromUse((ExpressionTree) node);
    }

    // Specialization for return type.
    public static final ExecutableElement elementFromUse(NewClassTree node) {
        return (ExecutableElement) elementFromUse((ExpressionTree) node);
    }


    /**
     * Determine whether the given ExpressionTree has an underlying element.
     *
     * @param node the ExpressionTree to test
     * @return whether the tree refers to an identifier, member select, or method invocation
     */
    public static final boolean isUseOfElement(ExpressionTree node) {
        node = TreeUtils.skipParens(node);
        switch (node.getKind()) {
            case IDENTIFIER:
            case MEMBER_SELECT:
            case METHOD_INVOCATION:
            case NEW_CLASS:
                return true;
            default:
                return false;
        }
    }

    /**
     * @return the name of the invoked method
     */
    public static final Name methodName(MethodInvocationTree node) {
        ExpressionTree expr = node.getMethodSelect();
        if (expr.getKind() == Tree.Kind.IDENTIFIER) {
            return ((IdentifierTree)expr).getName();
        } else if (expr.getKind() == Tree.Kind.MEMBER_SELECT) {
            return ((MemberSelectTree)expr).getIdentifier();
        }
        ErrorReporter.errorAbort("TreeUtils.methodName: cannot be here: " + node);
        return null; // dead code
    }

    /**
     * @return true if the first statement in the body is a self constructor
     *  invocation within a constructor
     */
    public static final boolean containsThisConstructorInvocation(MethodTree node) {
        if (!TreeUtils.isConstructor(node)
                || node.getBody().getStatements().isEmpty())
            return false;

        StatementTree st = node.getBody().getStatements().get(0);
        if (!(st instanceof ExpressionStatementTree)
                || !(((ExpressionStatementTree)st).getExpression() instanceof MethodInvocationTree))
            return false;

        MethodInvocationTree invocation = (MethodInvocationTree)
            ((ExpressionStatementTree)st).getExpression();

        return "this".contentEquals(TreeUtils.methodName(invocation));
    }

    public static final Tree firstStatement(Tree tree) {
        Tree first;
        if (tree.getKind() == Tree.Kind.BLOCK) {
            BlockTree block = (BlockTree)tree;
            if (block.getStatements().isEmpty()) {
                first = block;
            } else {
                first = block.getStatements().iterator().next();
            }
        } else {
            first = tree;
        }
        return first;
    }

    /**
     * Determine whether the given class contains an explicit constructor.
     *
     * @param node a class tree
     * @return true, iff there is an explicit constructor
     */
    public static boolean hasExplicitConstructor(ClassTree node) {
        TypeElement elem = TreeUtils.elementFromDeclaration(node);

        for ( ExecutableElement ee : ElementFilter.constructorsIn(elem.getEnclosedElements())) {
            MethodSymbol ms = (MethodSymbol) ee;
            long mod = ms.flags();

            if ((mod & Flags.SYNTHETIC) == 0) {
                return true;
            }
        }
        return false;
    }

    /**
     * Returns true if the tree is of a diamond type.
     * In contrast to the implementation in TreeInfo, this version
     * works on Trees.
     *
     * @see com.sun.tools.javac.tree.TreeInfo#isDiamond(JCTree)
     */
    public static final boolean isDiamondTree(Tree tree) {
        switch (tree.getKind()) {
        case ANNOTATED_TYPE: return isDiamondTree(((AnnotatedTypeTree)tree).getUnderlyingType());
        case PARAMETERIZED_TYPE: return ((ParameterizedTypeTree)tree).getTypeArguments().isEmpty();
        case NEW_CLASS: return isDiamondTree(((NewClassTree)tree).getIdentifier());
        default: return false;
        }
    }

    /**
     * Returns true if the tree represents a {@code String} concatenation
     * operation
     */
    public static final boolean isStringConcatenation(Tree tree) {
        return (tree.getKind() == Tree.Kind.PLUS
                && TypesUtils.isString(InternalUtils.typeOf(tree)));
    }

    /**
     * Returns true if the compound assignment tree is a string concatenation
     */
    public static final boolean isStringCompoundConcatenation(CompoundAssignmentTree tree) {
        return (tree.getKind() == Tree.Kind.PLUS_ASSIGNMENT
                && TypesUtils.isString(InternalUtils.typeOf(tree)));
    }

    /**
     * Returns true if the node is a constant-time expression.
     *
     * A tree is a constant-time expression if it is:
     * <ol>
     * <li>a literal tree
     * <li>a reference to a final variable initialized with a compile time
     *  constant
     * <li>a String concatenation of two compile time constants
     * </ol>
     */
    public static boolean isCompileTimeString(ExpressionTree node) {
        ExpressionTree tree = TreeUtils.skipParens(node);
        if (tree instanceof LiteralTree) {
            return true;
        }

        if (TreeUtils.isUseOfElement(tree)) {
            Element elt = TreeUtils.elementFromUse(tree);
            return ElementUtils.isCompileTimeConstant(elt);
        } else if (TreeUtils.isStringConcatenation(tree)) {
            BinaryTree binOp = (BinaryTree) tree;
            return isCompileTimeString(binOp.getLeftOperand())
                && isCompileTimeString(binOp.getRightOperand());
        } else {
            return false;
        }
    }

    /**
     * Returns the receiver tree of a field access or a method invocation
     */
    public static ExpressionTree getReceiverTree(ExpressionTree expression) {
        ExpressionTree receiver = TreeUtils.skipParens(expression);

        if (!(receiver.getKind() == Tree.Kind.METHOD_INVOCATION
                || receiver.getKind() == Tree.Kind.MEMBER_SELECT
                || receiver.getKind() == Tree.Kind.IDENTIFIER
                || receiver.getKind() == Tree.Kind.ARRAY_ACCESS)) {
            // No receiver tree for anything but these four kinds.
            return null;
        }

        if (receiver.getKind() == Tree.Kind.METHOD_INVOCATION) {
            // Trying to handle receiver calls to trees of the form
            //     ((m).getArray())
            // returns the type of 'm' in this case
            receiver = ((MethodInvocationTree)receiver).getMethodSelect();

            if (receiver.getKind() == Tree.Kind.IDENTIFIER) {
                // It's a method call "m(foo)" without an explicit receiver
                return null;
            } else if (receiver.getKind() == Tree.Kind.MEMBER_SELECT) {
                receiver = ((MemberSelectTree)receiver).getExpression();
            } else {
                // Otherwise, e.g. a NEW_CLASS: nothing to do.
            }
        } else if (receiver.getKind() == Tree.Kind.IDENTIFIER) {
            // It's a field access on implicit this or a local variable/parameter.
            return null;
        } else if (receiver.getKind() == Tree.Kind.ARRAY_ACCESS) {
            return TreeUtils.skipParens(((ArrayAccessTree)receiver).getExpression());
        } else if (receiver.getKind() == Tree.Kind.MEMBER_SELECT) {
            receiver = ((MemberSelectTree)receiver).getExpression();
            // Avoid int.class
            if (receiver instanceof PrimitiveTypeTree) {
                return null;
            }
        }

        // Receiver is now really just the receiver tree.
        return TreeUtils.skipParens(receiver);
    }

    // TODO: What about anonymous classes?
    // Adding Tree.Kind.NEW_CLASS here doesn't work, because then a
    // tree gets cast to ClassTree when it is actually a NewClassTree,
    // for example in enclosingClass above.
    private final static Set<Tree.Kind> classTreeKinds = EnumSet.of(
            Tree.Kind.CLASS,
            Tree.Kind.ENUM,
            Tree.Kind.INTERFACE,
            Tree.Kind.ANNOTATION_TYPE
    );

    public static Set<Tree.Kind> classTreeKinds() {
        return classTreeKinds;
    }

    /**
     * Is the given tree kind a class, i.e. a class, enum,
     * interface, or annotation type.
     *
     * @param tree the tree to test
     * @return true, iff the given kind is a class kind
     */
    public static boolean isClassTree(Tree tree) {
        return classTreeKinds().contains(tree.getKind());
    }

    private final static Set<Tree.Kind> typeTreeKinds = EnumSet.of(
            Tree.Kind.PRIMITIVE_TYPE,
            Tree.Kind.PARAMETERIZED_TYPE,
            Tree.Kind.TYPE_PARAMETER,
            Tree.Kind.ARRAY_TYPE,
            Tree.Kind.UNBOUNDED_WILDCARD,
            Tree.Kind.EXTENDS_WILDCARD,
            Tree.Kind.SUPER_WILDCARD,
            Tree.Kind.ANNOTATED_TYPE
    );

    public static Set<Tree.Kind> typeTreeKinds() {
        return typeTreeKinds;
    }

    /**
     * Is the given tree a type instantiation?
     *
     * TODO: this is an under-approximation: e.g. an identifier could
     * be either a type use or an expression. How can we distinguish.
     *
     * @param tree the tree to test
     * @return true, iff the given tree is a type
     */
    public static boolean isTypeTree(Tree tree) {
        return typeTreeKinds().contains(tree.getKind());
    }

    /**
     * Returns true if the given element is an invocation of the method, or
     * of any method that overrides that one.
     */
    public static boolean isMethodInvocation(Tree tree, ExecutableElement method, ProcessingEnvironment env) {
        if (!(tree instanceof MethodInvocationTree)) {
            return false;
        }
        MethodInvocationTree methInvok = (MethodInvocationTree)tree;
        ExecutableElement invoked = TreeUtils.elementFromUse(methInvok);
        return isMethod(invoked, method, env);
    }

    /** Returns true if the given element is, or overrides, method. */
    private static boolean isMethod(ExecutableElement questioned, ExecutableElement method, ProcessingEnvironment env) {
        return (questioned.equals(method)
                || env.getElementUtils().overrides(questioned, method,
                        (TypeElement)questioned.getEnclosingElement()));
    }

    /**
     * Returns the ExecutableElement for a method declaration of
     * methodName, in class typeName, with params parameters.
     *
     * TODO: to precisely resolve method overloading, we should use parameter types and not just
     * the number of parameters!
     */
    public static ExecutableElement getMethod(String typeName, String methodName, int params, ProcessingEnvironment env) {
        TypeElement mapElt = env.getElementUtils().getTypeElement(typeName);
        for (ExecutableElement exec : ElementFilter.methodsIn(mapElt.getEnclosedElements())) {
            if (exec.getSimpleName().contentEquals(methodName)
                    && exec.getParameters().size() == params)
                return exec;
        }
        ErrorReporter.errorAbort("TreeUtils.getMethod: shouldn't be here!");
        return null; // dead code
    }

    /**
     * Determine whether the given expression is either "this" or an outer
     * "C.this".
     *
     * <p>
     * TODO: Should this also handle "super"?
     */
    public static final boolean isExplicitThisDereference(ExpressionTree tree) {
        if (tree.getKind() == Tree.Kind.IDENTIFIER
                && ((IdentifierTree)tree).getName().contentEquals("this")) {
            // Explicit this reference "this"
            return true;
        }

        if (tree.getKind() != Tree.Kind.MEMBER_SELECT) {
            return false;
        }

        MemberSelectTree memSelTree = (MemberSelectTree) tree;
        if (memSelTree.getIdentifier().contentEquals("this")) {
            // Outer this reference "C.this"
            return true;
        }
        return false;
    }

    /**
     * Determine whether {@code tree} is a class literal, such
     * as
     *
     * <pre>
     *   <em>Object</em> . <em>class</em>
     * </pre>
     *
     * @return true iff if tree is a class literal
     */
    public static boolean isClassLiteral(Tree tree) {
        if (tree.getKind() != Tree.Kind.MEMBER_SELECT) {
            return false;
        }
        return "class".equals(((MemberSelectTree) tree).getIdentifier().toString());
    }

    /**
     * Determine whether {@code tree} is a field access expressions, such
     * as
     *
     * <pre>
     *   <em>f</em>
     *   <em>obj</em> . <em>f</em>
     * </pre>
     *
     * @return true iff if tree is a field access expression (implicit or
     *         explicit)
     */
    public static boolean isFieldAccess(Tree tree) {
        if (tree.getKind().equals(Tree.Kind.MEMBER_SELECT)) {
            // explicit field access
            MemberSelectTree memberSelect = (MemberSelectTree) tree;
            Element el = TreeUtils.elementFromUse(memberSelect);
            return el.getKind().isField();
        } else if (tree.getKind().equals(Tree.Kind.IDENTIFIER)) {
            // implicit field access
            IdentifierTree ident = (IdentifierTree) tree;
            Element el = TreeUtils.elementFromUse(ident);
            return el.getKind().isField()
                    && !ident.getName().contentEquals("this") && !ident.getName().contentEquals("super");
        }
        return false;
    }

    /**
     * Compute the name of the field that the field access {@code tree}
     * accesses. Requires {@code tree} to be a field access, as determined
     * by {@code isFieldAccess}.
     *
     * @return the name of the field accessed by {@code tree}.
     */
    public static String getFieldName(Tree tree) {
        assert isFieldAccess(tree);
        if (tree.getKind().equals(Tree.Kind.MEMBER_SELECT)) {
            MemberSelectTree mtree = (MemberSelectTree) tree;
            return mtree.getIdentifier().toString();
        } else {
            IdentifierTree itree = (IdentifierTree) tree;
            return itree.getName().toString();
        }
    }

    /**
     * Determine whether {@code tree} refers to a method element, such
     * as
     *
     * <pre>
     *   <em>m</em>(...)
     *   <em>obj</em> . <em>m</em>(...)
     * </pre>
     *
     * @return true iff if tree is a method access expression (implicit or
     *         explicit)
     */
    public static boolean isMethodAccess(Tree tree) {
        if (tree.getKind().equals(Tree.Kind.MEMBER_SELECT)) {
            // explicit method access
            MemberSelectTree memberSelect = (MemberSelectTree) tree;
            Element el = TreeUtils.elementFromUse(memberSelect);
            return el.getKind() == ElementKind.METHOD
                    || el.getKind() == ElementKind.CONSTRUCTOR;
        } else if (tree.getKind().equals(Tree.Kind.IDENTIFIER)) {
            // implicit method access
            IdentifierTree ident = (IdentifierTree) tree;
            // The field "super" and "this" are also legal methods
            if (ident.getName().contentEquals("super")
                    || ident.getName().contentEquals("this")) {
                return true;
            }
            Element el = TreeUtils.elementFromUse(ident);
            return el.getKind() == ElementKind.METHOD
                    || el.getKind() == ElementKind.CONSTRUCTOR;
        }
        return false;
    }

    /**
     * Compute the name of the method that the method access {@code tree}
     * accesses. Requires {@code tree} to be a method access, as determined
     * by {@code isMethodAccess}.
     *
     * @return the name of the method accessed by {@code tree}.
     */
    public static String getMethodName(Tree tree) {
        assert isMethodAccess(tree);
        if (tree.getKind().equals(Tree.Kind.MEMBER_SELECT)) {
            MemberSelectTree mtree = (MemberSelectTree) tree;
            return mtree.getIdentifier().toString();
        } else {
            IdentifierTree itree = (IdentifierTree) tree;
            return itree.getName().toString();
        }
    }

    /**
     * @return {@code true} if and only if {@code tree} can have a type
     *         annotation.
     *
     * TODO: is this implementation precise enough? E.g. does
     * a .class literal work correctly?
     */
    public static boolean canHaveTypeAnnotation(Tree tree) {
        return ((JCTree) tree).type != null;
    }

    /**
     * Returns true if and only if the given {@code tree} represents a field
     * access of the given {@link VariableElement}.
     */
    public static boolean isSpecificFieldAccess(Tree tree, VariableElement var) {
        if (tree instanceof MemberSelectTree) {
            MemberSelectTree memSel = (MemberSelectTree) tree;
            Element field = TreeUtils.elementFromUse(memSel);
            return field.equals(var);
        } else if (tree instanceof IdentifierTree) {
            IdentifierTree idTree = (IdentifierTree) tree;
            Element field = TreeUtils.elementFromUse(idTree);
            return field.equals(var);
        } else {
            return false;
        }
    }

    /**
     * Returns the VariableElement for a field declaration.
     *
     * @param typeName the class where the field is declared
     * @param fieldName the name of the field
     * @param env the processing environment
     * @return the VariableElement for typeName.fieldName
     */
    public static VariableElement getField(String typeName, String fieldName, ProcessingEnvironment env) {
        TypeElement mapElt = env.getElementUtils().getTypeElement(typeName);
        for (VariableElement var : ElementFilter.fieldsIn(mapElt.getEnclosedElements())) {
            if (var.getSimpleName().contentEquals(fieldName)) {
                return var;
            }
        }
        ErrorReporter.errorAbort("TreeUtils.getField: shouldn't be here!");
        return null; // dead code
    }

    /** Determine whether the given tree represents an ExpressionTree.
     *
     * TODO: is there a nicer way than an instanceof?
     *
     * @param tree the Tree to test
     * @return whether the tree is an ExpressionTree
     */
    public static boolean isExpressionTree(Tree tree) {
        return tree instanceof ExpressionTree;
    }

    /**
     * @param node the method invocation to check
     * @return true if this is a super call to the {@link Enum} constructor
     */
    public static boolean isEnumSuper(MethodInvocationTree node) {
        ExecutableElement ex = TreeUtils.elementFromUse(node);
        Name name = ElementUtils.getQualifiedClassName(ex);
        boolean correctClass = "java.lang.Enum".contentEquals(name);
        boolean correctMethod = "<init>".contentEquals(ex.getSimpleName());
        return correctClass && correctMethod;
    }

    /** Determine whether the given tree represents a declaration of a type
     * (including type parameters).
     *
     * @param node  the Tree to test
     * @return true if the tree is a type declaration
     */
    public static boolean isTypeDeclaration(Tree node) {
        switch (node.getKind()) {
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
     * @see Object#getClass()
     * @return true iff invocationTree is an instance of getClass()
     */
    public static boolean isGetClassInvocation(MethodInvocationTree invocationTree) {
        final Element declarationElement = elementFromUse(invocationTree);
        String ownerName = ElementUtils.getQualifiedClassName(declarationElement.getEnclosingElement()).toString();
        return ownerName.equals("java.lang.Object")
                && declarationElement.getSimpleName().toString().equals("getClass");
    }
}
