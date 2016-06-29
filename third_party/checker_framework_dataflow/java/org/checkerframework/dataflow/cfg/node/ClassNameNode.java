package org.checkerframework.dataflow.cfg.node;

/*>>>
import org.checkerframework.checker.nullness.qual.Nullable;
*/

import org.checkerframework.dataflow.util.HashCodeUtils;

import org.checkerframework.javacutil.InternalUtils;
import org.checkerframework.javacutil.TreeUtils;

import java.util.Collection;
import java.util.Collections;

import javax.lang.model.element.Element;

import com.sun.source.tree.ClassTree;
import com.sun.source.tree.IdentifierTree;
import com.sun.source.tree.MemberSelectTree;
import com.sun.source.tree.Tree;

/**
 * A node representing a class name used in an expression
 * such as a static method invocation.
 *
 * parent.<em>class</em> .forName(...)
 *
 * @author Stefan Heule
 * @author Charlie Garrett
 *
 */
public class ClassNameNode extends Node {

    protected final Tree tree;
    /** The class named by this node */
    protected final Element element;

    /** The parent name, if any. */
    protected final /*@Nullable*/ Node parent;

    public ClassNameNode(IdentifierTree tree) {
        super(InternalUtils.typeOf(tree));
        assert tree.getKind() == Tree.Kind.IDENTIFIER;
        this.tree = tree;
        this.element = TreeUtils.elementFromUse(tree);
        this.parent = null;
    }

    public ClassNameNode(ClassTree tree) {
        super(InternalUtils.typeOf(tree));
        assert tree.getKind() == Tree.Kind.CLASS || tree.getKind() == Tree.Kind.ENUM || tree.getKind() == Tree.Kind.INTERFACE || tree.getKind() == Tree.Kind.ANNOTATION_TYPE;
        this.tree = tree;
        this.element = TreeUtils.elementFromDeclaration(tree);
        this.parent = null;
    }

    public ClassNameNode(MemberSelectTree tree, Node parent) {
        super(InternalUtils.typeOf(tree));
        this.tree = tree;
        this.element = TreeUtils.elementFromUse(tree);
        this.parent = parent;
    }

    public Element getElement() {
        return element;
    }

    public Node getParent() {
        return parent;
    }

    @Override
    public Tree getTree() {
        return tree;
    }

    @Override
    public <R, P> R accept(NodeVisitor<R, P> visitor, P p) {
        return visitor.visitClassName(this, p);
    }

    @Override
    public String toString() {
        return getElement().getSimpleName().toString();
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == null || !(obj instanceof ClassNameNode)) {
            return false;
        }
        ClassNameNode other = (ClassNameNode) obj;
        if (getParent() == null) {
            return other.getParent() == null
                    && getElement().equals(other.getElement());
        } else {
            return getParent().equals(other.getParent())
                    && getElement().equals(other.getElement());
        }
    }

    @Override
    public int hashCode() {
        if (parent == null) {
            return HashCodeUtils.hash(getElement());
        }
        return HashCodeUtils.hash(getElement(), getParent());
    }

    @Override
    public Collection<Node> getOperands() {
        if (parent == null) {
            return Collections.emptyList();
        }
        return Collections.singleton(parent);
    }
}
