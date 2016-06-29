package org.checkerframework.dataflow.cfg.node;

import java.util.Collection;
import java.util.Collections;

import javax.lang.model.element.Element;

import org.checkerframework.dataflow.util.HashCodeUtils;

import org.checkerframework.javacutil.InternalUtils;
import org.checkerframework.javacutil.TreeUtils;

import com.sun.source.tree.IdentifierTree;
import com.sun.source.tree.Tree;
import com.sun.source.tree.VariableTree;

/**
 * A node for a local variable or a parameter:
 *
 * <pre>
 *   <em>identifier</em>
 * </pre>
 *
 * We allow local variable uses introduced by the {@link org.checkerframework.dataflow.cfg.CFGBuilder} without
 * corresponding AST {@link Tree}s.
 *
 * @author Stefan Heule
 *
 */
// TODO: don't use for parameters, as they don't have a tree
public class LocalVariableNode extends Node {

    protected Tree tree;
    protected Node receiver;

    public LocalVariableNode(Tree t) {
        super(InternalUtils.typeOf(t));
        // IdentifierTree for normal uses of the local variable or parameter,
        // and VariableTree for the translation of an initializer block
        assert t != null;
        assert t instanceof IdentifierTree || t instanceof VariableTree;
        tree = t;
        this.receiver = null;
    }

    public LocalVariableNode(Tree t, Node receiver) {
        this(t);
        this.receiver = receiver;
    }

    public Element getElement() {
        Element el;
        if (tree instanceof IdentifierTree) {
            el = TreeUtils.elementFromUse((IdentifierTree) tree);
        } else {
            assert tree instanceof VariableTree;
            el = TreeUtils.elementFromDeclaration((VariableTree) tree);
        }
        return el;
    }

    public Node getReceiver() {
        return receiver;
    }

    public String getName() {
        if (tree instanceof IdentifierTree) {
            return ((IdentifierTree) tree).getName().toString();
        }
        return ((VariableTree) tree).getName().toString();
    }

    @Override
    public Tree getTree() {
        return tree;
    }

    @Override
    public <R, P> R accept(NodeVisitor<R, P> visitor, P p) {
        return visitor.visitLocalVariable(this, p);
    }

    @Override
    public String toString() {
        return getName().toString();
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == null || !(obj instanceof LocalVariableNode)) {
            return false;
        }
        LocalVariableNode other = (LocalVariableNode) obj;
        return getName().equals(other.getName());
    }

    @Override
    public int hashCode() {
        return HashCodeUtils.hash(getName());
    }

    @Override
    public Collection<Node> getOperands() {
        return Collections.emptyList();
    }
}
