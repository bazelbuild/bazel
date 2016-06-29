package org.checkerframework.dataflow.cfg.node;

import java.util.Collection;
import java.util.LinkedList;

import org.checkerframework.dataflow.util.HashCodeUtils;

import org.checkerframework.javacutil.InternalUtils;

import com.sun.source.tree.BinaryTree;
import com.sun.source.tree.Tree.Kind;

/**
 * A node for a conditional or expression:
 *
 * <pre>
 *   <em>expression</em> || <em>expression</em>
 * </pre>
 *
 * @author Stefan Heule
 *
 */
public class ConditionalOrNode extends Node {

    protected BinaryTree tree;
    protected Node lhs;
    protected Node rhs;

    public ConditionalOrNode(BinaryTree tree, Node lhs, Node rhs) {
        super(InternalUtils.typeOf(tree));
        assert tree.getKind().equals(Kind.CONDITIONAL_OR);
        this.tree = tree;
        this.lhs = lhs;
        this.rhs = rhs;
    }

    public Node getLeftOperand() {
        return lhs;
    }

    public Node getRightOperand() {
        return rhs;
    }

    @Override
    public BinaryTree getTree() {
        return tree;
    }

    @Override
    public <R, P> R accept(NodeVisitor<R, P> visitor, P p) {
        return visitor.visitConditionalOr(this, p);
    }

    @Override
    public String toString() {
        return "(" + getLeftOperand() + " || " + getRightOperand() + ")";
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == null || !(obj instanceof ConditionalOrNode)) {
            return false;
        }
        ConditionalOrNode other = (ConditionalOrNode) obj;
        return getLeftOperand().equals(other.getLeftOperand())
                && getRightOperand().equals(other.getRightOperand());
    }

    @Override
    public int hashCode() {
        return HashCodeUtils.hash(getLeftOperand(), getRightOperand());
    }

    @Override
    public Collection<Node> getOperands() {
        LinkedList<Node> list = new LinkedList<Node>();
        list.add(getLeftOperand());
        list.add(getRightOperand());
        return list;
    }
}
