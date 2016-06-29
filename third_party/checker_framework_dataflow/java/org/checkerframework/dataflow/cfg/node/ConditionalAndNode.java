package org.checkerframework.dataflow.cfg.node;

import java.util.Collection;
import java.util.LinkedList;

import org.checkerframework.dataflow.util.HashCodeUtils;

import org.checkerframework.javacutil.InternalUtils;

import com.sun.source.tree.BinaryTree;
import com.sun.source.tree.Tree.Kind;

/**
 * A node for a conditional and expression:
 *
 * <pre>
 *   <em>expression</em> &amp;&amp; <em>expression</em>
 * </pre>
 *
 * @author Stefan Heule
 * @author Charlie Garrett
 *
 */
public class ConditionalAndNode extends Node {

    protected BinaryTree tree;
    protected Node lhs;
    protected Node rhs;

    public ConditionalAndNode(BinaryTree tree, Node lhs, Node rhs) {
        super(InternalUtils.typeOf(tree));
        assert tree.getKind().equals(Kind.CONDITIONAL_AND);
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
        return visitor.visitConditionalAnd(this, p);
    }

    @Override
    public String toString() {
        return "(" + getLeftOperand() + " && " + getRightOperand() + ")";
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == null || !(obj instanceof ConditionalAndNode)) {
            return false;
        }
        ConditionalAndNode other = (ConditionalAndNode) obj;
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
