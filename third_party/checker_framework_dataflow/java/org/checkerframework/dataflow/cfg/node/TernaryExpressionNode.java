package org.checkerframework.dataflow.cfg.node;

import java.util.Collection;
import java.util.LinkedList;

import org.checkerframework.dataflow.util.HashCodeUtils;

import org.checkerframework.javacutil.InternalUtils;

import com.sun.source.tree.ConditionalExpressionTree;
import com.sun.source.tree.Tree.Kind;

/**
 * A node for a conditional expression:
 *
 * <pre>
 *   <em>expression</em> ? <em>expression</em> : <em>expression</em>
 * </pre>
 *
 * @author Stefan Heule
 * @author Charlie Garrett
 *
 */
public class TernaryExpressionNode extends Node {

    protected ConditionalExpressionTree tree;
    protected Node condition;
    protected Node thenOperand;
    protected Node elseOperand;

    public TernaryExpressionNode(ConditionalExpressionTree tree, Node condition,
            Node thenOperand, Node elseOperand) {
        super(InternalUtils.typeOf(tree));
        assert tree.getKind().equals(Kind.CONDITIONAL_EXPRESSION);
        this.tree = tree;
        this.condition = condition;
        this.thenOperand = thenOperand;
        this.elseOperand = elseOperand;
    }

    public Node getConditionOperand() {
        return condition;
    }

    public Node getThenOperand() {
        return thenOperand;
    }

    public Node getElseOperand() {
        return elseOperand;
    }

    @Override
    public ConditionalExpressionTree getTree() {
        return tree;
    }

    @Override
    public <R, P> R accept(NodeVisitor<R, P> visitor, P p) {
        return visitor.visitTernaryExpression(this, p);
    }

    @Override
    public String toString() {
        return "(" + getConditionOperand() + " ? " + getThenOperand() + " : "
                + getElseOperand() + ")";
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == null || !(obj instanceof TernaryExpressionNode)) {
            return false;
        }
        TernaryExpressionNode other = (TernaryExpressionNode) obj;
        return getConditionOperand().equals(other.getConditionOperand())
                && getThenOperand().equals(other.getThenOperand())
                && getElseOperand().equals(other.getElseOperand());
    }

    @Override
    public int hashCode() {
        return HashCodeUtils.hash(getConditionOperand(), getThenOperand(),
                getElseOperand());
    }

    @Override
    public Collection<Node> getOperands() {
        LinkedList<Node> list = new LinkedList<Node>();
        list.add(getConditionOperand());
        list.add(getThenOperand());
        list.add(getElseOperand());
        return list;
    }
}
