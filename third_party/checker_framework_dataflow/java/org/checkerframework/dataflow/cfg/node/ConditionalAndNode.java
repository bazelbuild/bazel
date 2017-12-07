package org.checkerframework.dataflow.cfg.node;

import com.sun.source.tree.BinaryTree;
import com.sun.source.tree.Tree.Kind;
import org.checkerframework.dataflow.util.HashCodeUtils;

/**
 * A node for a conditional and expression:
 *
 * <pre>
 *   <em>expression</em> &amp;&amp; <em>expression</em>
 * </pre>
 *
 * @author Stefan Heule
 * @author Charlie Garrett
 */
public class ConditionalAndNode extends BinaryOperationNode {

    public ConditionalAndNode(BinaryTree tree, Node left, Node right) {
        super(tree, left, right);
        assert tree.getKind().equals(Kind.CONDITIONAL_AND);
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
}
