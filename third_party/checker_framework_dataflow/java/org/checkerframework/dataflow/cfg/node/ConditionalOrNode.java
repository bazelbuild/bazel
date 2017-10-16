package org.checkerframework.dataflow.cfg.node;

import com.sun.source.tree.BinaryTree;
import com.sun.source.tree.Tree.Kind;
import org.checkerframework.dataflow.util.HashCodeUtils;

/**
 * A node for a conditional or expression:
 *
 * <pre>
 *   <em>expression</em> || <em>expression</em>
 * </pre>
 *
 * @author Stefan Heule
 */
public class ConditionalOrNode extends BinaryOperationNode {

    public ConditionalOrNode(BinaryTree tree, Node left, Node right) {
        super(tree, left, right);
        assert tree.getKind().equals(Kind.CONDITIONAL_OR);
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
}
