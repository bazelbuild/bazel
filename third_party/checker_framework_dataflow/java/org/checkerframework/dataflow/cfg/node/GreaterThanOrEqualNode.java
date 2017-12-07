package org.checkerframework.dataflow.cfg.node;

import com.sun.source.tree.BinaryTree;
import com.sun.source.tree.Tree.Kind;
import org.checkerframework.dataflow.util.HashCodeUtils;

/**
 * A node for the greater than or equal comparison:
 *
 * <pre>
 *   <em>expression</em> &gt;= <em>expression</em>
 * </pre>
 *
 * @author Stefan Heule
 * @author Charlie Garrett
 */
public class GreaterThanOrEqualNode extends BinaryOperationNode {

    public GreaterThanOrEqualNode(BinaryTree tree, Node left, Node right) {
        super(tree, left, right);
        assert tree.getKind() == Kind.GREATER_THAN_EQUAL;
    }

    @Override
    public <R, P> R accept(NodeVisitor<R, P> visitor, P p) {
        return visitor.visitGreaterThanOrEqual(this, p);
    }

    @Override
    public String toString() {
        return "(" + getLeftOperand() + " >= " + getRightOperand() + ")";
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == null || !(obj instanceof GreaterThanOrEqualNode)) {
            return false;
        }
        GreaterThanOrEqualNode other = (GreaterThanOrEqualNode) obj;
        return getLeftOperand().equals(other.getLeftOperand())
                && getRightOperand().equals(other.getRightOperand());
    }

    @Override
    public int hashCode() {
        return HashCodeUtils.hash(getLeftOperand(), getRightOperand());
    }
}
