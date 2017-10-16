package org.checkerframework.dataflow.cfg.node;

import com.sun.source.tree.BinaryTree;
import com.sun.source.tree.Tree.Kind;
import org.checkerframework.dataflow.util.HashCodeUtils;

/**
 * A node for bitwise right shift operations with zero extension:
 *
 * <pre>
 *   <em>expression</em> &gt;&gt;&gt; <em>expression</em>
 * </pre>
 *
 * @author Stefan Heule
 * @author Charlie Garrett
 */
public class UnsignedRightShiftNode extends BinaryOperationNode {

    public UnsignedRightShiftNode(BinaryTree tree, Node left, Node right) {
        super(tree, left, right);
        assert tree.getKind() == Kind.UNSIGNED_RIGHT_SHIFT;
    }

    @Override
    public <R, P> R accept(NodeVisitor<R, P> visitor, P p) {
        return visitor.visitUnsignedRightShift(this, p);
    }

    @Override
    public String toString() {
        return "(" + getLeftOperand() + " >>> " + getRightOperand() + ")";
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == null || !(obj instanceof UnsignedRightShiftNode)) {
            return false;
        }
        UnsignedRightShiftNode other = (UnsignedRightShiftNode) obj;
        return getLeftOperand().equals(other.getLeftOperand())
                && getRightOperand().equals(other.getRightOperand());
    }

    @Override
    public int hashCode() {
        return HashCodeUtils.hash(getLeftOperand(), getRightOperand());
    }
}
