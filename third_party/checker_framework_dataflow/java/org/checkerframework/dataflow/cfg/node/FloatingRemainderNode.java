package org.checkerframework.dataflow.cfg.node;

import com.sun.source.tree.BinaryTree;
import com.sun.source.tree.Tree.Kind;
import org.checkerframework.dataflow.util.HashCodeUtils;

/**
 * A node for the floating-point remainder:
 *
 * <pre>
 *   <em>expression</em> % <em>expression</em>
 * </pre>
 *
 * @author Stefan Heule
 * @author Charlie Garrett
 */
public class FloatingRemainderNode extends BinaryOperationNode {

    public FloatingRemainderNode(BinaryTree tree, Node left, Node right) {
        super(tree, left, right);
        assert tree.getKind() == Kind.REMAINDER;
    }

    @Override
    public <R, P> R accept(NodeVisitor<R, P> visitor, P p) {
        return visitor.visitFloatingRemainder(this, p);
    }

    @Override
    public String toString() {
        return "(" + getLeftOperand() + " % " + getRightOperand() + ")";
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == null || !(obj instanceof FloatingRemainderNode)) {
            return false;
        }
        FloatingRemainderNode other = (FloatingRemainderNode) obj;
        return getLeftOperand().equals(other.getLeftOperand())
                && getRightOperand().equals(other.getRightOperand());
    }

    @Override
    public int hashCode() {
        return HashCodeUtils.hash(getLeftOperand(), getRightOperand());
    }
}
