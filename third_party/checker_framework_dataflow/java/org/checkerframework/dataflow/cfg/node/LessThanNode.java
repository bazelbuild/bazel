package org.checkerframework.dataflow.cfg.node;

import com.sun.source.tree.BinaryTree;
import com.sun.source.tree.Tree;
import com.sun.source.tree.Tree.Kind;
import org.checkerframework.dataflow.util.HashCodeUtils;

/**
 * A node for the less than comparison:
 *
 * <pre>
 *   <em>expression</em> &lt; <em>expression</em>
 * </pre>
 *
 * We allow less than nodes without corresponding AST {@link Tree}s.
 *
 * @author Stefan Heule
 * @author Charlie Garrett
 */
public class LessThanNode extends BinaryOperationNode {

    public LessThanNode(BinaryTree tree, Node left, Node right) {
        super(tree, left, right);
        assert tree.getKind() == Kind.LESS_THAN;
    }

    @Override
    public <R, P> R accept(NodeVisitor<R, P> visitor, P p) {
        return visitor.visitLessThan(this, p);
    }

    @Override
    public String toString() {
        return "(" + getLeftOperand() + " < " + getRightOperand() + ")";
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == null || !(obj instanceof LessThanNode)) {
            return false;
        }
        LessThanNode other = (LessThanNode) obj;
        return getLeftOperand().equals(other.getLeftOperand())
                && getRightOperand().equals(other.getRightOperand());
    }

    @Override
    public int hashCode() {
        return HashCodeUtils.hash(getLeftOperand(), getRightOperand());
    }
}
