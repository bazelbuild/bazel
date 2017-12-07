package org.checkerframework.dataflow.cfg.node;

import com.sun.source.tree.Tree.Kind;
import com.sun.source.tree.UnaryTree;
import org.checkerframework.dataflow.util.HashCodeUtils;

/**
 * A node for the unary plus operation:
 *
 * <pre>
 *   + <em>expression</em>
 * </pre>
 *
 * @author Stefan Heule
 * @author Charlie Garrett
 */
public class NumericalPlusNode extends UnaryOperationNode {

    public NumericalPlusNode(UnaryTree tree, Node operand) {
        super(tree, operand);
        assert tree.getKind() == Kind.UNARY_PLUS;
    }

    @Override
    public <R, P> R accept(NodeVisitor<R, P> visitor, P p) {
        return visitor.visitNumericalPlus(this, p);
    }

    @Override
    public String toString() {
        return "(+ " + getOperand() + ")";
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == null || !(obj instanceof NumericalPlusNode)) {
            return false;
        }
        NumericalPlusNode other = (NumericalPlusNode) obj;
        return getOperand().equals(other.getOperand());
    }

    @Override
    public int hashCode() {
        return HashCodeUtils.hash(getOperand());
    }
}
