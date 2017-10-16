package org.checkerframework.dataflow.cfg.node;

import com.sun.source.tree.Tree.Kind;
import com.sun.source.tree.UnaryTree;
import org.checkerframework.dataflow.util.HashCodeUtils;

/**
 * A node for the unary minus operation:
 *
 * <pre>
 *   - <em>expression</em>
 * </pre>
 *
 * @author Stefan Heule
 * @author Charlie Garrett
 */
public class NumericalMinusNode extends UnaryOperationNode {

    public NumericalMinusNode(UnaryTree tree, Node operand) {
        super(tree, operand);
        assert tree.getKind() == Kind.UNARY_MINUS;
    }

    @Override
    public <R, P> R accept(NodeVisitor<R, P> visitor, P p) {
        return visitor.visitNumericalMinus(this, p);
    }

    @Override
    public String toString() {
        return "(- " + getOperand() + ")";
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == null || !(obj instanceof NumericalMinusNode)) {
            return false;
        }
        NumericalMinusNode other = (NumericalMinusNode) obj;
        return getOperand().equals(other.getOperand());
    }

    @Override
    public int hashCode() {
        return HashCodeUtils.hash(getOperand());
    }
}
