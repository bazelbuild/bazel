package org.checkerframework.dataflow.cfg.node;

import com.sun.source.tree.Tree.Kind;
import com.sun.source.tree.UnaryTree;
import org.checkerframework.dataflow.util.HashCodeUtils;

/**
 * A node for the bitwise complement operation:
 *
 * <pre>
 *   ~ <em>expression</em>
 * </pre>
 *
 * @author Stefan Heule
 * @author Charlie Garrett
 */
public class BitwiseComplementNode extends UnaryOperationNode {

    public BitwiseComplementNode(UnaryTree tree, Node operand) {
        super(tree, operand);
        assert tree.getKind() == Kind.BITWISE_COMPLEMENT;
    }

    @Override
    public <R, P> R accept(NodeVisitor<R, P> visitor, P p) {
        return visitor.visitBitwiseComplement(this, p);
    }

    @Override
    public String toString() {
        return "(~ " + getOperand() + ")";
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == null || !(obj instanceof BitwiseComplementNode)) {
            return false;
        }
        BitwiseComplementNode other = (BitwiseComplementNode) obj;
        return getOperand().equals(other.getOperand());
    }

    @Override
    public int hashCode() {
        return HashCodeUtils.hash(getOperand());
    }
}
