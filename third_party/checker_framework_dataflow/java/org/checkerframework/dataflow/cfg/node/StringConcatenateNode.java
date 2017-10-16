package org.checkerframework.dataflow.cfg.node;

import com.sun.source.tree.BinaryTree;
import com.sun.source.tree.Tree.Kind;
import org.checkerframework.dataflow.util.HashCodeUtils;

/**
 * A node for string concatenation:
 *
 * <pre>
 *   <em>expression</em> + <em>expression</em>
 * </pre>
 *
 * @author Stefan Heule
 * @author Charlie Garrett
 */
public class StringConcatenateNode extends BinaryOperationNode {

    public StringConcatenateNode(BinaryTree tree, Node left, Node right) {
        super(tree, left, right);
        assert tree.getKind() == Kind.PLUS;
    }

    @Override
    public <R, P> R accept(NodeVisitor<R, P> visitor, P p) {
        return visitor.visitStringConcatenate(this, p);
    }

    @Override
    public String toString() {
        return "(" + getLeftOperand() + " + " + getRightOperand() + ")";
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == null || !(obj instanceof StringConcatenateNode)) {
            return false;
        }
        StringConcatenateNode other = (StringConcatenateNode) obj;
        return getLeftOperand().equals(other.getLeftOperand())
                && getRightOperand().equals(other.getRightOperand());
    }

    @Override
    public int hashCode() {
        return HashCodeUtils.hash(getLeftOperand(), getRightOperand());
    }
}
