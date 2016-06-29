package org.checkerframework.dataflow.cfg.node;

import java.util.Collection;
import java.util.Collections;

import com.sun.source.tree.LiteralTree;
import com.sun.source.tree.Tree;

/**
 * A node for a double literal. For example:
 *
 * <pre>
 *   <em>-9.</em>
 *   <em>3.14159D</em>
 * </pre>
 *
 * @author Stefan Heule
 * @author Charlie Garrett
 *
 */
public class DoubleLiteralNode extends ValueLiteralNode {

    public DoubleLiteralNode(LiteralTree t) {
        super(t);
        assert t.getKind().equals(Tree.Kind.DOUBLE_LITERAL);
    }

    @Override
    public Double getValue() {
        return (Double) tree.getValue();
    }

    @Override
    public <R, P> R accept(NodeVisitor<R, P> visitor, P p) {
        return visitor.visitDoubleLiteral(this, p);
    }

    @Override
    public boolean equals(Object obj) {
        // test that obj is a DoubleLiteralNode
        if (obj == null || !(obj instanceof DoubleLiteralNode)) {
            return false;
        }
        // super method compares values
        return super.equals(obj);
    }

    @Override
    public Collection<Node> getOperands() {
        return Collections.emptyList();
    }
}
