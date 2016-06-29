package org.checkerframework.dataflow.cfg.node;

import java.util.Collection;
import java.util.Collections;

import com.sun.source.tree.LiteralTree;
import com.sun.source.tree.Tree;

/**
 * A node for an integer literal. For example:
 *
 * <pre>
 *   <em>42</em>
 * </pre>
 *
 * @author Stefan Heule
 *
 */
public class IntegerLiteralNode extends ValueLiteralNode {

    int value;

    public IntegerLiteralNode(LiteralTree t) {
        super(t);
        assert t.getKind().equals(Tree.Kind.INT_LITERAL);
        value = (Integer) tree.getValue();
    }

    @Override
    public Integer getValue() {
        return value;
    }

    @Override
    public <R, P> R accept(NodeVisitor<R, P> visitor, P p) {
        return visitor.visitIntegerLiteral(this, p);
    }

    @Override
    public boolean equals(Object obj) {
        // test that obj is a IntegerLiteralNode
        if (!(obj instanceof IntegerLiteralNode)) {
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
