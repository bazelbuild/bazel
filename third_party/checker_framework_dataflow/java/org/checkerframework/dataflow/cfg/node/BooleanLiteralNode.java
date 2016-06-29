package org.checkerframework.dataflow.cfg.node;

import java.util.Collection;
import java.util.Collections;

import com.sun.source.tree.LiteralTree;
import com.sun.source.tree.Tree;

/**
 * A node for a boolean literal:
 *
 * <pre>
 *   <em>true</em>
 *   <em>false</em>
 * </pre>
 *
 * @author Stefan Heule
 *
 */
public class BooleanLiteralNode extends ValueLiteralNode {

    public BooleanLiteralNode(LiteralTree t) {
        super(t);
        assert t.getKind().equals(Tree.Kind.BOOLEAN_LITERAL);
    }

    @Override
    public Boolean getValue() {
        return (Boolean) tree.getValue();
    }

    @Override
    public <R, P> R accept(NodeVisitor<R, P> visitor, P p) {
        return visitor.visitBooleanLiteral(this, p);
    }

    @Override
    public boolean equals(Object obj) {
        // test that obj is a BooleanLiteralNode
        if (!(obj instanceof BooleanLiteralNode)) {
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
