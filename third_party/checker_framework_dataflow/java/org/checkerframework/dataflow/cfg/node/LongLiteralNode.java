package org.checkerframework.dataflow.cfg.node;

import java.util.Collection;
import java.util.Collections;

import com.sun.source.tree.LiteralTree;
import com.sun.source.tree.Tree;

/**
 * A node for a long literal. For example:
 *
 * <pre>
 *   <em>-3l</em>
 *   <em>0x80808080L</em>
 * </pre>
 *
 * @author Stefan Heule
 * @author Charlie Garrett
 *
 */
public class LongLiteralNode extends ValueLiteralNode {

    public LongLiteralNode(LiteralTree t) {
        super(t);
        assert t.getKind().equals(Tree.Kind.LONG_LITERAL);
    }

    @Override
    public Long getValue() {
        return (Long) tree.getValue();
    }

    @Override
    public <R, P> R accept(NodeVisitor<R, P> visitor, P p) {
        return visitor.visitLongLiteral(this, p);
    }

    @Override
    public boolean equals(Object obj) {
        // test that obj is a LongLiteralNode
        if (obj == null || !(obj instanceof LongLiteralNode)) {
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
