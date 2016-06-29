package org.checkerframework.dataflow.cfg.node;

import java.util.Collection;
import java.util.Collections;

import com.sun.source.tree.LiteralTree;
import com.sun.source.tree.Tree;

/**
 * A node for an string literal. For example:
 *
 * <pre>
 *   <em>"abc"</em>
 * </pre>
 *
 * @author Stefan Heule
 *
 */
public class StringLiteralNode extends ValueLiteralNode {

    public StringLiteralNode(LiteralTree t) {
        super(t);
        assert t.getKind().equals(Tree.Kind.STRING_LITERAL);
    }

    @Override
    public String getValue() {
        return (String) tree.getValue();
    }

    @Override
    public <R, P> R accept(NodeVisitor<R, P> visitor, P p) {
        return visitor.visitStringLiteral(this, p);
    }

    @Override
    public boolean equals(Object obj) {
        // test that obj is a StringLiteralNode
        if (!(obj instanceof StringLiteralNode)) {
            return false;
        }
        // super method compares values
        return super.equals(obj);
    }

    @Override
    public Collection<Node> getOperands() {
        return Collections.emptyList();
    }

    @Override
    public String toString() {
        return "\"" + super.toString() + "\"";
    }
}
