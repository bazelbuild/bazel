package org.checkerframework.dataflow.cfg.node;

import java.util.Collection;
import java.util.Collections;

import com.sun.source.tree.LiteralTree;
import com.sun.source.tree.Tree;

/**
 * A node for a character literal. For example:
 *
 * <pre>
 *   <em>'a'</em>
 *   <em>'\t'</em>
 *   <em>'\u03a9'</em>
 * </pre>
 *
 * @author Stefan Heule
 * @author Charlie Garrett
 *
 */
public class CharacterLiteralNode extends ValueLiteralNode {

    public CharacterLiteralNode(LiteralTree t) {
        super(t);
        assert t.getKind().equals(Tree.Kind.CHAR_LITERAL);
    }

    @Override
    public Character getValue() {
        return (Character) tree.getValue();
    }

    @Override
    public <R, P> R accept(NodeVisitor<R, P> visitor, P p) {
        return visitor.visitCharacterLiteral(this, p);
    }

    @Override
    public boolean equals(Object obj) {
        // test that obj is a CharacterLiteralNode
        if (obj == null || !(obj instanceof CharacterLiteralNode)) {
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
