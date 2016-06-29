package org.checkerframework.dataflow.cfg.node;

import java.util.Collection;
import java.util.Collections;

import javax.lang.model.type.TypeKind;
import javax.lang.model.util.Types;

import org.checkerframework.dataflow.util.HashCodeUtils;

import com.sun.source.tree.ThrowTree;
import com.sun.source.tree.Tree;

/**
 * A node for exception throws:
 *
 * <pre>
 *   <em>throw</em> expr
 * </pre>
 *
 * @author Stefan Heule
 * @author Charlie Garrett
 *
 */
public class ThrowNode extends Node {

    protected ThrowTree tree;
    protected Node expression;

    public ThrowNode(ThrowTree tree,
            Node expression, Types types) {
        super(types.getNoType(TypeKind.NONE));
        this.tree = tree;
        this.expression = expression;
    }

    public Node getExpression() {
        return expression;
    }

    @Override
    public Tree getTree() {
        return tree;
    }

    @Override
    public <R, P> R accept(NodeVisitor<R, P> visitor, P p) {
        return visitor.visitThrow(this, p);
    }

    @Override
    public String toString() {
        return "throw " + expression;
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == null || !(obj instanceof ThrowNode)) {
            return false;
        }
        ThrowNode other = (ThrowNode) obj;
        return getExpression().equals(other.getExpression());
    }

    @Override
    public int hashCode() {
        return HashCodeUtils.hash(expression);
    }

    @Override
    public Collection<Node> getOperands() {
        return Collections.singletonList(expression);
    }
}
