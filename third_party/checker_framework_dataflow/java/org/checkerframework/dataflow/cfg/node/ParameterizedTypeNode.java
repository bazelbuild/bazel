package org.checkerframework.dataflow.cfg.node;

import java.util.Collection;
import java.util.Collections;

import org.checkerframework.dataflow.util.HashCodeUtils;

import org.checkerframework.javacutil.InternalUtils;

import com.sun.source.tree.ParameterizedTypeTree;
import com.sun.source.tree.Tree;

/**
 * A node for a parameterized type occurring in an expression:
 *
 * <pre>
 *   <em>type&lt;arg1, arg2&gt;</em>
 * </pre>
 *
 * Parameterized types don't represent any computation to be done
 * at runtime, so we might choose to represent them differently by
 * modifying the {@link Node}s in which parameterized types can occur, such
 * as {@link ObjectCreationNode}s.
 *
 * @author Stefan Heule
 * @author Charlie Garrett
 *
 */

public class ParameterizedTypeNode extends Node {

    protected Tree tree;

    public ParameterizedTypeNode(Tree t) {
        super(InternalUtils.typeOf(t));
        assert t instanceof ParameterizedTypeTree;
        tree = t;
    }

    @Override
    public Tree getTree() {
        return tree;
    }

    @Override
    public <R, P> R accept(NodeVisitor<R, P> visitor, P p) {
        return visitor.visitParameterizedType(this, p);
    }

    @Override
    public String toString() {
        return getTree().toString();
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == null || !(obj instanceof ParameterizedTypeNode)) {
            return false;
        }
        ParameterizedTypeNode other = (ParameterizedTypeNode) obj;
        return getTree().equals(other.getTree());
    }

    @Override
    public int hashCode() {
        return HashCodeUtils.hash(getTree());
    }

    @Override
    public Collection<Node> getOperands() {
        return Collections.emptyList();
    }
}
