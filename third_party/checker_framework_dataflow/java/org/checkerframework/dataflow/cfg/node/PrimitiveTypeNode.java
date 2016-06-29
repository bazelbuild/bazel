package org.checkerframework.dataflow.cfg.node;

import java.util.Collection;
import java.util.Collections;

import org.checkerframework.dataflow.util.HashCodeUtils;

import org.checkerframework.javacutil.InternalUtils;

import com.sun.source.tree.PrimitiveTypeTree;
import com.sun.source.tree.Tree;

/**
 * A node representing a primitive type used in an expression
 * such as a field access
 *
 * <em>type</em> .class
 *
 * @author Stefan Heule
 * @author Charlie Garrett
 *
 */
public class PrimitiveTypeNode extends Node {

    protected final PrimitiveTypeTree tree;

    public PrimitiveTypeNode(PrimitiveTypeTree tree) {
        super(InternalUtils.typeOf(tree));
        this.tree = tree;
    }

    @Override
    public Tree getTree() {
        return tree;
    }

    @Override
    public <R, P> R accept(NodeVisitor<R, P> visitor, P p) {
        return visitor.visitPrimitiveType(this, p);
    }

    @Override
    public String toString() {
        return tree.toString();
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == null || !(obj instanceof PrimitiveTypeNode)) {
            return false;
        }
        PrimitiveTypeNode other = (PrimitiveTypeNode) obj;
        return getType().equals(other.getType());
    }

    @Override
    public int hashCode() {
        return HashCodeUtils.hash(getType());
    }

    @Override
    public Collection<Node> getOperands() {
        return Collections.emptyList();
    }
}
