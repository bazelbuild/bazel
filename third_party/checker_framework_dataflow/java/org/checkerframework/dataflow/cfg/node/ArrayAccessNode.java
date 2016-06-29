package org.checkerframework.dataflow.cfg.node;

import java.util.Collection;
import java.util.LinkedList;

import org.checkerframework.dataflow.util.HashCodeUtils;

import org.checkerframework.javacutil.InternalUtils;

import com.sun.source.tree.ArrayAccessTree;
import com.sun.source.tree.Tree;

/**
 * A node for an array access:
 *
 * <pre>
 *   <em>array ref</em> [ <em>index</em> ]
 * </pre>
 *
 * We allow array accesses without corresponding AST {@link Tree}s.
 *
 * @author Stefan Heule
 * @author Charlie Garrett
 *
 */

public class ArrayAccessNode extends Node {

    protected Tree tree;
    protected Node array;
    protected Node index;

    public ArrayAccessNode(Tree t, Node array, Node index) {
        super(InternalUtils.typeOf(t));
        assert t instanceof ArrayAccessTree;
        this.tree = t;
        this.array = array;
        this.index = index;
    }

    public Node getArray() {
        return array;
    }

    public Node getIndex() {
        return index;
    }

    @Override
    public Tree getTree() {
        return tree;
    }

    @Override
    public <R, P> R accept(NodeVisitor<R, P> visitor, P p) {
        return visitor.visitArrayAccess(this, p);
    }

    @Override
    public String toString() {
        String base = getArray().toString() + "[" + getIndex().toString() + "]";
        return base;
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == null || !(obj instanceof ArrayAccessNode)) {
            return false;
        }
        ArrayAccessNode other = (ArrayAccessNode) obj;
        return getArray().equals(other.getArray())
                && getIndex().equals(other.getIndex());
    }

    @Override
    public int hashCode() {
        return HashCodeUtils.hash(getArray(), getIndex());
    }

    @Override
    public Collection<Node> getOperands() {
        LinkedList<Node> list = new LinkedList<Node>();
        list.add(getArray());
        list.add(getIndex());
        return list;
    }
}
