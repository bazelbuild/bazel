package org.checkerframework.dataflow.cfg.node;

import java.util.Collection;
import java.util.LinkedList;

import javax.lang.model.type.TypeMirror;

import org.checkerframework.dataflow.util.HashCodeUtils;

import com.sun.source.tree.Tree;
import com.sun.source.tree.Tree.Kind;

/**
 * A node for the {@link AssertionError} when an assertion fails.
 *
 * <pre>
 *   assert <em>condition</em> : <em>detail</em> ;
 * </pre>
 *
 * @author Stefan Heule
 * @author Charlie Garrett
 *
 */
public class AssertionErrorNode extends Node {

    protected Tree tree;
    protected Node condition;
    protected Node detail;

    public AssertionErrorNode(Tree tree, Node condition, Node detail, TypeMirror type) {
        // TODO: Find out the correct "type" for statements.
        // Is it TypeKind.NONE?
        super(type);
        assert tree.getKind() == Kind.ASSERT;
        this.tree = tree;
        this.condition = condition;
        this.detail = detail;
    }

    public Node getCondition() {
        return condition;
    }

    public Node getDetail() {
        return detail;
    }

    @Override
    public Tree getTree() {
        return tree;
    }

    @Override
    public <R, P> R accept(NodeVisitor<R, P> visitor, P p) {
        return visitor.visitAssertionError(this, p);
    }

    @Override
    public String toString() {
        return "AssertionError(" + getDetail() + ")";
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == null || !(obj instanceof AssertionErrorNode)) {
            return false;
        }
        AssertionErrorNode other = (AssertionErrorNode) obj;
        return getCondition().equals(other.getCondition()) &&
            getDetail().equals(other.getDetail());
    }

    @Override
    public int hashCode() {
        return HashCodeUtils.hash(getCondition(), getDetail());
    }

    @Override
    public Collection<Node> getOperands() {
        LinkedList<Node> list = new LinkedList<Node>();
        list.add(getCondition());
        list.add(getDetail());
        return list;
    }
}
