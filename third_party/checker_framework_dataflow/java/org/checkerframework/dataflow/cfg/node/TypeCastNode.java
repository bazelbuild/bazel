package org.checkerframework.dataflow.cfg.node;

import java.util.Collection;
import java.util.Collections;

import javax.lang.model.type.TypeMirror;

import org.checkerframework.dataflow.util.HashCodeUtils;

import com.sun.source.tree.Tree;

/**
 * A node for the cast operator:
 *
 * (<em>Point</em>) <em>x</em>
 *
 * @author Stefan Heule
 * @author Charlie Garrett
 *
 */
public class TypeCastNode extends Node {

    protected Tree tree;
    protected Node operand;

    public TypeCastNode(Tree tree, Node operand, TypeMirror type) {
        super(type);
        this.tree = tree;
        this.operand = operand;
    }

    public Node getOperand() {
        return operand;
    }

    public TypeMirror getType() {
        return type;
    }

    @Override
    public Tree getTree() {
        return tree;
    }

    @Override
    public <R, P> R accept(NodeVisitor<R, P> visitor, P p) {
        return visitor.visitTypeCast(this, p);
    }

    @Override
    public String toString() {
        return "(" + getType() + ")" + getOperand();
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == null || !(obj instanceof TypeCastNode)) {
            return false;
        }
        TypeCastNode other = (TypeCastNode) obj;
        // TODO: TypeMirror.equals may be too restrictive.
        // Check whether Types.isSameType is the better comparison.
        return getOperand().equals(other.getOperand())
                && getType().equals(other.getType());
    }

    @Override
    public int hashCode() {
        return HashCodeUtils.hash(getOperand());
    }

    @Override
    public Collection<Node> getOperands() {
        return Collections.singletonList(getOperand());
    }
}
