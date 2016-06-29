package org.checkerframework.dataflow.cfg.node;

import java.util.Collection;
import java.util.Collections;

import javax.lang.model.type.TypeMirror;

import org.checkerframework.dataflow.util.HashCodeUtils;

import org.checkerframework.javacutil.TypesUtils;

import com.sun.source.tree.Tree;

/**
 * A node for the narrowing primitive conversion operation. See JLS 5.1.3 for
 * the definition of narrowing primitive conversion.
 *
 * A {@link NarrowingConversionNode} does not correspond to any tree node in the
 * parsed AST. It is introduced when a value of some primitive type appears in a
 * context that requires a different primitive with more bits of precision.
 *
 * @author Stefan Heule
 * @author Charlie Garrett
 *
 */
public class NarrowingConversionNode extends Node {

    protected Tree tree;
    protected Node operand;

    public NarrowingConversionNode(Tree tree, Node operand, TypeMirror type) {
        super(type);
        assert TypesUtils.isPrimitive(type) : "non-primitive type in narrowing conversion";
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
        return visitor.visitNarrowingConversion(this, p);
    }

    @Override
    public String toString() {
        return "NarrowingConversion(" + getOperand() + ", " + type + ")";
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == null || !(obj instanceof NarrowingConversionNode)) {
            return false;
        }
        NarrowingConversionNode other = (NarrowingConversionNode) obj;
        return getOperand().equals(other.getOperand())
                && TypesUtils.areSamePrimitiveTypes(getType(), other.getType());
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
