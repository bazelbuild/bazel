package org.checkerframework.dataflow.cfg.node;

import java.util.Collection;
import java.util.LinkedList;

import org.checkerframework.javacutil.InternalUtils;

import com.sun.source.tree.Tree;
import com.sun.source.tree.Tree.Kind;

/**
 * A node for the string concatenation compound assignment:
 *
 * <pre>
 *   <em>variable</em> += <em>expression</em>
 * </pre>
 *
 * @author Stefan Heule
 * @author Charlie Garrett
 *
 */
public class StringConcatenateAssignmentNode extends Node {
    protected Tree tree;
    protected Node left;
    protected Node right;

    public StringConcatenateAssignmentNode(Tree tree, Node left, Node right) {
        super(InternalUtils.typeOf(tree));
        assert tree.getKind() == Kind.PLUS_ASSIGNMENT;
        this.tree = tree;
        this.left = left;
        this.right = right;
    }

    public Node getLeftOperand() {
        return left;
    }

    public Node getRightOperand() {
        return right;
    }

    @Override
    public Tree getTree() {
        return tree;
    }

    @Override
    public <R, P> R accept(NodeVisitor<R, P> visitor, P p) {
        return visitor.visitStringConcatenateAssignment(this, p);
    }

    @Override
    public Collection<Node> getOperands() {
        LinkedList<Node> list = new LinkedList<Node>();
        list.add(getLeftOperand());
        list.add(getRightOperand());
        return list;
    }
}
