package org.checkerframework.dataflow.cfg.node;

import java.util.Collection;
import java.util.Collections;

import javax.lang.model.element.ExecutableElement;

import org.checkerframework.dataflow.util.HashCodeUtils;

import org.checkerframework.javacutil.InternalUtils;
import org.checkerframework.javacutil.TreeUtils;

import com.sun.source.tree.ExpressionTree;
import com.sun.source.tree.Tree;

/**
 * A node for a method access, including a method accesses:
 *
 * <pre>
 *   <em>expression</em> . <em>method</em> ()
 * </pre>
 *
 * @author Stefan Heule
 *
 */
public class MethodAccessNode extends Node {

    protected ExpressionTree tree;
    protected ExecutableElement method;
    protected Node receiver;

    // TODO: add method to get modifiers (static, access level, ..)

    public MethodAccessNode(ExpressionTree tree, Node receiver) {
        super(InternalUtils.typeOf(tree));
        assert TreeUtils.isMethodAccess(tree);
        this.tree = tree;
        this.method = (ExecutableElement) TreeUtils.elementFromUse(tree);
        this.receiver = receiver;
    }

    public ExecutableElement getMethod() {
        return method;
    }

    public Node getReceiver() {
        return receiver;
    }

    @Override
    public Tree getTree() {
        return tree;
    }

    @Override
    public <R, P> R accept(NodeVisitor<R, P> visitor, P p) {
        return visitor.visitMethodAccess(this, p);
    }

    @Override
    public String toString() {
        return getReceiver() + "." + method.getSimpleName();
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == null || !(obj instanceof MethodAccessNode)) {
            return false;
        }
        MethodAccessNode other = (MethodAccessNode) obj;
        return getReceiver().equals(other.getReceiver())
                && getMethod().equals(other.getMethod());
    }

    @Override
    public int hashCode() {
        return HashCodeUtils.hash(getReceiver(), getMethod());
    }

    @Override
    public Collection<Node> getOperands() {
        return Collections.singletonList(receiver);
    }
}
