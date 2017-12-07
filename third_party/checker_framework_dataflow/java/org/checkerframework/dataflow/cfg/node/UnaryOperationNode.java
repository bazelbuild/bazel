package org.checkerframework.dataflow.cfg.node;

import com.sun.source.tree.UnaryTree;
import java.util.Collection;
import java.util.Collections;
import org.checkerframework.javacutil.InternalUtils;

/**
 * A node for a postfix or an unary expression.
 *
 * <p>For example:
 *
 * <pre>
 *   <em>operator</em> <em>expressionNode</em>
 *
 *   <em>expressionNode</em> <em>operator</em>
 * </pre>
 *
 * @author charleszhuochen
 */
public abstract class UnaryOperationNode extends Node {

    protected final UnaryTree tree;
    protected final Node operand;

    public UnaryOperationNode(UnaryTree tree, Node operand) {
        super(InternalUtils.typeOf(tree));
        this.tree = tree;
        this.operand = operand;
    }

    public Node getOperand() {
        return this.operand;
    }

    @Override
    public UnaryTree getTree() {
        return tree;
    }

    @Override
    public Collection<Node> getOperands() {
        return Collections.singletonList(getOperand());
    }
}
