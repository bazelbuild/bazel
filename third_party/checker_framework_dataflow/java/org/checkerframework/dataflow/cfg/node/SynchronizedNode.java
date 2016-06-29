package org.checkerframework.dataflow.cfg.node;

/*>>>
import org.checkerframework.checker.nullness.qual.Nullable;
*/

/*
 * This represents the start and end of synchronized code block.
 * If startOfBlock == true it is the node preceding a synchronized code block.
 * Otherwise it is the node immediately after a synchronized code block.
 */

import org.checkerframework.dataflow.util.HashCodeUtils;

import java.util.Collection;
import java.util.Collections;

import javax.lang.model.type.TypeKind;
import javax.lang.model.util.Types;

import com.sun.source.tree.Tree;

public class SynchronizedNode extends Node {

    protected /*@Nullable*/ Tree tree;
    protected Node expression;
    protected boolean startOfBlock;

    public SynchronizedNode(/*@Nullable*/ Tree tree, Node expression, boolean startOfBlock, Types types) {
        super(types.getNoType(TypeKind.NONE));
        this.tree = tree;
        this.expression = expression;
        this.startOfBlock = startOfBlock;
    }

    @Override
    public Tree getTree() {
        return tree;
    }

    public Node getExpression() {
        return expression;
    }

    public boolean getIsStartOfBlock() {
        return startOfBlock;
    }

    @Override
    public <R, P> R accept(NodeVisitor<R, P> visitor, P p) {
        return visitor.visitSynchronized(this, p);
    }

    @Override
    public String toString() {
        StringBuffer sb = new StringBuffer();
        sb.append("synchronized ");
        sb.append("(" + expression + ")");
        return sb.toString();
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == null || !(obj instanceof SynchronizedNode)) {
            return false;
        }
        SynchronizedNode other = (SynchronizedNode) obj;
        if (tree == null && other.getTree() != null) {
            return false;
        }

        return getTree().equals(other.getTree())
                && getExpression().equals(other.getExpression())
                && startOfBlock == other.startOfBlock;
    }

    @Override
    public int hashCode() {
        int hash = 0;
        if (tree != null) {
            hash = HashCodeUtils.hash(tree);
        }
        hash = HashCodeUtils.hash(startOfBlock);
        return HashCodeUtils.hash(hash, getExpression());
    }

    @Override
    public Collection<Node> getOperands() {
        return Collections.emptyList();
    }
}
