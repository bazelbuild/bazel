package org.checkerframework.dataflow.cfg.block;

import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import javax.lang.model.type.TypeMirror;

import org.checkerframework.dataflow.cfg.node.Node;

/**
 * Base class of the {@link Block} implementation hierarchy.
 *
 * @author Stefan Heule
 *
 */
public class ExceptionBlockImpl extends SingleSuccessorBlockImpl implements
        ExceptionBlock {

    /** Set of exceptional successors. */
    protected Map<TypeMirror, Set<Block>> exceptionalSuccessors;

    public ExceptionBlockImpl() {
        type = BlockType.EXCEPTION_BLOCK;
        exceptionalSuccessors = new HashMap<>();
    }

    /** The node of this block. */
    protected Node node;

    /**
     * Set the node.
     */
    public void setNode(Node c) {
        node = c;
        c.setBlock(this);
    }

    @Override
    public Node getNode() {
        return node;
    }

    /**
     * Add an exceptional successor.
     */
    public void addExceptionalSuccessor(BlockImpl b,
            TypeMirror cause) {
        if (exceptionalSuccessors == null) {
            exceptionalSuccessors = new HashMap<>();
        }
        Set<Block> blocks = exceptionalSuccessors.get(cause);
        if (blocks == null) {
            blocks = new HashSet<Block>();
            exceptionalSuccessors.put(cause, blocks);
        }
        blocks.add(b);
        b.addPredecessor(this);
    }

    @Override
    public Map<TypeMirror, Set<Block>> getExceptionalSuccessors() {
        if (exceptionalSuccessors == null) {
            return Collections.emptyMap();
        }
        return Collections.unmodifiableMap(exceptionalSuccessors);
    }

    @Override
    public String toString() {
        return "ExceptionBlock(" + node + ")";
    }

}
