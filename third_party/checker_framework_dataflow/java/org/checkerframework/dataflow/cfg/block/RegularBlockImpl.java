package org.checkerframework.dataflow.cfg.block;

import java.util.Collections;
import java.util.LinkedList;
import java.util.List;

import org.checkerframework.dataflow.cfg.node.Node;

/**
 * Implementation of a regular basic block.
 *
 * @author Stefan Heule
 *
 */
public class RegularBlockImpl extends SingleSuccessorBlockImpl implements
        RegularBlock {

    /** Internal representation of the contents. */
    protected List<Node> contents;

    /**
     * Initialize an empty basic block to be filled with contents and linked to
     * other basic blocks later.
     */
    public RegularBlockImpl() {
        contents = new LinkedList<>();
        type = BlockType.REGULAR_BLOCK;
    }

    /**
     * Add a node to the contents of this basic block.
     */
    public void addNode(Node t) {
        contents.add(t);
        t.setBlock(this);
    }

    /**
     * Add multiple nodes to the contents of this basic block.
     */
    public void addNodes(List<? extends Node> ts) {
        for (Node t : ts) {
            addNode(t);
        }
    }

    @Override
    public List<Node> getContents() {
        return Collections.unmodifiableList(contents);
    }

    @Override
    public BlockImpl getRegularSuccessor() {
        return successor;
    }

    @Override
    public String toString() {
        return "RegularBlock(" + contents + ")";
    }

    @Override
    public boolean isEmpty() {
        return contents.isEmpty();
    }

}
