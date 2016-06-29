package org.checkerframework.dataflow.cfg.block;

import java.util.Map;
import java.util.Set;

import javax.lang.model.type.TypeMirror;

import org.checkerframework.dataflow.cfg.node.Node;

/**
 * Represents a basic block that contains exactly one {@link Node} which can
 * throw an exception. This block has exactly one non-exceptional successor, and
 * one or more exceptional successors.
 *
 * <p>
 *
 * The following invariant holds.
 *
 * <pre>
 * getNode().getBlock() == this
 * </pre>
 *
 * @author Stefan Heule
 *
 */
public interface ExceptionBlock extends SingleSuccessorBlock {

    /**
     * @return the node of this block
     */
    Node getNode();

    /**
     * @return the list of exceptional successor blocks as an unmodifiable map
     */
    Map<TypeMirror, Set<Block>> getExceptionalSuccessors();

}
