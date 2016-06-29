package org.checkerframework.dataflow.cfg.block;

/*>>>
import org.checkerframework.checker.nullness.qual.Nullable;
*/

import org.checkerframework.dataflow.analysis.Store;

/**
 * A basic block that has at exactly one non-exceptional successor.
 *
 * @author Stefan Heule
 *
 */
public interface SingleSuccessorBlock extends Block {

    /**
     * @return the non-exceptional successor block, or {@code null} if there is
     *         no successor.
     */
    /*@Nullable*/ Block getSuccessor();

    /**
     * @return the flow rule for information flowing from this block to its successor
     */
    Store.FlowRule getFlowRule();

    /**
     * Set the flow rule for information flowing from this block to its successor.
     */
    void setFlowRule(Store.FlowRule rule);
}
