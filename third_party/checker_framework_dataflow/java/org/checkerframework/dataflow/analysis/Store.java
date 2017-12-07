package org.checkerframework.dataflow.analysis;

import org.checkerframework.dataflow.cfg.CFGVisualizer;

/**
 * A store is used to keep track of the information that the org.checkerframework.dataflow analysis
 * has accumulated at any given point in time.
 *
 * @author Stefan Heule
 * @param <S> the type of the store returned by {@code copy} and that is used in {@code
 *     leastUpperBound}. Usually it is the implementing class itself, e.g. in {@code T extends
 *     Store<T>}.
 */
public interface Store<S extends Store<S>> {

    // We maintain a then store and an else store before each basic block.
    // When they are identical (by reference equality), they can be treated
    // as a regular unconditional store.
    // Once we have some information for both the then and else store, we
    // create a TransferInput for the block and allow it to be analyzed.
    public static enum Kind {
        THEN,
        ELSE,
        BOTH
    }

    /** A flow rule describes how stores flow along one edge between basic blocks. */
    public static enum FlowRule {
        EACH_TO_EACH, // The normal case, then store flows to the then store
        // and else store flows to the else store.
        THEN_TO_BOTH, // Then store flows to both then and else of successor.
        ELSE_TO_BOTH, // Else store flows to both then and else of successor.
        THEN_TO_THEN, // Then store flows to the then of successor.  Else store is ignored.
        ELSE_TO_ELSE, // Else store flows to the else of successor.  Then store is ignored.
    }

    /** @return an exact copy of this store. */
    S copy();

    /**
     * Compute the least upper bound of two stores.
     *
     * <p><em>Important</em>: This method must fulfill the following contract:
     *
     * <ul>
     *   <li>Does not change {@code this}.
     *   <li>Does not change {@code other}.
     *   <li>Returns a fresh object which is not aliased yet.
     *   <li>Returns an object of the same (dynamic) type as {@code this}, even if the signature is
     *       more permissive.
     *   <li>Is commutative.
     * </ul>
     */
    S leastUpperBound(S other);

    /**
     * Compute an upper bound of two stores that is wider than the least upper bound of the two
     * stores. Used to jump to a higher abstraction to allow faster termination of the fixed point
     * computations in {@link Analysis}. {@code previous} must be the previous store.
     *
     * <p>A particular analysis might not require widening and should implement this method by
     * calling leastUpperBound.
     *
     * <p><em>Important</em>: This method must fulfill the following contract:
     *
     * <ul>
     *   <li>Does not change {@code this}.
     *   <li>Does not change {@code previous}.
     *   <li>Returns a fresh object which is not aliased yet.
     *   <li>Returns an object of the same (dynamic) type as {@code this}, even if the signature is
     *       more permissive.
     *   <li>Is commutative.
     * </ul>
     *
     * @param previous must be the previous store
     */
    S widenedUpperBound(S previous);

    /**
     * Can the objects {@code a} and {@code b} be aliases? Returns a conservative answer (i.e.,
     * returns {@code true} if not enough information is available to determine aliasing).
     */
    boolean canAlias(FlowExpressions.Receiver a, FlowExpressions.Receiver b);

    /**
     * Delegate visualization responsibility to a visualizer.
     *
     * @param viz the visualizer to visualize this store
     */
    void visualize(CFGVisualizer<?, S, ?> viz);
}
