package org.checkerframework.dataflow.analysis;

/**
 * An abstract value used in the org.checkerframework.dataflow analysis.
 *
 * @author Stefan Heule
 *
 */
public interface AbstractValue<V extends AbstractValue<V>> {

    /**
     * Compute the least upper bound of two stores.
     *
     * <p>
     *
     * <em>Important</em>: This method must fulfill the following contract:
     * <ul>
     * <li>Does not change {@code this}.</li>
     * <li>Does not change {@code other}.</li>
     * <li>Returns a fresh object which is not aliased yet.</li>
     * <li>Returns an object of the same (dynamic) type as {@code this}, even if
     * the signature is more permissive.</li>
     * <li>Is commutative.</li>
     * </ul>
     */
    V leastUpperBound(V other);
}
