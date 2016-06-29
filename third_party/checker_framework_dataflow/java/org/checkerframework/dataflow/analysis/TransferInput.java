package org.checkerframework.dataflow.analysis;

/*>>>
import org.checkerframework.checker.nullness.qual.Nullable;
*/

import org.checkerframework.dataflow.cfg.node.Node;
import org.checkerframework.dataflow.util.HashCodeUtils;

/**
 * {@code TransferInput} is used as the input type of the individual transfer
 * functions of a {@link TransferFunction}. It also contains a reference to the
 * node for which the transfer function will be applied.
 *
 * <p>
 *
 * A {@code TransferInput} contains one or two stores. If two stores are
 * present, one belongs to 'then', and the other to 'else'.
 *
 * @author Stefan Heule
 *
 * @param <S>
 *            The {@link Store} used to keep track of intermediate results.
 */
public class TransferInput<A extends AbstractValue<A>, S extends Store<S>> {

    /**
     * The corresponding node.
     */
    protected Node node;

    /**
     * The regular result store (or {@code null} if none is present). The
     * following invariant is maintained:
     *
     * <pre>{@code
     * store == null &hArr; thenStore != null &amp;&amp; elseStore != null
     * }</pre>
     */
    protected final /*@Nullable*/ S store;

    /**
     * The 'then' result store (or {@code null} if none is present). The
     * following invariant is maintained:
     *
     * <pre>{@code
     * store == null &hArr; thenStore != null &amp;&amp; elseStore != null
     * }</pre>
     */
    protected final /*@Nullable*/ S thenStore;

    /**
     * The 'else' result store (or {@code null} if none is present). The
     * following invariant is maintained:
     *
     * <pre>{@code
     * store == null &hArr; thenStore != null &amp;&amp; elseStore != null
     * }</pre>
     */
    protected final /*@Nullable*/ S elseStore;

    /**
     * The corresponding analysis class to get intermediate flow results.
     */
    protected final Analysis<A, S, ?> analysis;

    /**
     * Create a {@link TransferInput}, given a {@link TransferResult} and a
     * node-value mapping.
     *
     * <p>
     *
     * <em>Aliasing</em>: The stores returned by any methods of {@code to} will
     * be stored internally and are not allowed to be used elsewhere. Full
     * control of them is transfered to this object.
     *
     * <p>
     *
     * The node-value mapping {@code nodeValues} is provided by the analysis and
     * is only read from within this {@link TransferInput}.
     */
    public TransferInput(Node n, Analysis<A, S, ?> analysis,
            TransferResult<A, S> to) {
        node = n;
        this.analysis = analysis;
        if (to.containsTwoStores()) {
            thenStore = to.getThenStore();
            elseStore = to.getElseStore();
            store = null;
        } else {
            store = to.getRegularStore();
            thenStore = elseStore = null;
        }
    }

    /**
     * Create a {@link TransferInput}, given a store and a node-value mapping.
     *
     * <p>
     *
     * <em>Aliasing</em>: The store {@code s} will be stored internally and is
     * not allowed to be used elsewhere. Full control over {@code s} is
     * transfered to this object.
     *
     * <p>
     *
     * The node-value mapping {@code nodeValues} is provided by the analysis and
     * is only read from within this {@link TransferInput}.
     */
    public TransferInput(Node n, Analysis<A, S, ?> analysis, S s) {
        node = n;
        this.analysis = analysis;
        store = s;
        thenStore = elseStore = null;
    }

    /**
     * Create a {@link TransferInput}, given two stores and a node-value
     * mapping.
     *
     * <p>
     *
     * <em>Aliasing</em>: The two stores {@code s1} and {@code s2} will be
     * stored internally and are not allowed to be used elsewhere. Full control
     * of them is transfered to this object.
     */
    public TransferInput(Node n, Analysis<A, S, ?> analysis, S s1, S s2) {
        node = n;
        this.analysis = analysis;
        thenStore = s1;
        elseStore = s2;
        store = null;
    }

    /**
     * Copy constructor.
     */
    protected TransferInput(TransferInput<A, S> from) {
        this.node = from.node;
        this.analysis = from.analysis;
        if (from.store == null) {
            thenStore = from.thenStore.copy();
            elseStore = from.elseStore.copy();
            store = null;
        } else {
            store = from.store.copy();
            thenStore = elseStore = null;
        }
    }

    /**
     * @return the {@link Node} for this {@link TransferInput}.
     */
    public Node getNode() {
        return node;
    }

    /**
     * @return the abstract value of {@link Node} {@code n}, which is required
     *         to be a 'sub-node' (that is, a direct or indirect child) of the
     *         node this transfer input is associated with. Furthermore,
     *         {@code n} cannot be a l-value node. Returns {@code null} if no
     *         value if available.
     */
    public /*@Nullable*/ A getValueOfSubNode(Node n) {
        return analysis.getValue(n);
    }

    /**
     * @return the regular result store produced if no exception is thrown by
     *         the {@link Node} corresponding to this transfer function result.
     */
    public S getRegularStore() {
        if (store == null) {
            return thenStore.leastUpperBound(elseStore);
        } else {
            return store;
        }
    }

    /**
     * @return the result store produced if the {@link Node} this result belongs
     *         to evaluates to {@code true}.
     */
    public S getThenStore() {
        if (store == null) {
            return thenStore;
        }
        return store;
    }

    /**
     * @return the result store produced if the {@link Node} this result belongs
     *         to evaluates to {@code false}.
     */
    public S getElseStore() {
        if (store == null) {
            return elseStore;
        }
        // copy the store such that it is the same as the result of getThenStore
        // (that is, identical according to equals), but two different objects.
        return store.copy();
    }

    /**
     * @return {@code true} if and only if this transfer input contains two
     *         stores that are potentially not equal. Note that the result
     *         {@code true} does not imply that {@code getRegularStore} cannot
     *         be called (or vice versa for {@code false}). Rather, it indicates
     *         that {@code getThenStore} or {@code getElseStore} can be used to
     *         give more precise results. Otherwise, if the result is
     *         {@code false}, then all three methods {@code getRegularStore},
     *         {@code getThenStore}, and {@code getElseStore} return equivalent
     *         stores.
     */
    public boolean containsTwoStores() {
        return (thenStore != null && elseStore != null);
    }

    /** @return an exact copy of this store. */
    public TransferInput<A, S> copy() {
        return new TransferInput<>(this);
    }

    /**
     * Compute the least upper bound of two stores.
     *
     * <p>
     *
     * <em>Important</em>: This method must fulfill the same contract as
     * {@code leastUpperBound} of {@link Store}.
     */
    public TransferInput<A, S> leastUpperBound(TransferInput<A, S> other) {
        if (store == null) {
            S newThenStore = thenStore.leastUpperBound(other.getThenStore());
            S newElseStore = elseStore.leastUpperBound(other.getElseStore());
            return new TransferInput<>(node, analysis, newThenStore,
                    newElseStore);
        } else {
            if (other.store == null) {
                // make sure we do not lose precision and keep two stores if at
                // least one of the two TransferInput's has two stores.
                return other.leastUpperBound(this);
            }
            return new TransferInput<>(node, analysis,
                    store.leastUpperBound(other.getRegularStore()));
        }
    }

    @Override
    public boolean equals(Object o) {
        if (o != null && o instanceof TransferInput) {
            @SuppressWarnings("unchecked")
            TransferInput<A, S> other = (TransferInput<A, S>) o;
            if (containsTwoStores()) {
                if (other.containsTwoStores()) {
                    return getThenStore().equals(other.getThenStore()) &&
                        getElseStore().equals(other.getElseStore());
                }
            } else {
                if (!other.containsTwoStores()) {
                    return getRegularStore().equals(other.getRegularStore());
                }
            }
        }
        return false;
    }

    @Override
    public int hashCode() {
        return HashCodeUtils.hash(this.analysis, this.node, this.store, this.thenStore, this.elseStore);
    }

    @Override
    public String toString() {
        if (store == null) {
            return "[then=" + thenStore + ", else=" + elseStore + "]";
        } else {
            return "[" + store + "]";
        }
    }

}
