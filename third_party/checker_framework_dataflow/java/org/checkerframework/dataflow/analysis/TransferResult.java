package org.checkerframework.dataflow.analysis;

/*>>>
import org.checkerframework.checker.nullness.qual.Nullable;
*/

import java.util.Map;

import javax.lang.model.type.TypeMirror;

/**
 * {@code TransferResult} is used as the result type of the individual transfer
 * functions of a {@link TransferFunction}. It always belongs to the result of
 * the individual transfer function for a particular {@link org.checkerframework.dataflow.cfg.node.Node}, even though
 * that {@code org.checkerframework.dataflow.cfg.node.Node} is not explicitly store in {@code TransferResult}.
 *
 * <p>
 *
 * A {@code TransferResult} contains one or two stores (for 'then' and 'else'),
 * and zero or more stores with a cause ({@link TypeMirror}).
 *
 * @author Stefan Heule
 *
 * @param <S>
 *            The {@link Store} used to keep track of intermediate results.
 */
abstract public class TransferResult<A extends AbstractValue<A>, S extends Store<S>> {

    /**
     * The stores in case the basic block throws an exception (or {@code null}
     * if the corresponding {@link org.checkerframework.dataflow.cfg.node.Node} does not throw any exceptions). Does
     * not necessarily contain a store for every exception, in which case the
     * in-store will be used.
     */
    protected /*@Nullable*/ Map<TypeMirror, S> exceptionalStores;

    /**
     * The abstract value of the {@link org.checkerframework.dataflow.cfg.node.Node} associated with this
     * {@link TransferResult}, or {@code null} if no value has been produced.
     */
    protected /*@Nullable*/ A resultValue;

    public TransferResult(/*@Nullable*/ A resultValue) {
        this.resultValue = resultValue;
    }

    /**
     * @return the abstract value produced by the transfer function
     */
    public A getResultValue() {
        return resultValue;
    }

    public void setResultValue(A resultValue) {
        this.resultValue = resultValue;
    }

    /**
     * @return the regular result store produced if no exception is thrown by
     *         the {@link org.checkerframework.dataflow.cfg.node.Node} corresponding to this transfer function result.
     */
    abstract public S getRegularStore();

    /**
     * @return the result store produced if the {@link org.checkerframework.dataflow.cfg.node.Node} this result belongs
     *         to evaluates to {@code true}.
     */
    abstract public S getThenStore();

    /**
     * @return the result store produced if the {@link org.checkerframework.dataflow.cfg.node.Node} this result belongs
     *         to evaluates to {@code false}.
     */
    abstract public S getElseStore();

    /**
     * @return the store that flows along the outgoing exceptional edge labeled
     *         with {@code exception} (or {@code null} if no special handling is
     *         required for exceptional edges).
     */
    public /*@Nullable*/ S getExceptionalStore(
            TypeMirror exception) {
        if (exceptionalStores == null) {
            return null;
        }
        return exceptionalStores.get(exception);
    }

    /**
     * @return a Map of {@link TypeMirror} to {@link Store}.
     *
     * @see TransferResult#getExceptionalStore(TypeMirror)
     */
    public Map<TypeMirror, S> getExceptionalStores() {
        return exceptionalStores;
    }

    /**
     * @return {@code true} if and only if this transfer result contains two
     *         stores that are potentially not equal. Note that the result
     *         {@code true} does not imply that {@code getRegularStore} cannot
     *         be called (or vice versa for {@code false}). Rather, it indicates
     *         that {@code getThenStore} or {@code getElseStore} can be used to
     *         give more precise results. Otherwise, if the result is
     *         {@code false}, then all three methods {@code getRegularStore},
     *         {@code getThenStore}, and {@code getElseStore} return equivalent
     *         stores.
     */
    abstract public boolean containsTwoStores();

    /**
     * @return {@code true} if and only if the transfer function returning this
     *         transfer result changed the regularStore, elseStore, or thenStore.
     */
    abstract public boolean storeChanged();
}
