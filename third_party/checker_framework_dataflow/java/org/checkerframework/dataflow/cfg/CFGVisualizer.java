package org.checkerframework.dataflow.cfg;

/*>>>
import org.checkerframework.checker.nullness.qual.Nullable;
*/

import org.checkerframework.dataflow.analysis.AbstractValue;
import org.checkerframework.dataflow.analysis.Analysis;
import org.checkerframework.dataflow.analysis.FlowExpressions;
import org.checkerframework.dataflow.analysis.Store;
import org.checkerframework.dataflow.analysis.TransferFunction;
import org.checkerframework.dataflow.cfg.block.Block;
import org.checkerframework.dataflow.cfg.block.SpecialBlock;
import org.checkerframework.dataflow.cfg.node.Node;

import java.util.Map;

/**
 * Perform some visualization on a control flow graph.
 * The particular operations depend on the implementation.
 */
public interface CFGVisualizer<A extends AbstractValue<A>,
        S extends Store<S>, T extends TransferFunction<A, S>> {
    /**
     * Initialization method guaranteed to be called once before the
     * first invocation of {@link visualize}.
     *
     * @param args implementation-dependent options
     */
    void init(Map<String, Object> args);

    /**
     * Output a visualization representing the control flow graph starting
     * at {@code entry}.
     * The concrete actions are implementation dependent.
     *
     * An invocation {@code visualize(cfg, entry, null);} does not
     * output stores at the beginning of basic blocks.
     *
     * @param cfg
     *            The CFG to visualize.
     * @param entry
     *            The entry node of the control flow graph to be represented.
     * @param analysis
     *            An analysis containing information about the program
     *            represented by the CFG. The information includes {@link Store}s
     *            that are valid at the beginning of basic blocks reachable
     *            from {@code entry} and per-node information for value
     *            producing {@link Node}s. Can also be {@code null} to
     *            indicate that this information should not be output.
     * @return possible analysis results, e.g. generated file names.
     */
    /*@Nullable*/ Map<String, Object> visualize(ControlFlowGraph cfg, Block entry,
            /*@Nullable*/ Analysis<A, S, T> analysis);

    /**
     * Delegate the visualization responsibility
     * to the passed {@link Store} instance, which will call back to this
     * visualizer instance for sub-components.
     *
     * @param store the store to visualize
     */
    void visualizeStore(S store);

    /**
     * Called by a {@code CFAbstractStore} to visualize
     * the class name before calling the
     * {@code CFAbstractStore#internalVisualize()} method.
     *
     * @param classCanonicalName the canonical name of the class
     */
    void visualizeStoreHeader(String classCanonicalName);

    /**
     * Called by {@code CFAbstractStore#internalVisualize()} to visualize
     * a local variable.
     *
     * @param localVar the local variable
     * @param value the value of the local variable
     */
    void visualizeStoreLocalVar(FlowExpressions.LocalVariable localVar, A value);

    /**
     * Called by {@code CFAbstractStore#internalVisualize()} to visualize
     * the value of the current object {@code this} in this Store.
     *
     * @param value the value of the current object this
     */
    void visualizeStoreThisVal(A value);

    /**
     * Called by {@code CFAbstractStore#internalVisualize()} to visualize
     * the value of fields collected by this Store.
     *
     * @param fieldAccess the field
     * @param value the value of the field
     */
    void visualizeStoreFieldVals(FlowExpressions.FieldAccess fieldAccess, A value);

    /**
     * Called by {@code CFAbstractStore#internalVisualize()} to visualize
     * the value of arrays collected by this Store.
     *
     * @param arrayValue the array
     * @param value the value of the array
     */
    void visualizeStoreArrayVal(FlowExpressions.ArrayAccess arrayValue, A value);

    /**
     * Called by {@code CFAbstractStore#internalVisualize()} to visualize
     * the value of pure method calls collected by this Store.
     *
     * @param methodCall the pure method call
     * @param value the value of the pure method call
     */
    void visualizeStoreMethodVals(FlowExpressions.MethodCall methodCall, A value);

    /**
     * Called by {@code CFAbstractStore#internalVisualize()} to visualize
     * the value of class names collected by this Store.
     *
     * @param className the class name
     * @param value the value of the class name
     */
    void visualizeStoreClassVals(FlowExpressions.ClassName className, A value);

    /**
     * Called by {@code CFAbstractStore#internalVisualize()} to visualize
     * the specific information collected according to the specific kind of Store.
     * Currently, these Stores call this method: {@code LockStore},
     * {@code NullnessStore}, and {@code InitializationStore} to visualize additional
     * information.
     *
     * @param keyName the name of the specific information to be visualized
     * @param value the value of the specific information to be visualized
     */
    void visualizeStoreKeyVal(String keyName, Object value);

    /**
     * Called by {@code CFAbstractStore} to visualize
     * any information after the invocation of {@code CFAbstractStore#internalVisualize()}.
     */
    void visualizeStoreFooter();

    /**
     * Visualize a block based on the analysis.
     *
     * @param bb the block
     * @param analysis the current analysis
     */
    void visualizeBlock(Block bb, /*@Nullable*/ Analysis<A, S, T> analysis);

    /**
     * Visualize a SpecialBlock.
     *
     * @param sbb the special block
     */
    void visualizeSpecialBlock(SpecialBlock sbb);

    /**
     * Visualize the transferInput of a Block based on the analysis.
     *
     * @param bb the block
     * @param analysis the current analysis
     */
    void visualizeBlockTransferInput(Block bb, Analysis<A, S, T> analysis);

    /**
     * Visualize a Node based on the analysis.
     *
     * @param t the node
     * @param analysis the current analysis
     */
    void visualizeBlockNode(Node t, /*@Nullable*/ Analysis<A, S, T> analysis);

    /**
     * Shutdown method called once from the shutdown hook of the
     * {@code BaseTypeChecker}.
     */
    void shutdown();
}
