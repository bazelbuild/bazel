package org.checkerframework.dataflow.analysis;

/*>>>
import org.checkerframework.checker.nullness.qual.Nullable;
*/

import org.checkerframework.dataflow.cfg.block.Block;
import org.checkerframework.dataflow.cfg.block.ExceptionBlock;
import org.checkerframework.dataflow.cfg.block.RegularBlock;
import org.checkerframework.dataflow.cfg.node.Node;

import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.Map;
import java.util.Map.Entry;

import javax.lang.model.element.Element;

import com.sun.source.tree.Tree;

/**
 * An {@link AnalysisResult} represents the result of a org.checkerframework.dataflow analysis by
 * providing the abstract values given a node or a tree. Note that it does not
 * keep track of custom results computed by some analysis.
 *
 * @author Stefan Heule
 *
 * @param <A>
 *            type of the abstract value that is tracked.
 */
public class AnalysisResult<A extends AbstractValue<A>, S extends Store<S>> {

    /** Abstract values of nodes. */
    protected final IdentityHashMap<Node, A> nodeValues;

    /** Map from AST {@link Tree}s to {@link Node}s. */
    protected final IdentityHashMap<Tree, Node> treeLookup;

    /** Map from (effectively final) local variable elements to their abstract value. */
    protected final HashMap<Element, A> finalLocalValues;

    /**
     * The stores before every method call.
     */
    protected final IdentityHashMap<Block, TransferInput<A, S>> stores;

    /**
     * Initialize with a given node-value mapping.
     */
    public AnalysisResult(Map<Node, A> nodeValues,
            IdentityHashMap<Block, TransferInput<A, S>> stores,
            IdentityHashMap<Tree, Node> treeLookup, HashMap<Element, A> finalLocalValues) {
        this.nodeValues = new IdentityHashMap<>(nodeValues);
        this.treeLookup = new IdentityHashMap<>(treeLookup);
        this.stores = stores;
        this.finalLocalValues = finalLocalValues;
    }

    /**
     * Initialize empty result.
     */
    public AnalysisResult() {
        nodeValues = new IdentityHashMap<>();
        treeLookup = new IdentityHashMap<>();
        stores = new IdentityHashMap<>();
        finalLocalValues = new HashMap<>();
    }

    /**
     * Combine with another analysis result.
     */
    public void combine(AnalysisResult<A, S> other) {
        for (Entry<Node, A> e : other.nodeValues.entrySet()) {
            nodeValues.put(e.getKey(), e.getValue());
        }
        for (Entry<Tree, Node> e : other.treeLookup.entrySet()) {
            treeLookup.put(e.getKey(), e.getValue());
        }
        for (Entry<Block, TransferInput<A, S>> e : other.stores.entrySet()) {
            stores.put(e.getKey(), e.getValue());
        }
        for (Entry<Element, A> e : other.finalLocalValues.entrySet()) {
            finalLocalValues.put(e.getKey(), e.getValue());
        }
    }

    /**
     * @return the value of effectively final local variables
     */
    public HashMap<Element, A> getFinalLocalValues() {
        return finalLocalValues;
    }

    /**
     * @return the abstract value for {@link Node} {@code n}, or {@code null} if
     *         no information is available.
     */
    public /*@Nullable*/ A getValue(Node n) {
        return nodeValues.get(n);
    }

    /**
     * @return the abstract value for {@link Tree} {@code t}, or {@code null} if
     *         no information is available.
     */
    public /*@Nullable*/ A getValue(Tree t) {
        A val = getValue(treeLookup.get(t));
        return val;
    }

    /**
     * @return the {@link Node} for a given {@link Tree}.
     */
    public /*@Nullable*/ Node getNodeForTree(Tree tree) {
        return treeLookup.get(tree);
    }

    /**
     * @return the store immediately before a given {@link Tree}.
     */
    public S getStoreBefore(Tree tree) {
        Node node = getNodeForTree(tree);
        if (node == null) {
            return null;
        }
        return getStoreBefore(node);
    }

    /**
     * @return the store immediately before a given {@link Node}.
     */
    public S getStoreBefore(Node node) {
        return runAnalysisFor(node, true);
    }

    /**
     * @return the store immediately after a given {@link Tree}.
     */
    public S getStoreAfter(Tree tree) {
        Node node = getNodeForTree(tree);
        if (node == null) {
            return null;
        }
        return runAnalysisFor(node, false);
    }

    /**
     * Runs the analysis again within the block of {@code node} and returns the
     * store at the location of {@code node}. If {@code before} is true, then
     * the store immediately before the {@link Node} {@code node} is returned.
     * Otherwise, the store after {@code node} is returned.
     *
     * <p>
     * If the given {@link Node} cannot be reached (in the control flow graph),
     * then {@code null} is returned.
     */
    protected S runAnalysisFor(Node node, boolean before) {
        Block block = node.getBlock();
        TransferInput<A, S> transferInput = stores.get(block);
        if (transferInput == null) {
            return null;
        }
        return runAnalysisFor(node, before, transferInput);
    }

    /**
     * Runs the analysis again within the block of {@code node} and returns the
     * store at the location of {@code node}. If {@code before} is true, then
     * the store immediately before the {@link Node} {@code node} is returned.
     * Otherwise, the store after {@code node} is returned.
     */
    public static <A extends AbstractValue<A>, S extends Store<S>> S runAnalysisFor(
            Node node, boolean before, TransferInput<A, S> transferInput) {
        assert node != null;
        Block block = node.getBlock();
        assert transferInput != null;
        Analysis<A, S, ?> analysis = transferInput.analysis;
        Node oldCurrentNode = analysis.currentNode;

        if (analysis.isRunning) {
            return analysis.currentInput.getRegularStore();
        }
        analysis.isRunning = true;
        try {
            switch (block.getType()) {
            case REGULAR_BLOCK: {
                RegularBlock rb = (RegularBlock) block;

                // Apply transfer function to contents until we found the node
                // we
                // are looking for.
                TransferInput<A, S> store = transferInput;
                TransferResult<A, S> transferResult = null;
                for (Node n : rb.getContents()) {
                    analysis.currentNode = n;
                    if (n == node && before) {
                        return store.getRegularStore();
                    }
                    transferResult = analysis.callTransferFunction(n, store);
                    if (n == node) {
                        return transferResult.getRegularStore();
                    }
                    store = new TransferInput<>(n, analysis, transferResult);
                }
                // This point should never be reached. If the block of 'node' is
                // 'block', then 'node' must be part of the contents of 'block'.
                assert false;
                return null;
            }

            case EXCEPTION_BLOCK: {
                ExceptionBlock eb = (ExceptionBlock) block;

                // apply transfer function to content
                assert eb.getNode() == node;
                if (before) {
                    return transferInput.getRegularStore();
                }
                analysis.currentNode = node;
                TransferResult<A, S> transferResult = analysis
                        .callTransferFunction(node, transferInput);
                return transferResult.getRegularStore();
            }

            default:
                // Only regular blocks and exceptional blocks can hold nodes.
                assert false;
                break;
            }

            return null;
        } finally {
            analysis.currentNode = oldCurrentNode;
            analysis.isRunning = false;
        }
    }
}
