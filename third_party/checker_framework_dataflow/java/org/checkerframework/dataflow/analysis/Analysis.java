package org.checkerframework.dataflow.analysis;

/*>>>
import org.checkerframework.checker.nullness.qual.Nullable;
*/

import com.sun.source.tree.LambdaExpressionTree;
import org.checkerframework.dataflow.cfg.ControlFlowGraph;
import org.checkerframework.dataflow.cfg.UnderlyingAST;
import org.checkerframework.dataflow.cfg.UnderlyingAST.CFGLambda;
import org.checkerframework.dataflow.cfg.UnderlyingAST.CFGMethod;
import org.checkerframework.dataflow.cfg.UnderlyingAST.Kind;
import org.checkerframework.dataflow.cfg.block.Block;
import org.checkerframework.dataflow.cfg.block.ConditionalBlock;
import org.checkerframework.dataflow.cfg.block.ExceptionBlock;
import org.checkerframework.dataflow.cfg.block.RegularBlock;
import org.checkerframework.dataflow.cfg.block.SpecialBlock;
import org.checkerframework.dataflow.cfg.node.AssignmentNode;
import org.checkerframework.dataflow.cfg.node.LocalVariableNode;
import org.checkerframework.dataflow.cfg.node.Node;
import org.checkerframework.dataflow.cfg.node.ReturnNode;

import org.checkerframework.javacutil.ElementUtils;
import org.checkerframework.javacutil.Pair;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Objects;
import java.util.PriorityQueue;
import java.util.Set;

import javax.annotation.processing.ProcessingEnvironment;
import javax.lang.model.element.Element;
import javax.lang.model.type.TypeMirror;
import javax.lang.model.util.Types;

import com.sun.source.tree.ClassTree;
import com.sun.source.tree.MethodTree;
import com.sun.source.tree.Tree;
import com.sun.source.tree.VariableTree;

/**
 * An implementation of an iterative algorithm to solve a org.checkerframework.dataflow problem,
 * given a control flow graph and a transfer function.
 *
 * @author Stefan Heule
 *
 * @param <A>
 *            The abstract value type to be tracked by the analysis.
 * @param <S>
 *            The store type used in the analysis.
 * @param <T>
 *            The transfer function type that is used to approximated runtime
 *            behavior.
 */
public class Analysis<A extends AbstractValue<A>, S extends Store<S>, T extends TransferFunction<A, S>> {

    /** Is the analysis currently running? */
    protected boolean isRunning = false;

    /** The transfer function for regular nodes. */
    protected T transferFunction;

    /** The control flow graph to perform the analysis on. */
    protected ControlFlowGraph cfg;

    /** The associated processing environment */
    protected final ProcessingEnvironment env;

    /** Instance of the types utility. */
    protected final Types types;

    /**
     * Then stores before every basic block (assumed to be 'no information' if
     * not present).
     */
    protected IdentityHashMap<Block, S> thenStores;

    /**
     * Else stores before every basic block (assumed to be 'no information' if
     * not present).
     */
    protected IdentityHashMap<Block, S> elseStores;

    /**
     * The transfer inputs before every basic block (assumed to be 'no information' if
     * not present).
     */
    protected IdentityHashMap<Block, TransferInput<A, S>> inputs;

    /**
     * The stores after every return statement.
     */
    protected IdentityHashMap<ReturnNode, TransferResult<A, S>> storesAtReturnStatements;

    /** The worklist used for the fix-point iteration. */
    protected Worklist worklist;

    /** Abstract values of nodes. */
    protected IdentityHashMap<Node, A> nodeValues;

    /** Map from (effectively final) local variable elements to their abstract value. */
    public HashMap<Element, A> finalLocalValues;

    /**
     * The node that is currently handled in the analysis (if it is running).
     * The following invariant holds:
     *
     * <pre>
     *   !isRunning &rArr; (currentNode == null)
     * </pre>
     */
    protected Node currentNode;

    /**
     * The tree that is currently being looked at. The transfer function can set
     * this tree to make sure that calls to {@code getValue} will not return
     * information for this given tree.
     */
    protected Tree currentTree;

    /**
     * The current transfer input when the analysis is running.
     */
    protected TransferInput<A, S> currentInput;

    public Tree getCurrentTree() {
        return currentTree;
    }

    public void setCurrentTree(Tree currentTree) {
        this.currentTree = currentTree;
    }

    /**
     * Construct an object that can perform a org.checkerframework.dataflow analysis over a control
     * flow graph. The transfer function is set later using
     * {@code setTransferFunction}.
     */
    public Analysis(ProcessingEnvironment env) {
        this.env = env;
        types = env.getTypeUtils();
    }

    /**
     * Construct an object that can perform a org.checkerframework.dataflow analysis over a control
     * flow graph, given a transfer function.
     */
    public Analysis(ProcessingEnvironment env, T transfer) {
        this(env);
        this.transferFunction = transfer;
    }

    public void setTransferFunction(T transfer) {
        this.transferFunction = transfer;
    }

    public T getTransferFunction() {
        return transferFunction;
    }

    public Types getTypes() {
        return types;
    }

    public ProcessingEnvironment getEnv() {
        return env;
    }

    /**
     * Perform the actual analysis. Should only be called once after the object
     * has been created.
     */
    public void performAnalysis(ControlFlowGraph cfg) {
        assert isRunning == false;
        isRunning = true;

        init(cfg);

        while (!worklist.isEmpty()) {
            Block b = worklist.poll();

            switch (b.getType()) {
            case REGULAR_BLOCK: {
                RegularBlock rb = (RegularBlock) b;

                // apply transfer function to contents
                TransferInput<A, S> inputBefore = getInputBefore(rb);
                currentInput = inputBefore.copy();
                TransferResult<A, S> transferResult = null;
                Node lastNode = null;
                boolean addToWorklistAgain = false;
                for (Node n : rb.getContents()) {
                    transferResult = callTransferFunction(n, currentInput);
                    addToWorklistAgain |= updateNodeValues(n, transferResult);
                    currentInput = new TransferInput<>(n, this, transferResult);
                    lastNode = n;
                }
                // loop will run at least one, making transferResult non-null

                // propagate store to successors
                Block succ = rb.getSuccessor();
                assert succ != null : "regular basic block without non-exceptional successor unexpected";
                propagateStoresTo(succ, lastNode, currentInput, rb.getFlowRule(), addToWorklistAgain);
                break;
            }

            case EXCEPTION_BLOCK: {
                ExceptionBlock eb = (ExceptionBlock) b;

                // apply transfer function to content
                TransferInput<A, S> inputBefore = getInputBefore(eb);
                currentInput = inputBefore.copy();
                Node node = eb.getNode();
                TransferResult<A, S> transferResult = callTransferFunction(
                        node, currentInput);
                boolean addToWorklistAgain = updateNodeValues(node, transferResult);

                // propagate store to successor
                Block succ = eb.getSuccessor();
                if (succ != null) {
                    currentInput = new TransferInput<>(node, this, transferResult);
                    // TODO? Variable wasn't used.
                    // Store.FlowRule storeFlow = eb.getFlowRule();
                    propagateStoresTo(succ, node, currentInput, eb.getFlowRule(), addToWorklistAgain);
                }

                // propagate store to exceptional successors
                for (Entry<TypeMirror, Set<Block>> e : eb.getExceptionalSuccessors()
                        .entrySet()) {
                    TypeMirror cause = e.getKey();
                    S exceptionalStore = transferResult
                            .getExceptionalStore(cause);
                    if (exceptionalStore != null) {
                        for (Block exceptionSucc : e.getValue()) {
                            addStoreBefore(exceptionSucc, node, exceptionalStore, Store.Kind.BOTH,
                                           addToWorklistAgain);
                        }
                    } else {
                        for (Block exceptionSucc : e.getValue()) {
                            addStoreBefore(exceptionSucc, node, inputBefore.copy().getRegularStore(),
                                           Store.Kind.BOTH, addToWorklistAgain);
                        }
                    }
                }
                break;
            }

            case CONDITIONAL_BLOCK: {
                ConditionalBlock cb = (ConditionalBlock) b;

                // get store before
                TransferInput<A, S> inputBefore = getInputBefore(cb);
                TransferInput<A, S> input = inputBefore.copy();

                // propagate store to successor
                Block thenSucc = cb.getThenSuccessor();
                Block elseSucc = cb.getElseSuccessor();

                propagateStoresTo(thenSucc, null, input, cb.getThenFlowRule(), false);
                propagateStoresTo(elseSucc, null, input, cb.getElseFlowRule(), false);
                break;
            }

            case SPECIAL_BLOCK: {
                // special basic blocks are empty and cannot throw exceptions,
                // thus there is no need to perform any analysis.
                SpecialBlock sb = (SpecialBlock) b;
                Block succ = sb.getSuccessor();
                if (succ != null) {
                    propagateStoresTo(succ, null, getInputBefore(b), sb.getFlowRule(), false);
                }
                break;
            }

            default:
                assert false;
                break;
            }
        }

        assert isRunning == true;
        isRunning = false;
    }

    /**
     * Propagate the stores in currentInput to the successor block, succ, according to the
     * flowRule.
     */
    protected void propagateStoresTo(Block succ, Node node, TransferInput<A, S> currentInput,
            Store.FlowRule flowRule, boolean addToWorklistAgain) {
        switch (flowRule) {
        case EACH_TO_EACH:
            if (currentInput.containsTwoStores()) {
                addStoreBefore(succ, node, currentInput.getThenStore(), Store.Kind.THEN,
                        addToWorklistAgain);
                addStoreBefore(succ, node, currentInput.getElseStore(), Store.Kind.ELSE,
                        addToWorklistAgain);
            } else {
                addStoreBefore(succ, node, currentInput.getRegularStore(), Store.Kind.BOTH,
                        addToWorklistAgain);
            }
            break;
        case THEN_TO_BOTH:
            addStoreBefore(succ, node, currentInput.getThenStore(), Store.Kind.BOTH,
                    addToWorklistAgain);
            break;
        case ELSE_TO_BOTH:
            addStoreBefore(succ, node, currentInput.getElseStore(), Store.Kind.BOTH,
                    addToWorklistAgain);
            break;
        case THEN_TO_THEN:
            addStoreBefore(succ, node, currentInput.getThenStore(), Store.Kind.THEN,
                    addToWorklistAgain);
            break;
        case ELSE_TO_ELSE:
            addStoreBefore(succ, node, currentInput.getElseStore(), Store.Kind.ELSE,
                    addToWorklistAgain);
            break;
        }
    }

    /**
     * Updates the value of node {@code node} to the value of the
     * {@code transferResult}. Returns true if the node's value changed, or a
     * store was updated.
     */
    protected boolean updateNodeValues(Node node, TransferResult<A, S> transferResult) {
      A newVal = transferResult.getResultValue();
      boolean nodeValueChanged = false;

      if (newVal != null) {
          A oldVal = nodeValues.get(node);
          nodeValues.put(node, newVal);
          nodeValueChanged = !Objects.equals(oldVal, newVal);
      }

      return nodeValueChanged || transferResult.storeChanged();
    }

    /**
     * Call the transfer function for node {@code node}, and set that node as
     * current node first.
     */
    protected TransferResult<A, S> callTransferFunction(Node node,
            TransferInput<A, S> store) {

        if (node.isLValue()) {
            // TODO: should the default behavior be to return either a regular
            // transfer result or a conditional transfer result (depending on
            // store.hasTwoStores()), or is the following correct?
            return new RegularTransferResult<A, S>(null,
                    store.getRegularStore());
        }
        store.node = node;
        currentNode = node;
        TransferResult<A, S> transferResult = node.accept(transferFunction,
                store);
        currentNode = null;
        if (node instanceof ReturnNode) {
            // save a copy of the store to later check if some property held at
            // a given return statement
            storesAtReturnStatements.put((ReturnNode) node, transferResult);
        }
        if (node instanceof AssignmentNode) {
            // store the flow-refined value for effectively final local variables
            AssignmentNode assignment = (AssignmentNode) node;
            Node lhst = assignment.getTarget();
            if (lhst instanceof LocalVariableNode) {
                LocalVariableNode lhs = (LocalVariableNode) lhst;
                Element elem = lhs.getElement();
                if (ElementUtils.isEffectivelyFinal(elem)) {
                    finalLocalValues.put(elem, transferResult.getResultValue());
                }
            }
        }
        return transferResult;
    }

    /** Initialize the analysis with a new control flow graph. */
    protected void init(ControlFlowGraph cfg) {
        this.cfg = cfg;
        thenStores = new IdentityHashMap<>();
        elseStores = new IdentityHashMap<>();
        inputs = new IdentityHashMap<>();
        storesAtReturnStatements = new IdentityHashMap<>();
        worklist = new Worklist(cfg);
        nodeValues = new IdentityHashMap<>();
        finalLocalValues = new HashMap<>();
        worklist.add(cfg.getEntryBlock());

        List<LocalVariableNode> parameters = null;
        UnderlyingAST underlyingAST = cfg.getUnderlyingAST();
        if (underlyingAST.getKind() == Kind.METHOD) {
            MethodTree tree = ((CFGMethod) underlyingAST).getMethod();
            parameters = new ArrayList<>();
            for (VariableTree p : tree.getParameters()) {
                LocalVariableNode var = new LocalVariableNode(p);
                parameters.add(var);
                // TODO: document that LocalVariableNode has no block that it
                // belongs to
            }
        } else if (underlyingAST.getKind() == Kind.LAMBDA) {
            LambdaExpressionTree lambda = ((CFGLambda) underlyingAST).getLambdaTree();
            parameters = new ArrayList<>();
            for (VariableTree p : lambda.getParameters()) {
                LocalVariableNode var = new LocalVariableNode(p);
                parameters.add(var);
                // TODO: document that LocalVariableNode has no block that it
                // belongs to
            }

        } else {
            // nothing to do
        }
        S initialStore = transferFunction.initialStore(underlyingAST, parameters);
        Block entry = cfg.getEntryBlock();
        thenStores.put(entry, initialStore);
        elseStores.put(entry, initialStore);
        inputs.put(entry, new TransferInput<>(null, this, initialStore));
    }

    /**
     * Add a basic block to the worklist. If {@code b} is already present,
     * the method does nothing.
     */
    protected void addToWorklist(Block b) {
        // TODO: use a more efficient way to check if b is already present
        if (!worklist.contains(b)) {
            worklist.add(b);
        }
    }

    /**
     * Add a store before the basic block {@code b} by merging with the
     * existing stores for that location.
     */
    protected void addStoreBefore(Block b, Node node, S s, Store.Kind kind,
            boolean addBlockToWorklist) {
        S thenStore = getStoreBefore(b, Store.Kind.THEN);
        S elseStore = getStoreBefore(b, Store.Kind.ELSE);

        switch (kind) {
        case THEN: {
            // Update the then store
            S newThenStore = (thenStore != null) ?
                thenStore.leastUpperBound(s) : s;
            if (!newThenStore.equals(thenStore)) {
                thenStores.put(b, newThenStore);
                if (elseStore != null) {
                    inputs.put(b, new TransferInput<>(node, this, newThenStore, elseStore));
                    addBlockToWorklist = true;
                }
            }
            break;
        }
        case ELSE: {
            // Update the else store
            S newElseStore = (elseStore != null) ?
                elseStore.leastUpperBound(s) : s;
            if (!newElseStore.equals(elseStore)) {
                elseStores.put(b, newElseStore);
                if (thenStore != null) {
                    inputs.put(b, new TransferInput<>(node, this, thenStore, newElseStore));
                    addBlockToWorklist = true;
                }
            }
            break;
        }
        case BOTH:
            if (thenStore == elseStore) {
                // Currently there is only one regular store
                S newStore = (thenStore != null) ?
                    thenStore.leastUpperBound(s) : s;
                if (!newStore.equals(thenStore)) {
                    thenStores.put(b, newStore);
                    elseStores.put(b, newStore);
                    inputs.put(b, new TransferInput<>(node, this, newStore));
                    addBlockToWorklist = true;
                }
            } else {
                boolean storeChanged = false;

                S newThenStore = (thenStore != null) ?
                    thenStore.leastUpperBound(s) : s;
                if (!newThenStore.equals(thenStore)) {
                    thenStores.put(b, newThenStore);
                    storeChanged = true;
                }

                S newElseStore = (elseStore != null) ?
                    elseStore.leastUpperBound(s) : s;
                if (!newElseStore.equals(elseStore)) {
                    elseStores.put(b, newElseStore);
                    storeChanged = true;
                }

                if (storeChanged) {
                    inputs.put(b, new TransferInput<>(node, this, newThenStore, newElseStore));
                    addBlockToWorklist = true;
                }
            }
        }

        if (addBlockToWorklist) {
            addToWorklist(b);
        }
    }

    /**
     * A worklist is a priority queue of blocks in which the order is given
     * by depth-first ordering to place non-loop predecessors ahead of successors.
     */
    protected static class Worklist {

        /** Map all blocks in the CFG to their depth-first order. */
        protected IdentityHashMap<Block, Integer> depthFirstOrder;

        /** Comparator to allow priority queue to order blocks by their depth-first
            order. */
        public class DFOComparator implements Comparator<Block> {
            @Override
            public int compare(Block b1, Block b2) {
                return depthFirstOrder.get(b1) - depthFirstOrder.get(b2);
            }
        }

        /** The backing priority queue. */
        protected PriorityQueue<Block> queue;


        public Worklist(ControlFlowGraph cfg) {
            depthFirstOrder = new IdentityHashMap<>();
            int count = 1;
            for (Block b : cfg.getDepthFirstOrderedBlocks()) {
                depthFirstOrder.put(b, count++);
            }

            queue = new PriorityQueue<Block>(11, new DFOComparator());
        }

        public boolean isEmpty() {
            return queue.isEmpty();
        }

        public boolean contains(Block block) {
            return queue.contains(block);
        }

        public void add(Block block) {
            queue.add(block);
        }

        public Block poll() {
            return queue.poll();
        }

        @Override
        public String toString() {
            return "Worklist(" + queue + ")";
        }
    }

    /**
     * Read the {@link TransferInput} for a particular basic block (or {@code null} if
     * none exists yet).
     */
    public /*@Nullable*/ TransferInput<A, S> getInput(Block b) {
        return getInputBefore(b);
    }

    /**
     * @return the transfer input corresponding to the location right before the basic
     *         block {@code b}.
     */
    protected /*@Nullable*/ TransferInput<A, S> getInputBefore(Block b) {
        return inputs.get(b);
    }

    /**
     * @return the store corresponding to the location right before the basic
     *         block {@code b}.
     */
    protected /*@Nullable*/ S getStoreBefore(Block b, Store.Kind kind) {
        switch (kind) {
        case THEN:
            return readFromStore(thenStores, b);
        case ELSE:
            return readFromStore(elseStores, b);
        default:
            assert false;
            return null;
        }
    }

    /**
     * Read the {@link Store} for a particular basic block from a map of stores
     * (or {@code null} if none exists yet).
     */
    protected static <S> /*@Nullable*/ S readFromStore(Map<Block, S> stores,
            Block b) {
        return stores.get(b);
    }

    /** Is the analysis currently running? */
    public boolean isRunning() {
        return isRunning;
    }

    /**
     * @return the abstract value for {@link Node} {@code n}, or {@code null} if
     *         no information is available. Note that if the analysis has not
     *         finished yet, this value might not represent the final value for
     *         this node.
     */
    public /*@Nullable*/ A getValue(Node n) {
        if (isRunning) {
            // we do not yet have a org.checkerframework.dataflow fact about the current node
            if (currentNode == n
                    || (currentTree != null && currentTree == n.getTree())) {
                return null;
            }
            // check that 'n' is a subnode of 'node'. Check immediate operands
            // first for efficiency.
            assert currentNode != null;
            assert !n.isLValue() : "Did not expect an lvalue, but got " + n;
            if (!(currentNode != n && (currentNode.getOperands().contains(n) || currentNode
                    .getTransitiveOperands().contains(n)))) {
                return null;
            }
            return nodeValues.get(n);
        }
        return nodeValues.get(n);
    }

    /**
     * @return the abstract value for {@link Tree} {@code t}, or {@code null} if
     *         no information is available. Note that if the analysis has not
     *         finished yet, this value might not represent the final value for
     *         this node.
     */
    public /*@Nullable*/ A getValue(Tree t) {
        // we do not yet have a org.checkerframework.dataflow fact about the current node
        if (t == currentTree) {
            return null;
        }
        Node nodeCorrespondingToTree = getNodeForTree(t);
        if (nodeCorrespondingToTree == null || nodeCorrespondingToTree.isLValue()) {
            return null;
        }
        return getValue(nodeCorrespondingToTree);
    }

    /**
     * Get the {@link Node} for a given {@link Tree}.
     */
    public Node getNodeForTree(Tree t) {
        return cfg.getNodeCorrespondingToTree(t);
    }

    /**
     * Get the {@link MethodTree} of the current CFG if the argument {@link Tree} maps
     * to a {@link Node} in the CFG or null otherwise.
     */
    public /*@Nullable*/ MethodTree getContainingMethod(Tree t) {
        return cfg.getContainingMethod(t);
    }

    /**
     * Get the {@link ClassTree} of the current CFG if the argument {@link Tree} maps
     * to a {@link Node} in the CFG or null otherwise.
     */
    public /*@Nullable*/ ClassTree getContainingClass(Tree t) {
        return cfg.getContainingClass(t);
    }

    public List<Pair<ReturnNode, TransferResult<A, S>>> getReturnStatementStores() {
        List<Pair<ReturnNode, TransferResult<A, S>>> result = new ArrayList<>();
        for (ReturnNode returnNode : cfg.getReturnNodes()) {
            TransferResult<A, S> store = storesAtReturnStatements
                    .get(returnNode);
            result.add(Pair.of(returnNode, store));
        }
        return result;
    }

    public AnalysisResult<A, S> getResult() {
        assert !isRunning;
        IdentityHashMap<Tree, Node> treeLookup = cfg.getTreeLookup();
        return new AnalysisResult<>(nodeValues, inputs, treeLookup, finalLocalValues);
    }

    /**
     * @return the regular exit store, or {@code null}, if there is no such
     *         store (because the method cannot exit through the regular exit
     *         block).
     */
    public /*@Nullable*/ S getRegularExitStore() {
        SpecialBlock regularExitBlock = cfg.getRegularExitBlock();
        if (inputs.containsKey(regularExitBlock)) {
            S regularExitStore = inputs.get(regularExitBlock).getRegularStore();
            return regularExitStore;
        } else {
            return null;
        }
    }

    public S getExceptionalExitStore() {
        S exceptionalExitStore = inputs.get(cfg.getExceptionalExitBlock())
                .getRegularStore();
        return exceptionalExitStore;
    }
}
