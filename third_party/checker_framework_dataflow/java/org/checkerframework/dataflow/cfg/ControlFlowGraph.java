package org.checkerframework.dataflow.cfg;

/*>>>
import org.checkerframework.checker.nullness.qual.Nullable;
*/

import org.checkerframework.dataflow.cfg.block.Block;
import org.checkerframework.dataflow.cfg.block.Block.BlockType;
import org.checkerframework.dataflow.cfg.block.ConditionalBlock;
import org.checkerframework.dataflow.cfg.block.ExceptionBlock;
import org.checkerframework.dataflow.cfg.block.SingleSuccessorBlock;
import org.checkerframework.dataflow.cfg.block.SpecialBlock;
import org.checkerframework.dataflow.cfg.block.SpecialBlockImpl;
import org.checkerframework.dataflow.cfg.node.Node;
import org.checkerframework.dataflow.cfg.node.ReturnNode;

import java.util.Collections;
import java.util.Deque;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.Set;

import com.sun.source.tree.ClassTree;
import com.sun.source.tree.MethodTree;
import com.sun.source.tree.Tree;

/**
 * A control flow graph (CFG for short) of a single method.
 *
 * @author Stefan Heule
 *
 */
public class ControlFlowGraph {

    /** The entry block of the control flow graph. */
    protected final SpecialBlock entryBlock;

    /** The regular exit block of the control flow graph. */
    protected final SpecialBlock regularExitBlock;

    /** The exceptional exit block of the control flow graph. */
    protected final SpecialBlock exceptionalExitBlock;

    /** The AST this CFG corresponds to. */
    protected UnderlyingAST underlyingAST;

    /**
     * Maps from AST {@link Tree}s to {@link Node}s.  Every Tree that produces
     * a value will have at least one corresponding Node.  Trees
     * that undergo conversions, such as boxing or unboxing, can map to two
     * distinct Nodes.  The Node for the pre-conversion value is stored
     * in treeLookup, while the Node for the post-conversion value
     * is stored in convertedTreeLookup.
     */
    protected IdentityHashMap<Tree, Node> treeLookup;

    /** Map from AST {@link Tree}s to post-conversion {@link Node}s. */
    protected IdentityHashMap<Tree, Node> convertedTreeLookup;

    /**
     * All return nodes (if any) encountered. Only includes return
     * statements that actually return something
     */
    protected final List<ReturnNode> returnNodes;

    public ControlFlowGraph(SpecialBlock entryBlock, SpecialBlockImpl regularExitBlock, SpecialBlockImpl exceptionalExitBlock, UnderlyingAST underlyingAST,
            IdentityHashMap<Tree, Node> treeLookup,
            IdentityHashMap<Tree, Node> convertedTreeLookup,
            List<ReturnNode> returnNodes) {
        super();
        this.entryBlock = entryBlock;
        this.underlyingAST = underlyingAST;
        this.treeLookup = treeLookup;
        this.convertedTreeLookup = convertedTreeLookup;
        this.regularExitBlock = regularExitBlock;
        this.exceptionalExitBlock = exceptionalExitBlock;
        this.returnNodes = returnNodes;
    }

    /**
     * @return the {@link Node} to which the {@link Tree} {@code t}
     *         corresponds.
     */
    public Node getNodeCorrespondingToTree(Tree t) {
        if (convertedTreeLookup.containsKey(t)) {
            return convertedTreeLookup.get(t);
        } else {
            return treeLookup.get(t);
        }
    }

    /** @return the entry block of the control flow graph. */
    public SpecialBlock getEntryBlock() {
        return entryBlock;
    }

    public List<ReturnNode> getReturnNodes() {
        return returnNodes;
    }

    public SpecialBlock getRegularExitBlock() {
        return regularExitBlock;
    }

    public SpecialBlock getExceptionalExitBlock() {
        return exceptionalExitBlock;
    }

    /** @return the AST this CFG corresponds to. */
    public UnderlyingAST getUnderlyingAST() {
        return underlyingAST;
    }

    /**
     * @return the set of all basic block in this control flow graph
     */
    public Set<Block> getAllBlocks() {
        Set<Block> visited = new HashSet<>();
        Queue<Block> worklist = new LinkedList<>();
        Block cur = entryBlock;
        visited.add(entryBlock);

        // traverse the whole control flow graph
        while (true) {
            if (cur == null) {
                break;
            }

            Queue<Block> succs = new LinkedList<>();
            if (cur.getType() == BlockType.CONDITIONAL_BLOCK) {
                ConditionalBlock ccur = ((ConditionalBlock) cur);
                succs.add(ccur.getThenSuccessor());
                succs.add(ccur.getElseSuccessor());
            } else {
                assert cur instanceof SingleSuccessorBlock;
                Block b = ((SingleSuccessorBlock) cur).getSuccessor();
                if (b != null) {
                    succs.add(b);
                }
            }

            if (cur.getType() == BlockType.EXCEPTION_BLOCK) {
                ExceptionBlock ecur = (ExceptionBlock) cur;
                for (Set<Block> exceptionSuccSet : ecur.getExceptionalSuccessors().values()) {
                    succs.addAll(exceptionSuccSet);
                }
            }

            for (Block b : succs) {
                if (!visited.contains(b)) {
                    visited.add(b);
                    worklist.add(b);
                }
            }

            cur = worklist.poll();
        }

        return visited;
    }

    /**
     * @return the list of all basic block in this control flow graph
     * in reversed depth-first postorder sequence.
     *
     * Blocks may appear more than once in the sequence.
     */
    public List<Block> getDepthFirstOrderedBlocks() {
        List<Block> dfsOrderResult = new LinkedList<>();
        Set<Block> visited = new HashSet<>();
        Deque<Block> worklist = new LinkedList<>();
        worklist.add(entryBlock);
        while (!worklist.isEmpty()) {
            Block cur = worklist.getLast();
            if (visited.contains(cur)) {
                dfsOrderResult.add(cur);
                worklist.removeLast();
            } else {
                visited.add(cur);
                Deque<Block> successors = getSuccessors(cur);
                successors.removeAll(visited);
                worklist.addAll(successors);
            }
        }

        Collections.reverse(dfsOrderResult);
        return dfsOrderResult;
    }

    /**
     * Get a list of all successor Blocks for cur
     * @return a Deque of successor Blocks
     */
    private Deque<Block> getSuccessors(Block cur) {
        Deque<Block> succs = new LinkedList<>();
        if (cur.getType() == BlockType.CONDITIONAL_BLOCK) {
            ConditionalBlock ccur = ((ConditionalBlock) cur);
            succs.add(ccur.getThenSuccessor());
            succs.add(ccur.getElseSuccessor());
        } else {
            assert cur instanceof SingleSuccessorBlock;
            Block b = ((SingleSuccessorBlock) cur).getSuccessor();
            if (b != null) {
                succs.add(b);
            }
        }

        if (cur.getType() == BlockType.EXCEPTION_BLOCK) {
            ExceptionBlock ecur = (ExceptionBlock) cur;
            for (Set<Block> exceptionSuccSet : ecur.getExceptionalSuccessors().values()) {
                succs.addAll(exceptionSuccSet);
            }
        }
        return succs;
    }

    /**
     * @return the tree-lookup map
     */
    public IdentityHashMap<Tree, Node> getTreeLookup() {
        return new IdentityHashMap<>(treeLookup);
    }

    /**
     * Get the {@link MethodTree} of the CFG if the argument {@link Tree} maps
     * to a {@link Node} in the CFG or null otherwise.
     */
    public /*@Nullable*/ MethodTree getContainingMethod(Tree t) {
        if (treeLookup.containsKey(t)) {
            if (underlyingAST.getKind() == UnderlyingAST.Kind.METHOD) {
                UnderlyingAST.CFGMethod cfgMethod = (UnderlyingAST.CFGMethod) underlyingAST;
                return cfgMethod.getMethod();
            }
        }
        return null;
    }

    /**
     * Get the {@link ClassTree} of the CFG if the argument {@link Tree} maps
     * to a {@link Node} in the CFG or null otherwise.
     */
    public /*@Nullable*/ ClassTree getContainingClass(Tree t) {
        if (treeLookup.containsKey(t)) {
            if (underlyingAST.getKind() == UnderlyingAST.Kind.METHOD) {
                UnderlyingAST.CFGMethod cfgMethod = (UnderlyingAST.CFGMethod) underlyingAST;
                return cfgMethod.getClassTree();
            }
        }
        return null;
    }
}
