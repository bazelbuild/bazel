package org.checkerframework.dataflow.cfg;

/*>>>
import org.checkerframework.checker.nullness.qual.Nullable;
*/

import org.checkerframework.dataflow.analysis.AbstractValue;
import org.checkerframework.dataflow.analysis.Analysis;
import org.checkerframework.dataflow.analysis.Store;
import org.checkerframework.dataflow.analysis.TransferFunction;
import org.checkerframework.dataflow.analysis.TransferInput;
import org.checkerframework.dataflow.cfg.block.Block;
import org.checkerframework.dataflow.cfg.block.Block.BlockType;
import org.checkerframework.dataflow.cfg.block.ConditionalBlock;
import org.checkerframework.dataflow.cfg.block.ExceptionBlock;
import org.checkerframework.dataflow.cfg.block.RegularBlock;
import org.checkerframework.dataflow.cfg.block.SingleSuccessorBlock;
import org.checkerframework.dataflow.cfg.block.SpecialBlock;
import org.checkerframework.dataflow.cfg.node.Node;

import javax.lang.model.type.TypeMirror;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map.Entry;
import java.util.Queue;
import java.util.Set;

/**
 * Generate a graph description in the DOT language of a control graph.
 *
 * @author Stefan Heule
 *
 */
public class CFGDOTVisualizer {

    /**
     * Output a graph description in the DOT language, representing the control
     * flow graph starting at <code>entry</code>. Does not output verbose information
     * or stores at the beginning of basic blocks.
     *
     * @see #visualize(ControlFlowGraph, Block, Analysis, boolean)
     */
    public static String visualize(ControlFlowGraph cfg, Block entry) {
        return visualize(cfg, entry, null, false);
    }

    /**
     * Output a graph description in the DOT language, representing the control
     * flow graph starting at <code>entry</code>.
     *
     * @param entry
     *            The entry node of the control flow graph to be represented.
     * @param analysis
     *            An analysis containing information about the program
     *            represented by the CFG. The information includes {@link Store}
     *            s that are valid at the beginning of basic blocks reachable
     *            from <code>entry</code> and per-node information for value
     *            producing {@link Node}s. Can also be <code>null</code> to
     *            indicate that this information should not be output.
     * @param verbose
     *            Add more output to the CFG description.
     * @return String representation of the graph in the DOT language.
     */
    public static <A extends AbstractValue<A>, S extends Store<S>, T extends TransferFunction<A, S>> String visualize(
            ControlFlowGraph cfg,
            Block entry,
            /*@Nullable*/ Analysis<A, S, T> analysis,
            boolean verbose) {
        StringBuilder sb1 = new StringBuilder();
        StringBuilder sb2 = new StringBuilder();
        Set<Block> visited = new HashSet<>();
        Queue<Block> worklist = new LinkedList<>();
        Block cur = entry;
        visited.add(entry);

        // header
        sb1.append("digraph {\n");
        sb1.append("    node [shape=rectangle];\n\n");

        // traverse control flow graph and define all arrows
        while (true) {
            if (cur == null)
                break;

            if (cur.getType() == BlockType.CONDITIONAL_BLOCK) {
                ConditionalBlock ccur = ((ConditionalBlock) cur);
                Block thenSuccessor = ccur.getThenSuccessor();
                sb2.append("    " + ccur.getId() + " -> "
                        + thenSuccessor.getId());
                sb2.append(" [label=\"then\\n" + ccur.getThenFlowRule() + "\"];\n");
                if (!visited.contains(thenSuccessor)) {
                    visited.add(thenSuccessor);
                    worklist.add(thenSuccessor);
                }
                Block elseSuccessor = ccur.getElseSuccessor();
                sb2.append("    " + ccur.getId() + " -> "
                        + elseSuccessor.getId());
                sb2.append(" [label=\"else\\n" + ccur.getElseFlowRule() + "\"];\n");
                if (!visited.contains(elseSuccessor)) {
                    visited.add(elseSuccessor);
                    worklist.add(elseSuccessor);
                }
            } else {
                assert cur instanceof SingleSuccessorBlock;
                Block b = ((SingleSuccessorBlock) cur).getSuccessor();
                if (b != null) {
                    sb2.append("    " + cur.getId() + " -> " + b.getId());
                    sb2.append(" [label=\"" + ((SingleSuccessorBlock) cur).getFlowRule() + "\"];\n");
                    if (!visited.contains(b)) {
                        visited.add(b);
                        worklist.add(b);
                    }
                }
            }

            // exceptional edges
            if (cur.getType() == BlockType.EXCEPTION_BLOCK) {
                ExceptionBlock ecur = (ExceptionBlock) cur;
                for (Entry<TypeMirror, Set<Block>> e : ecur
                        .getExceptionalSuccessors().entrySet()) {
                    Set<Block> blocks = e.getValue();
                    TypeMirror cause = e.getKey();
                    String exception = cause.toString();
                    if (exception.startsWith("java.lang.")) {
                        exception = exception.replace("java.lang.", "");
                    }

                    for (Block b : blocks) {
                        sb2.append("    " + cur.getId() + " -> " + b.getId());
                        sb2.append(" [label=\"" + exception + "\"];\n");
                        if (!visited.contains(b)) {
                            visited.add(b);
                            worklist.add(b);
                        }
                    }
                }
            }

            cur = worklist.poll();
        }

        IdentityHashMap<Block, List<Integer>> processOrder = getProcessOrder(cfg);

        // definition of all nodes including their labels
        for (Block v : visited) {
            sb1.append("    " + v.getId() + " [");
            if (v.getType() == BlockType.CONDITIONAL_BLOCK) {
                sb1.append("shape=polygon sides=8 ");
            } else if (v.getType() == BlockType.SPECIAL_BLOCK) {
                sb1.append("shape=oval ");
            }
            sb1.append("label=\"");
            if (verbose) {
                sb1.append("Process order: " + processOrder.get(v).toString().replaceAll("[\\[\\]]", "") + "\\n");
            }
            sb1.append(visualizeContent(v, analysis, verbose).replace("\\n", "\\l")
                    + " \",];\n");
        }

        sb1.append("\n");
        sb1.append(sb2);

        // footer
        sb1.append("}\n");

        return sb1.toString();
    }

    private static IdentityHashMap<Block, List<Integer>> getProcessOrder(ControlFlowGraph cfg) {
        IdentityHashMap<Block, List<Integer>> depthFirstOrder = new IdentityHashMap<>();
        int count = 1;
        for (Block b : cfg.getDepthFirstOrderedBlocks()) {
            if (depthFirstOrder.get(b) == null) {
                depthFirstOrder.put(b, new ArrayList<Integer>());
            }
            depthFirstOrder.get(b).add(count++);
        }
        return depthFirstOrder;
    }

    /**
     * Produce a string representation of the contests of a basic block.
     *
     * @param bb
     *            Basic block to visualize.
     * @return String representation.
     */
    protected static <A extends AbstractValue<A>, S extends Store<S>, T extends TransferFunction<A, S>> String visualizeContent(
            Block bb,
            /*@Nullable*/ Analysis<A, S, T> analysis,
            boolean verbose) {

        StringBuilder sb = new StringBuilder();

        // loop over contents
        List<Node> contents = new LinkedList<>();
        switch (bb.getType()) {
        case REGULAR_BLOCK:
            contents.addAll(((RegularBlock) bb).getContents());
            break;
        case EXCEPTION_BLOCK:
            contents.add(((ExceptionBlock) bb).getNode());
            break;
        case CONDITIONAL_BLOCK:
            break;
        case SPECIAL_BLOCK:
            break;
        default:
            assert false : "All types of basic blocks covered";
        }
        boolean notFirst = false;
        for (Node t : contents) {
            if (notFirst) {
                sb.append("\\n");
            }
            notFirst = true;
            sb.append(prepareString(visualizeNode(t, analysis)));
        }

        // handle case where no contents are present
        boolean centered = false;
        if (sb.length() == 0) {
            centered = true;
            if (bb.getType() == BlockType.SPECIAL_BLOCK) {
                SpecialBlock sbb = (SpecialBlock) bb;
                switch (sbb.getSpecialType()) {
                case ENTRY:
                    sb.append("<entry>");
                    break;
                case EXIT:
                    sb.append("<exit>");
                    break;
                case EXCEPTIONAL_EXIT:
                    sb.append("<exceptional-exit>");
                    break;
                }
            } else if (bb.getType() == BlockType.CONDITIONAL_BLOCK) {
                return "";
            } else {
                return "?? empty ??";
            }
        }

        // visualize transfer input if necessary
        if (analysis != null) {
            TransferInput<A, S> input = analysis.getInput(bb);
            StringBuilder sb2 = new StringBuilder();

            // split input representation to two lines
            String s = input.toDOToutput().replace("}, else={", "}\\nelse={");
            sb2.append("Before:");
            sb2.append(s.subSequence(1, s.length() - 1));

            // separator
            sb2.append("\\n~~~~~~~~~\\n");
            sb2.append(sb);
            sb = sb2;

            if (verbose) {
                Node lastNode = null;
                switch (bb.getType()) {
                    case REGULAR_BLOCK:
                        List<Node> blockContents = ((RegularBlock) bb).getContents();
                        lastNode = contents.get(blockContents.size() - 1);
                        break;
                    case EXCEPTION_BLOCK:
                        lastNode = ((ExceptionBlock) bb).getNode();
                        break;
                }
                if (lastNode != null) {
                    sb2.append("\\n~~~~~~~~~\\n");
                    s = analysis.getResult().getStoreAfter(lastNode.getTree()).
                            toDOToutput().replace("}, else={", "}\\nelse={");
                    sb2.append("After:");
                    sb2.append(s);
                }
            }
        }

        return sb.toString() + (centered ? "" : "\\n");
    }

    protected static <A extends AbstractValue<A>, S extends Store<S>, T extends TransferFunction<A, S>>
    String visualizeNode(Node t, /*@Nullable*/ Analysis<A, S, T> analysis) {
        A value = analysis.getValue(t);
        String valueInfo = "";
        if (value != null) {
            valueInfo = "    > " + value.toString();
        }
        return t.toString() + "   [ " + visualizeType(t) + " ]" + valueInfo;
    }

    protected static String visualizeType(Node t) {
        String name = t.getClass().getSimpleName();
        return name.replace("Node", "");
    }

    protected static String prepareString(String s) {
        return s.replace("\"", "\\\"");
    }
}
