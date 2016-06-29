package org.checkerframework.dataflow.cfg;

/*>>>
import org.checkerframework.checker.nullness.qual.Nullable;
*/

import org.checkerframework.dataflow.analysis.Store;
import org.checkerframework.dataflow.cfg.CFGBuilder.ExtendedNode.ExtendedNodeType;
import org.checkerframework.dataflow.cfg.UnderlyingAST.CFGMethod;
import org.checkerframework.dataflow.cfg.block.Block;
import org.checkerframework.dataflow.cfg.block.Block.BlockType;
import org.checkerframework.dataflow.cfg.block.BlockImpl;
import org.checkerframework.dataflow.cfg.block.ConditionalBlockImpl;
import org.checkerframework.dataflow.cfg.block.ExceptionBlockImpl;
import org.checkerframework.dataflow.cfg.block.RegularBlockImpl;
import org.checkerframework.dataflow.cfg.block.SingleSuccessorBlockImpl;
import org.checkerframework.dataflow.cfg.block.SpecialBlock.SpecialBlockType;
import org.checkerframework.dataflow.cfg.block.SpecialBlockImpl;
import org.checkerframework.dataflow.cfg.node.*;
import org.checkerframework.dataflow.qual.TerminatesExecution;
import org.checkerframework.dataflow.util.MostlySingleton;
import org.checkerframework.javacutil.AnnotationProvider;
import org.checkerframework.javacutil.BasicAnnotationProvider;
import org.checkerframework.javacutil.ElementUtils;
import org.checkerframework.javacutil.InternalUtils;
import org.checkerframework.javacutil.Pair;
import org.checkerframework.javacutil.TreeUtils;
import org.checkerframework.javacutil.TypesUtils;
import org.checkerframework.javacutil.trees.TreeBuilder;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import javax.annotation.processing.ProcessingEnvironment;
import javax.lang.model.element.Element;
import javax.lang.model.element.ExecutableElement;
import javax.lang.model.element.Name;
import javax.lang.model.element.TypeElement;
import javax.lang.model.element.VariableElement;
import javax.lang.model.type.ArrayType;
import javax.lang.model.type.DeclaredType;
import javax.lang.model.type.PrimitiveType;
import javax.lang.model.type.ReferenceType;
import javax.lang.model.type.TypeKind;
import javax.lang.model.type.TypeMirror;
import javax.lang.model.type.TypeVariable;
import javax.lang.model.type.UnionType;
import javax.lang.model.util.Elements;
import javax.lang.model.util.Types;

import com.sun.source.tree.*;
import com.sun.source.tree.Tree.Kind;
import com.sun.source.util.TreePath;
import com.sun.source.util.TreePathScanner;
import com.sun.source.util.Trees;
import com.sun.tools.javac.code.Symbol.MethodSymbol;
import com.sun.tools.javac.code.Type;
import com.sun.tools.javac.processing.JavacProcessingEnvironment;
import com.sun.tools.javac.util.Context;

/**
 * Builds the control flow graph of some Java code (either a method, or an
 * arbitrary statement).
 *
 * <p>
 *
 * The translation of the AST to the CFG is split into three phases:
 * <ol>
 * <li><em>Phase one.</em> In the first phase, the AST is translated into a
 * sequence of {@link org.checkerframework.dataflow.cfg.CFGBuilder.ExtendedNode}s. An extended node can either be a
 * {@link Node}, or one of several meta elements such as a conditional or
 * unconditional jump or a node with additional information about exceptions.
 * Some of the extended nodes contain labels (e.g., for the jump target), and
 * phase one additionally creates a mapping from labels to extended nodes.
 * Finally, the list of leaders is computed: A leader is an extended node which
 * will give rise to a basic block in phase two.</li>
 * <li><em>Phase two.</em> In this phase, the sequence of extended nodes is
 * translated to a graph of control flow blocks that contain nodes. The meta
 * elements from phase one are translated into the correct edges.</li>
 * <li><em>Phase three.</em> The control flow graph generated in phase two can
 * contain degenerate basic blocks such as empty regular basic blocks or
 * conditional basic blocks that have the same block as both 'then' and 'else'
 * successor. This phase removes these cases while preserving the control flow
 * structure.</li>
 * </ol>
 *
 * @author Stefan Heule
 *
 */
public class CFGBuilder {

    /** Can assertions be assumed to be disabled? */
    protected final boolean assumeAssertionsDisabled;

    /** Can assertions be assumed to be enabled? */
    protected final boolean assumeAssertionsEnabled;

    public CFGBuilder(boolean assumeAssertionsEnabled, boolean assumeAssertionsDisabled) {
        assert !(assumeAssertionsDisabled && assumeAssertionsEnabled);
        this.assumeAssertionsEnabled = assumeAssertionsEnabled;
        this.assumeAssertionsDisabled = assumeAssertionsDisabled;
    }

    /**
     * Class declarations that have been encountered when building the
     * control-flow graph for a method.
     */
    protected final List<ClassTree> declaredClasses = new LinkedList<>();

    public List<ClassTree> getDeclaredClasses() {
        return declaredClasses;
    }

    /**
     * Lambdas encountered when building the control-flow graph for
     * a method, variable initializer, or initializer.
     */
    protected final List<LambdaExpressionTree> declaredLambdas = new LinkedList<>();

    public List<LambdaExpressionTree> getDeclaredLambdas() {
        return declaredLambdas;
    }

    /**
     * Build the control flow graph of some code.
     */
    public static ControlFlowGraph build(
            CompilationUnitTree root, ProcessingEnvironment env,
            UnderlyingAST underlyingAST, boolean assumeAssertionsEnabled, boolean assumeAssertionsDisabled) {
        return new CFGBuilder(assumeAssertionsEnabled, assumeAssertionsDisabled).run(root, env, underlyingAST);
    }

    /**
     * Build the control flow graph of some code (method, initializer block, ...).
     * bodyPath is the TreePath to the body of that code.
     */
    public static ControlFlowGraph build(
            TreePath bodyPath, ProcessingEnvironment env,
            UnderlyingAST underlyingAST, boolean assumeAssertionsEnabled, boolean assumeAssertionsDisabled) {
        return new CFGBuilder(assumeAssertionsEnabled, assumeAssertionsDisabled).run(bodyPath, env, underlyingAST);
    }

    /**
     * Build the control flow graph of a method.
     */
    public static ControlFlowGraph build(
            CompilationUnitTree root, ProcessingEnvironment env,
            MethodTree tree, ClassTree classTree, boolean assumeAssertionsEnabled, boolean assumeAssertionsDisabled) {
        return new CFGBuilder(assumeAssertionsEnabled, assumeAssertionsDisabled).run(root, env, tree, classTree);
    }

    /**
     * Build the control flow graph of some code.
     */
    public static ControlFlowGraph build(
            CompilationUnitTree root, ProcessingEnvironment env,
            UnderlyingAST underlyingAST) {
        return new CFGBuilder(false, false).run(root, env, underlyingAST);
    }

    /**
     * Build the control flow graph of a method.
     */
    public static ControlFlowGraph build(
            CompilationUnitTree root, ProcessingEnvironment env,
            MethodTree tree, ClassTree classTree) {
        return new CFGBuilder(false, false).run(root, env, tree, classTree);
    }

    /**
     * Build the control flow graph of some code.
     */
    public ControlFlowGraph run(
            CompilationUnitTree root, ProcessingEnvironment env,
            UnderlyingAST underlyingAST) {
        declaredClasses.clear();
        declaredLambdas.clear();

        TreeBuilder builder = new TreeBuilder(env);
        AnnotationProvider annotationProvider = new BasicAnnotationProvider();
        PhaseOneResult phase1result = new CFGTranslationPhaseOne().process(
                root, env, underlyingAST, exceptionalExitLabel, builder, annotationProvider);
        ControlFlowGraph phase2result = new CFGTranslationPhaseTwo()
                .process(phase1result);
        ControlFlowGraph phase3result = CFGTranslationPhaseThree
                .process(phase2result);
        return phase3result;
    }

    /**
     * Build the control flow graph of some code (method, initializer block, ...).
     * bodyPath is the TreePath to the body of that code.
     */
    public ControlFlowGraph run(
            TreePath bodyPath, ProcessingEnvironment env,
            UnderlyingAST underlyingAST) {
        declaredClasses.clear();
        TreeBuilder builder = new TreeBuilder(env);
        AnnotationProvider annotationProvider = new BasicAnnotationProvider();
        PhaseOneResult phase1result = new CFGTranslationPhaseOne().process(
                bodyPath, env, underlyingAST, exceptionalExitLabel, builder, annotationProvider);
        ControlFlowGraph phase2result = new CFGTranslationPhaseTwo()
                .process(phase1result);
        ControlFlowGraph phase3result = CFGTranslationPhaseThree
                .process(phase2result);
        return phase3result;
    }

    /**
     * Build the control flow graph of a method.
     */
    public ControlFlowGraph run(
            CompilationUnitTree root, ProcessingEnvironment env,
            MethodTree tree, ClassTree classTree) {
        UnderlyingAST underlyingAST = new CFGMethod(tree, classTree);
        return run(root, env, underlyingAST);
    }

    /* --------------------------------------------------------- */
    /* Extended Node Types and Labels */
    /* --------------------------------------------------------- */

    /** Special label to identify the exceptional exit. */
    protected final Label exceptionalExitLabel = new Label();

    /** Special label to identify the regular exit. */
    protected final Label regularExitLabel = new Label();

    /**
     * An extended node can be one of several things (depending on its
     * {@code type}):
     * <ul>
     * <li><em>NODE</em>. An extended node of this type is just a wrapper for a
     * {@link Node} (that cannot throw exceptions).</li>
     * <li><em>EXCEPTION_NODE</em>. A wrapper for a {@link Node} which can throw
     * exceptions. It contains a label for every possible exception type the
     * node might throw.</li>
     * <li><em>UNCONDITIONAL_JUMP</em>. An unconditional jump to a label.</li>
     * <li><em>TWO_TARGET_CONDITIONAL_JUMP</em>. A conditional jump with two
     * targets for both the 'then' and 'else' branch.</li>
     * </ul>
     */
    protected static abstract class ExtendedNode {

        /**
         * The basic block this extended node belongs to (as determined in phase
         * two).
         */
        protected BlockImpl block;

        /** Type of this node. */
        protected ExtendedNodeType type;

        /** Does this node terminate the execution? (e.g., "System.exit()") */
        protected boolean terminatesExecution = false;

        public ExtendedNode(ExtendedNodeType type) {
            this.type = type;
        }

        /** Extended node types (description see above). */
        public enum ExtendedNodeType {
            NODE, EXCEPTION_NODE, UNCONDITIONAL_JUMP, CONDITIONAL_JUMP
        }

        public ExtendedNodeType getType() {
            return type;
        }

        public boolean getTerminatesExecution() {
            return terminatesExecution;
        }

        public void setTerminatesExecution(boolean terminatesExecution) {
            this.terminatesExecution = terminatesExecution;
        }

        /**
         * @return the node contained in this extended node (only applicable if
         *         the type is {@code NODE} or {@code EXCEPTION_NODE}).
         */
        public Node getNode() {
            assert false;
            return null;
        }

        /**
         * @return the label associated with this extended node (only applicable
         *         if type is {@link ExtendedNodeType#CONDITIONAL_JUMP} or
         *         {@link ExtendedNodeType#UNCONDITIONAL_JUMP}).
         */
        public Label getLabel() {
            assert false;
            return null;
        }

        public BlockImpl getBlock() {
            return block;
        }

        public void setBlock(BlockImpl b) {
            this.block = b;
        }

        @Override
        public String toString() {
            return "ExtendedNode(" + type + ")";
        }
    }

    /**
     * An extended node of type {@code NODE}.
     */
    protected static class NodeHolder extends ExtendedNode {

        protected Node node;

        public NodeHolder(Node node) {
            super(ExtendedNodeType.NODE);
            this.node = node;
        }

        @Override
        public Node getNode() {
            return node;
        }

        @Override
        public String toString() {
            return "NodeHolder(" + node + ")";
        }

    }

    /**
     * An extended node of type {@code EXCEPTION_NODE}.
     */
    protected static class NodeWithExceptionsHolder extends ExtendedNode {

        protected Node node;
        /**
         * Map from exception type to labels of successors that may
         * be reached as a result of that exception.
         */
        protected Map<TypeMirror, Set<Label>> exceptions;

        public NodeWithExceptionsHolder(Node node,
                Map<TypeMirror, Set<Label>> exceptions) {
            super(ExtendedNodeType.EXCEPTION_NODE);
            this.node = node;
            this.exceptions = exceptions;
        }

        public Map<TypeMirror, Set<Label>> getExceptions() {
            return exceptions;
        }

        @Override
        public Node getNode() {
            return node;
        }

        @Override
        public String toString() {
            return "NodeWithExceptionsHolder(" + node + ")";
        }

    }

    /**
     * An extended node of type {@link ExtendedNodeType#CONDITIONAL_JUMP}.
     *
     * <p>
     *
     * <em>Important:</em> In the list of extended nodes, there should not be
     * any labels that point to a conditional jump. Furthermore, the node
     * directly ahead of any conditional jump has to be a
     * {@link NodeWithExceptionsHolder} or {@link NodeHolder}, and the node held
     * by that extended node is required to be of boolean type.
     */
    protected static class ConditionalJump extends ExtendedNode {

        protected Label trueSucc;
        protected Label falseSucc;

        protected Store.FlowRule trueFlowRule;
        protected Store.FlowRule falseFlowRule;

        public ConditionalJump(Label trueSucc, Label falseSucc) {
            super(ExtendedNodeType.CONDITIONAL_JUMP);
            this.trueSucc = trueSucc;
            this.falseSucc = falseSucc;
        }

        public Label getThenLabel() {
            return trueSucc;
        }

        public Label getElseLabel() {
            return falseSucc;
        }

        public Store.FlowRule getTrueFlowRule() {
            return trueFlowRule;
        }

        public Store.FlowRule getFalseFlowRule() {
            return falseFlowRule;
        }

        public void setTrueFlowRule(Store.FlowRule rule) {
            trueFlowRule = rule;
        }

        public void setFalseFlowRule(Store.FlowRule rule) {
            falseFlowRule = rule;
        }

        @Override
        public String toString() {
            return "TwoTargetConditionalJump(" + getThenLabel() + ","
                    + getElseLabel() + ")";
        }
    }

    /**
     * An extended node of type {@link ExtendedNodeType#UNCONDITIONAL_JUMP}.
     */
    protected static class UnconditionalJump extends ExtendedNode {

        protected Label jumpTarget;

        public UnconditionalJump(Label jumpTarget) {
            super(ExtendedNodeType.UNCONDITIONAL_JUMP);
            this.jumpTarget = jumpTarget;
        }

        @Override
        public Label getLabel() {
            return jumpTarget;
        }

        @Override
        public String toString() {
            return "JumpMarker(" + getLabel() + ")";
        }
    }

    /**
     * A label is used to refer to other extended nodes using a mapping from
     * labels to extended nodes. Labels get their names either from labeled
     * statements in the source code or from internally generated unique names.
     */
    protected static class Label {
        private static int uid = 0;

        protected String name;

        public Label(String name) {
            this.name = name;
        }

        public Label() {
            this.name = uniqueName();
        }

        @Override
        public String toString() {
            return name;
        }

        /**
         * Return a new unique label name that cannot be confused with a Java
         * source code label.
         *
         * @return a new unique label name
         */
        private static String uniqueName() {
            return "%L" + uid++;
        }
    }

    /**
     * A TryFrame takes a thrown exception type and maps it to a set
     * of possible control-flow successors.
     */
    protected static interface TryFrame {
        /**
         * Given a type of thrown exception, add the set of possible control
         * flow successor {@link Label}s to the argument set.  Return true
         * if the exception is known to be caught by one of those labels and
         * false if it may propagate still further.
         */
        public boolean possibleLabels(TypeMirror thrown, Set<Label> labels);
    }

    /**
     * A TryCatchFrame contains an ordered list of catch labels that apply
     * to exceptions with specific types.
     */
    protected static class TryCatchFrame implements TryFrame {
        protected Types types;

        /** An ordered list of pairs because catch blocks are ordered. */
        protected List<Pair<TypeMirror, Label>> catchLabels;

        public TryCatchFrame(Types types, List<Pair<TypeMirror, Label>> catchLabels) {
            this.types = types;
            this.catchLabels = catchLabels;
        }

        /**
         * Given a type of thrown exception, add the set of possible control
         * flow successor {@link Label}s to the argument set.  Return true
         * if the exception is known to be caught by one of those labels and
         * false if it may propagate still further.
         */
        @Override
        public boolean possibleLabels(TypeMirror thrown, Set<Label> labels) {
            // A conservative approach would be to say that every catch block
            // might execute for any thrown exception, but we try to do better.
            //
            // We rely on several assumptions that seem to hold as of Java 7.
            // 1) An exception parameter in a catch block must be either
            //    a declared type or a union composed of declared types,
            //    all of which are subtypes of Throwable.
            // 2) A thrown type must either be a declared type or a variable
            //    that extends a declared type, which is a subtype of Throwable.
            //
            // Under those assumptions, if the thrown type (or its bound) is
            // a subtype of the caught type (or one of its alternatives), then
            // the catch block must apply and none of the later ones can apply.
            // Otherwise, if the thrown type (or its bound) is a supertype
            // of the caught type (or one of its alternatives), then the catch
            // block may apply, but so may later ones.
            // Otherwise, the thrown type and the caught type are unrelated
            // declared types, so they do not overlap on any non-null value.

            while (!(thrown instanceof DeclaredType)) {
                assert thrown instanceof TypeVariable :
                    "thrown type must be a variable or a declared type";
                thrown = ((TypeVariable)thrown).getUpperBound();
            }
            DeclaredType declaredThrown = (DeclaredType)thrown;
            assert thrown != null : "thrown type must be bounded by a declared type";

            for (Pair<TypeMirror, Label> pair : catchLabels) {
                TypeMirror caught = pair.first;
                boolean canApply = false;

                if (caught instanceof DeclaredType) {
                    DeclaredType declaredCaught = (DeclaredType)caught;
                    if (types.isSubtype(declaredThrown, declaredCaught)) {
                        // No later catch blocks can apply.
                        labels.add(pair.second);
                        return true;
                    } else if (types.isSubtype(declaredCaught, declaredThrown)) {
                        canApply = true;
                    }
                } else {
                    assert caught instanceof UnionType :
                        "caught type must be a union or a declared type";
                    UnionType caughtUnion = (UnionType)caught;
                    for (TypeMirror alternative : caughtUnion.getAlternatives()) {
                        assert alternative instanceof DeclaredType :
                            "alternatives of an caught union type must be declared types";
                        DeclaredType declaredAlt = (DeclaredType)alternative;
                        if (types.isSubtype(declaredThrown, declaredAlt)) {
                            // No later catch blocks can apply.
                            labels.add(pair.second);
                            return true;
                        } else if (types.isSubtype(declaredAlt, declaredThrown)) {
                            canApply = true;
                        }
                    }
                }

                if (canApply) {
                    labels.add(pair.second);
                }
            }

            return false;
        }
    }

    /**
     * A TryFinallyFrame applies to exceptions of any type
     */
    protected class TryFinallyFrame implements TryFrame {
        protected Label finallyLabel;

        public TryFinallyFrame(Label finallyLabel) {
            this.finallyLabel = finallyLabel;
        }

        @Override
        public boolean possibleLabels(TypeMirror thrown, Set<Label> labels) {
            labels.add(finallyLabel);
            return true;
        }
    }

    /**
     * An exception stack represents the set of all try-catch blocks
     * in effect at a given point in a program.  It maps an exception
     * type to a set of Labels and it maps a block exit (via return or
     * fall-through) to a single Label.
     */
    protected static class TryStack {
        protected Label exitLabel;
        protected LinkedList<TryFrame> frames;

        public TryStack(Label exitLabel) {
            this.exitLabel = exitLabel;
            this.frames = new LinkedList<>();
        }

        public void pushFrame(TryFrame frame) {
            frames.addFirst(frame);
        }

        public void popFrame() {
            frames.removeFirst();
        }

        /**
         * Returns the set of possible {@link Label}s where control may
         * transfer when an exception of the given type is thrown.
         */
        public Set<Label> possibleLabels(TypeMirror thrown) {
            // Work up from the innermost frame until the exception is known to
            // be caught.
            Set<Label> labels = new MostlySingleton<>();
            for (TryFrame frame : frames) {
                if (frame.possibleLabels(thrown, labels)) {
                    return labels;
                }
            }
            labels.add(exitLabel);
            return labels;
        }
    }

    /* --------------------------------------------------------- */
    /* Phase Three */
    /* --------------------------------------------------------- */

    /**
     * Class that performs phase three of the translation process. In
     * particular, the following degenerate cases of basic blocks are removed:
     *
     * <ol>
     * <li>Empty regular basic blocks: These blocks will be removed and their
     * predecessors linked directly to the successor.</li>
     * <li>Conditional basic blocks that have the same basic block as the 'then'
     * and 'else' successor: The conditional basic block will be removed in this
     * case.</li>
     * <li>Two consecutive, non-empty, regular basic blocks where the second
     * block has exactly one predecessor (namely the other of the two blocks):
     * In this case, the two blocks are merged.</li>
     * <li>Some basic blocks might not be reachable from the entryBlock. These
     * basic blocks are removed, and the list of predecessors (in the
     * doubly-linked structure of basic blocks) are adapted correctly.</li>
     * </ol>
     *
     * Eliminating the second type of degenerate cases might introduce cases of
     * the third problem. These are also removed.
     */
    public static class CFGTranslationPhaseThree {

        /**
         * A simple wrapper object that holds a basic block and allows to set
         * one of its successors.
         */
        protected interface PredecessorHolder {
            void setSuccessor(BlockImpl b);

            BlockImpl getBlock();
        }

        /**
         * Perform phase three on the control flow graph {@code cfg}.
         *
         * @param cfg
         *            The control flow graph. Ownership is transfered to this
         *            method and the caller is not allowed to read or modify
         *            {@code cfg} after the call to {@code process} any more.
         * @return the resulting control flow graph
         */
        public static ControlFlowGraph process(ControlFlowGraph cfg) {
            Set<Block> worklist = cfg.getAllBlocks();
            Set<Block> dontVisit = new HashSet<>();

            // note: this method has to be careful when relinking basic blocks
            // to not forget to adjust the predecessors, too

            // fix predecessor lists by removing any unreachable predecessors
            for (Block c : worklist) {
                BlockImpl cur = (BlockImpl) c;
                for (BlockImpl pred : new HashSet<>(cur.getPredecessors())) {
                    if (!worklist.contains(pred)) {
                        cur.removePredecessor(pred);
                    }
                }
            }

            // remove empty blocks
            for (Block cur : worklist) {
                if (dontVisit.contains(cur)) {
                    continue;
                }

                if (cur.getType() == BlockType.REGULAR_BLOCK) {
                    RegularBlockImpl b = (RegularBlockImpl) cur;
                    if (b.isEmpty()) {
                        Set<RegularBlockImpl> empty = new HashSet<>();
                        Set<PredecessorHolder> predecessors = new HashSet<>();
                        BlockImpl succ = computeNeighborhoodOfEmptyBlock(b,
                                empty, predecessors);
                        for (RegularBlockImpl e : empty) {
                            succ.removePredecessor(e);
                            dontVisit.add(e);
                        }
                        for (PredecessorHolder p : predecessors) {
                            BlockImpl block = p.getBlock();
                            dontVisit.add(block);
                            succ.removePredecessor(block);
                            p.setSuccessor(succ);
                        }
                    }
                }
            }

            // remove useless conditional blocks
            worklist = cfg.getAllBlocks();
            for (Block c : worklist) {
                BlockImpl cur = (BlockImpl) c;

                if (cur.getType() == BlockType.CONDITIONAL_BLOCK) {
                    ConditionalBlockImpl cb = (ConditionalBlockImpl) cur;
                    assert cb.getPredecessors().size() == 1;
                    if (cb.getThenSuccessor() == cb.getElseSuccessor()) {
                        BlockImpl pred = cb.getPredecessors().iterator().next();
                        PredecessorHolder predecessorHolder = getPredecessorHolder(
                                pred, cb);
                        BlockImpl succ = (BlockImpl) cb.getThenSuccessor();
                        succ.removePredecessor(cb);
                        predecessorHolder.setSuccessor(succ);
                    }
                }
            }

            // merge consecutive basic blocks if possible
            worklist = cfg.getAllBlocks();
            for (Block cur : worklist) {
                if (cur.getType() == BlockType.REGULAR_BLOCK) {
                    RegularBlockImpl b = (RegularBlockImpl) cur;
                    Block succ = b.getRegularSuccessor();
                    if (succ.getType() == BlockType.REGULAR_BLOCK) {
                        RegularBlockImpl rs = (RegularBlockImpl) succ;
                        if (rs.getPredecessors().size() == 1) {
                            b.setSuccessor(rs.getRegularSuccessor());
                            b.addNodes(rs.getContents());
                            rs.getRegularSuccessor().removePredecessor(rs);
                        }
                    }
                }
            }

            return cfg;
        }

        /**
         * Compute the set of empty regular basic blocks {@code empty}, starting
         * at {@code start} and going both forward and backwards. Furthermore,
         * compute the predecessors of these empty blocks ({@code predecessors}
         * ), and their single successor (return value).
         *
         * @param start
         *            The starting point of the search (an empty, regular basic
         *            block).
         * @param empty
         *            An empty set to be filled by this method with all empty
         *            basic blocks found (including {@code start}).
         * @param predecessors
         *            An empty set to be filled by this method with all
         *            predecessors.
         * @return the single successor of the set of the empty basic blocks
         */
        protected static BlockImpl computeNeighborhoodOfEmptyBlock(
                RegularBlockImpl start, Set<RegularBlockImpl> empty,
                Set<PredecessorHolder> predecessors) {

            // get empty neighborhood that come before 'start'
            computeNeighborhoodOfEmptyBlockBackwards(start, empty, predecessors);

            // go forward
            BlockImpl succ = (BlockImpl) start.getSuccessor();
            while (succ.getType() == BlockType.REGULAR_BLOCK) {
                RegularBlockImpl cur = (RegularBlockImpl) succ;
                if (cur.isEmpty()) {
                    computeNeighborhoodOfEmptyBlockBackwards(cur, empty,
                            predecessors);
                    assert empty.contains(cur) : "cur ought to be in empty";
                    succ = (BlockImpl) cur.getSuccessor();
                    if (succ == cur) {
                        // An infinite loop, making exit block unreachable
                        break;
                    }
                } else {
                    break;
                }
            }
            return succ;
        }

        /**
         * Compute the set of empty regular basic blocks {@code empty}, starting
         * at {@code start} and looking only backwards in the control flow
         * graph. Furthermore, compute the predecessors of these empty blocks (
         * {@code predecessors}).
         *
         * @param start
         *            The starting point of the search (an empty, regular basic
         *            block).
         * @param empty
         *            A set to be filled by this method with all empty basic
         *            blocks found (including {@code start}).
         * @param predecessors
         *            A set to be filled by this method with all predecessors.
         */
        protected static void computeNeighborhoodOfEmptyBlockBackwards(
                RegularBlockImpl start, Set<RegularBlockImpl> empty,
                Set<PredecessorHolder> predecessors) {

            RegularBlockImpl cur = start;
            empty.add(cur);
            for (final BlockImpl pred : cur.getPredecessors()) {
                switch (pred.getType()) {
                case SPECIAL_BLOCK:
                    // add pred correctly to predecessor list
                    predecessors.add(getPredecessorHolder(pred, cur));
                    break;
                case CONDITIONAL_BLOCK:
                    // add pred correctly to predecessor list
                    predecessors.add(getPredecessorHolder(pred, cur));
                    break;
                case EXCEPTION_BLOCK:
                    // add pred correctly to predecessor list
                    predecessors.add(getPredecessorHolder(pred, cur));
                    break;
                case REGULAR_BLOCK:
                    RegularBlockImpl r = (RegularBlockImpl) pred;
                    if (r.isEmpty()) {
                        // recursively look backwards
                        if (!empty.contains(r)) {
                            computeNeighborhoodOfEmptyBlockBackwards(r, empty,
                                    predecessors);
                        }
                    } else {
                        // add pred correctly to predecessor list
                        predecessors.add(getPredecessorHolder(pred, cur));
                    }
                    break;
                }
            }
        }

        /**
         * Return a predecessor holder that can be used to set the successor of
         * {@code pred} in the place where previously the edge pointed to
         * {@code cur}. Additionally, the predecessor holder also takes care of
         * unlinking (i.e., removing the {@code pred} from {@code cur's}
         * predecessors).
         */
        protected static PredecessorHolder getPredecessorHolder(
                final BlockImpl pred, final BlockImpl cur) {
            switch (pred.getType()) {
            case SPECIAL_BLOCK:
                SingleSuccessorBlockImpl s = (SingleSuccessorBlockImpl) pred;
                return singleSuccessorHolder(s, cur);
            case CONDITIONAL_BLOCK:
                // add pred correctly to predecessor list
                final ConditionalBlockImpl c = (ConditionalBlockImpl) pred;
                if (c.getThenSuccessor() == cur) {
                    return new PredecessorHolder() {
                        @Override
                        public void setSuccessor(BlockImpl b) {
                            c.setThenSuccessor(b);
                            cur.removePredecessor(pred);
                        }

                        @Override
                        public BlockImpl getBlock() {
                            return c;
                        }
                    };
                } else {
                    assert c.getElseSuccessor() == cur;
                    return new PredecessorHolder() {
                        @Override
                        public void setSuccessor(BlockImpl b) {
                            c.setElseSuccessor(b);
                            cur.removePredecessor(pred);
                        }

                        @Override
                        public BlockImpl getBlock() {
                            return c;
                        }
                    };
                }
            case EXCEPTION_BLOCK:
                // add pred correctly to predecessor list
                final ExceptionBlockImpl e = (ExceptionBlockImpl) pred;
                if (e.getSuccessor() == cur) {
                    return singleSuccessorHolder(e, cur);
                } else {
                    Set<Entry<TypeMirror, Set<Block>>> entrySet = e
                            .getExceptionalSuccessors().entrySet();
                    for (final Entry<TypeMirror, Set<Block>> entry : entrySet) {
                        if (entry.getValue().contains(cur)) {
                            return new PredecessorHolder() {
                                @Override
                                public void setSuccessor(BlockImpl b) {
                                    e.addExceptionalSuccessor(b, entry.getKey());
                                    cur.removePredecessor(pred);
                                }

                                @Override
                                public BlockImpl getBlock() {
                                    return e;
                                }
                            };
                        }
                    }
                }
                assert false;
                break;
            case REGULAR_BLOCK:
                RegularBlockImpl r = (RegularBlockImpl) pred;
                return singleSuccessorHolder(r, cur);
            }
            return null;
        }

        /**
         * @return a {@link PredecessorHolder} that sets the successor of a
         *         single successor block {@code s}.
         */
        protected static PredecessorHolder singleSuccessorHolder(
                final SingleSuccessorBlockImpl s, final BlockImpl old) {
            return new PredecessorHolder() {
                @Override
                public void setSuccessor(BlockImpl b) {
                    s.setSuccessor(b);
                    old.removePredecessor(s);
                }

                @Override
                public BlockImpl getBlock() {
                    return s;
                }
            };
        }
    }

    /* --------------------------------------------------------- */
    /* Phase Two */
    /* --------------------------------------------------------- */

    /** Tuple class with up to three members. */
    protected static class Tuple<A, B, C> {
        public A a;
        public B b;
        public C c;

        public Tuple(A a, B b) {
            this.a = a;
            this.b = b;
        }

        public Tuple(A a, B b, C c) {
            this.a = a;
            this.b = b;
            this.c = c;
        }
    }

    /**
     * Class that performs phase two of the translation process.
     */
    public class CFGTranslationPhaseTwo {

        public CFGTranslationPhaseTwo() {
        }

        /**
         * Perform phase two of the translation.
         *
         * @param in
         *            The result of phase one.
         * @return a control flow graph that might still contain degenerate
         *         basic block (such as empty regular basic blocks or
         *         conditional blocks with the same block as 'then' and 'else'
         *         sucessor)
         */
        public ControlFlowGraph process(PhaseOneResult in) {

            Map<Label, Integer> bindings = in.bindings;
            ArrayList<ExtendedNode> nodeList = in.nodeList;
            Set<Integer> leaders = in.leaders;

            assert in.nodeList.size() > 0;

            // exit blocks
            SpecialBlockImpl regularExitBlock = new SpecialBlockImpl(
                    SpecialBlockType.EXIT);
            SpecialBlockImpl exceptionalExitBlock = new SpecialBlockImpl(
                    SpecialBlockType.EXCEPTIONAL_EXIT);

            // record missing edges that will be added later
            Set<Tuple<? extends SingleSuccessorBlockImpl, Integer, ?>> missingEdges = new MostlySingleton<>();

            // missing exceptional edges
            Set<Tuple<ExceptionBlockImpl, Integer, TypeMirror>> missingExceptionalEdges = new HashSet<>();

            // create start block
            SpecialBlockImpl startBlock = new SpecialBlockImpl(
                    SpecialBlockType.ENTRY);
            missingEdges.add(new Tuple<>(startBlock, 0));

            // loop through all 'leaders' (while dynamically detecting the
            // leaders)
            RegularBlockImpl block = new RegularBlockImpl();
            int i = 0;
            for (ExtendedNode node : nodeList) {
                switch (node.getType()) {
                case NODE:
                    if (leaders.contains(i)) {
                        RegularBlockImpl b = new RegularBlockImpl();
                        block.setSuccessor(b);
                        block = b;
                    }
                    block.addNode(node.getNode());
                    node.setBlock(block);

                    // does this node end the execution (modeled as an edge to
                    // the exceptional exit block)
                    boolean terminatesExecution = node.getTerminatesExecution();
                    if (terminatesExecution) {
                        block.setSuccessor(exceptionalExitBlock);
                        block = new RegularBlockImpl();
                    }
                    break;
                case CONDITIONAL_JUMP: {
                    ConditionalJump cj = (ConditionalJump) node;
                    // Exception nodes may fall through to conditional jumps,
                    // so we set the block which is required for the insertion
                    // of missing edges.
                    node.setBlock(block);
                    assert block != null;
                    final ConditionalBlockImpl cb = new ConditionalBlockImpl();
                    if (cj.getTrueFlowRule() != null) {
                        cb.setThenFlowRule(cj.getTrueFlowRule());
                    }
                    if (cj.getFalseFlowRule() != null) {
                        cb.setElseFlowRule(cj.getFalseFlowRule());
                    }
                    block.setSuccessor(cb);
                    block = new RegularBlockImpl();
                    // use two anonymous SingleSuccessorBlockImpl that set the
                    // 'then' and 'else' successor of the conditional block
                    final Label thenLabel = cj.getThenLabel();
                    final Label elseLabel = cj.getElseLabel();
                    missingEdges.add(new Tuple<>(
                            new SingleSuccessorBlockImpl() {
                                @Override
                                public void setSuccessor(BlockImpl successor) {
                                    cb.setThenSuccessor(successor);
                                }
                            }, bindings.get(thenLabel)));
                    missingEdges.add(new Tuple<>(
                            new SingleSuccessorBlockImpl() {
                                @Override
                                public void setSuccessor(BlockImpl successor) {
                                    cb.setElseSuccessor(successor);
                                }
                            }, bindings.get(elseLabel)));
                    break;
                }
                case UNCONDITIONAL_JUMP:
                    if (leaders.contains(i)) {
                        RegularBlockImpl b = new RegularBlockImpl();
                        block.setSuccessor(b);
                        block = b;
                    }
                    node.setBlock(block);
                    if (node.getLabel() == regularExitLabel) {
                        block.setSuccessor(regularExitBlock);
                    } else if (node.getLabel() == exceptionalExitLabel) {
                        block.setSuccessor(exceptionalExitBlock);
                    } else {
                        missingEdges.add(new Tuple<>(block, bindings.get(node
                                .getLabel())));
                    }
                    block = new RegularBlockImpl();
                    break;
                case EXCEPTION_NODE:
                    NodeWithExceptionsHolder en = (NodeWithExceptionsHolder) node;
                    // create new exception block and link with previous block
                    ExceptionBlockImpl e = new ExceptionBlockImpl();
                    Node nn = en.getNode();
                    e.setNode(nn);
                    node.setBlock(e);
                    block.setSuccessor(e);
                    block = new RegularBlockImpl();

                    // ensure linking between e and next block (normal edge)
                    // Note: do not link to the next block for throw statements
                    // (these throw exceptions for sure)
                    if (!node.getTerminatesExecution()) {
                        missingEdges.add(new Tuple<>(e, i + 1));
                    }

                    // exceptional edges
                    for (Entry<TypeMirror, Set<Label>> entry : en.getExceptions()
                            .entrySet()) {
                        TypeMirror cause = entry.getKey();
                        for (Label label : entry.getValue()) {
                            Integer target = bindings.get(label);
                            missingExceptionalEdges
                                .add(new Tuple<ExceptionBlockImpl, Integer, TypeMirror>(
                                        e, target, cause));
                        }
                    }
                    break;
                }
                i++;
            }

            // add missing edges
            for (Tuple<? extends SingleSuccessorBlockImpl, Integer, ?> p : missingEdges) {
                Integer index = p.b;
                ExtendedNode extendedNode = nodeList.get(index);
                BlockImpl target = extendedNode.getBlock();
                SingleSuccessorBlockImpl source = p.a;
                source.setSuccessor(target);
            }

            // add missing exceptional edges
            for (Tuple<ExceptionBlockImpl, Integer, ?> p : missingExceptionalEdges) {
                Integer index = p.b;
                TypeMirror cause = (TypeMirror) p.c;
                ExceptionBlockImpl source = p.a;
                if (index == null) {
                    // edge to exceptional exit
                    source.addExceptionalSuccessor(exceptionalExitBlock, cause);
                } else {
                    // edge to specific target
                    ExtendedNode extendedNode = nodeList.get(index);
                    BlockImpl target = extendedNode.getBlock();
                    source.addExceptionalSuccessor(target, cause);
                }
            }

            return new ControlFlowGraph(startBlock, regularExitBlock, exceptionalExitBlock, in.underlyingAST,
                    in.treeLookupMap, in.convertedTreeLookupMap, in.returnNodes);
        }
    }

    /* --------------------------------------------------------- */
    /* Phase One */
    /* --------------------------------------------------------- */

    /**
     * A wrapper object to pass around the result of phase one. For a
     * documentation of the fields see {@link CFGTranslationPhaseOne}.
     */
    protected static class PhaseOneResult {

        private final IdentityHashMap<Tree, Node> treeLookupMap;
        private final IdentityHashMap<Tree, Node> convertedTreeLookupMap;
        private final UnderlyingAST underlyingAST;
        private final Map<Label, Integer> bindings;
        private final ArrayList<ExtendedNode> nodeList;
        private final Set<Integer> leaders;
        private final List<ReturnNode> returnNodes;

        public PhaseOneResult(UnderlyingAST underlyingAST,
                IdentityHashMap<Tree, Node> treeLookupMap,
                IdentityHashMap<Tree, Node> convertedTreeLookupMap,
                ArrayList<ExtendedNode> nodeList, Map<Label, Integer> bindings,
                Set<Integer> leaders, List<ReturnNode> returnNodes) {
            this.underlyingAST = underlyingAST;
            this.treeLookupMap = treeLookupMap;
            this.convertedTreeLookupMap = convertedTreeLookupMap;
            this.nodeList = nodeList;
            this.bindings = bindings;
            this.leaders = leaders;
            this.returnNodes = returnNodes;
        }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            for (ExtendedNode n : nodeList) {
                sb.append(nodeToString(n));
                sb.append("\n");
            }
            return sb.toString();
        }

        protected String nodeToString(ExtendedNode n) {
            if (n.getType() == ExtendedNodeType.CONDITIONAL_JUMP) {
                ConditionalJump t = (ConditionalJump) n;
                return "TwoTargetConditionalJump("
                        + resolveLabel(t.getThenLabel()) + ","
                        + resolveLabel(t.getElseLabel()) + ")";
            } else if (n.getType() == ExtendedNodeType.UNCONDITIONAL_JUMP) {
                return "UnconditionalJump(" + resolveLabel(n.getLabel()) + ")";
            } else {
                return n.toString();
            }
        }

        private String resolveLabel(Label label) {
            Integer index = bindings.get(label);
            if (index == null) {
                return "null";
            }
            return nodeToString(nodeList.get(index));
        }

    }

    /**
     * Class that performs phase one of the translation process. It generates
     * the following information:
     * <ul>
     * <li>A sequence of extended nodes.</li>
     * <li>A set of bindings from {@link Label}s to positions in the node
     * sequence.</li>
     * <li>A set of leader nodes that give rise to basic blocks in phase two.</li>
     * <li>A lookup map that gives the mapping from AST tree nodes to
     * {@link Node}s.</li>
     * </ul>
     *
     * <p>
     *
     * The return type of this scanner is {@link Node}. For expressions, the
     * corresponding node is returned to allow linking between different nodes.
     *
     * However, for statements there is usually no single {@link Node} that is
     * created, and thus no node is returned (rather, null is returned).
     *
     * <p>
     *
     * Every {@code visit*} method is assumed to add at least one extended node
     * to the list of nodes (which might only be a jump).
     *
     */
    public class CFGTranslationPhaseOne extends TreePathScanner<Node, Void> {

        public CFGTranslationPhaseOne() {
        }

        /**
         * Annotation processing environment and its associated type and tree
         * utilities.
         */
        protected ProcessingEnvironment env;
        protected Elements elements;
        protected Types types;
        protected Trees trees;
        protected TreeBuilder treeBuilder;
        protected AnnotationProvider annotationProvider;

        /**
         * Current {@link Label} to which a break statement with no label should
         * jump, or null if there is no valid destination.
         */
        protected /*@Nullable*/ Label breakTargetL;

        /**
         * Map from AST label Names to CFG {@link Label}s for breaks. Each
         * labeled statement creates two CFG {@link Label}s, one for break and
         * one for continue.
         */
        protected Map<Name, Label> breakLabels;

        /**
         * Current {@link Label} to which a continue statement with no label
         * should jump, or null if there is no valid destination.
         */
        protected /*@Nullable*/ Label continueTargetL;

        /**
         * Map from AST label Names to CFG {@link Label}s for continues. Each
         * labeled statement creates two CFG {@link Label}s, one for break and
         * one for continue.
         */
        protected Map<Name, Label> continueLabels;

        /**
         * Maps from AST {@link Tree}s to {@link Node}s.  Every Tree that produces
         * a value will have at least one corresponding Node.  Trees
         * that undergo conversions, such as boxing or unboxing, can map to two
         * distinct Nodes.  The Node for the pre-conversion value is stored
         * in the treeLookupMap, while the Node for the post-conversion value
         * is stored in the convertedTreeLookupMap.
         */
        protected IdentityHashMap<Tree, Node> treeLookupMap;

        /** Map from AST {@link Tree}s to post-conversion {@link Node}s. */
        protected IdentityHashMap<Tree, Node> convertedTreeLookupMap;

        /** The list of extended nodes. */
        protected ArrayList<ExtendedNode> nodeList;

        /**
         * The bindings of labels to positions (i.e., indices) in the
         * {@code nodeList}.
         */
        protected Map<Label, Integer> bindings;

        /** The set of leaders (represented as indices into {@code nodeList}). */
        protected Set<Integer> leaders;

        /**
         * All return nodes (if any) encountered. Only includes return
         * statements that actually return something
         */
        private List<ReturnNode> returnNodes;

        /**
         * Nested scopes of try-catch blocks in force at the current
         * program point.
         */
        private TryStack tryStack;

        /**
         * Performs the actual work of phase one.
         *
         * @param bodyPath
         *            path to the body of the underlying AST's method
         * @param env
         *            annotation processing environment containing type
         *            utilities
         * @param underlyingAST
         *            the AST for which the CFG is to be built
         * @param exceptionalExitLabel
         *            the label for exceptional exits from the CFG
         * @param treeBuilder
         *            builder for new AST nodes
         * @param annotationProvider
         *            extracts annotations from AST nodes
         * @return the result of phase one
         */
        public PhaseOneResult process(
                TreePath bodyPath, ProcessingEnvironment env,
                UnderlyingAST underlyingAST, Label exceptionalExitLabel,
                TreeBuilder treeBuilder, AnnotationProvider annotationProvider) {
            this.env = env;
            this.tryStack = new TryStack(exceptionalExitLabel);
            this.treeBuilder = treeBuilder;
            this.annotationProvider = annotationProvider;
            elements = env.getElementUtils();
            types = env.getTypeUtils();

            // initialize lists and maps
            treeLookupMap = new IdentityHashMap<>();
            convertedTreeLookupMap = new IdentityHashMap<>();
            nodeList = new ArrayList<>();
            bindings = new HashMap<>();
            leaders = new HashSet<>();
            breakLabels = new HashMap<>();
            continueLabels = new HashMap<>();
            returnNodes = new ArrayList<>();

            // traverse AST of the method body
            scan(bodyPath, null);

            // add marker to indicate that the next block will be the exit block
            // Note: if there is a return statement earlier in the method (which
            // is always the case for non-void methods), then this is not
            // strictly necessary. However, it is also not a problem, as it will
            // just generate a degenerated control graph case that will be
            // removed in a later phase.
            nodeList.add(new UnconditionalJump(regularExitLabel));

            return new PhaseOneResult(underlyingAST, treeLookupMap,
                    convertedTreeLookupMap, nodeList,
                    bindings, leaders, returnNodes);
        }

        public PhaseOneResult process(
                CompilationUnitTree root, ProcessingEnvironment env,
                UnderlyingAST underlyingAST, Label exceptionalExitLabel,
                TreeBuilder treeBuilder, AnnotationProvider annotationProvider) {
            trees = Trees.instance(env);
            TreePath bodyPath = trees.getPath(root, underlyingAST.getCode());
            return process(bodyPath, env, underlyingAST, exceptionalExitLabel, treeBuilder, annotationProvider);
        }

        /**
         * Perform any actions required when CFG translation creates a
         * new Tree that is not part of the original AST.
         *
         * @param tree  the newly created Tree
         */
        public void handleArtificialTree(Tree tree) {}

        /* --------------------------------------------------------- */
        /* Nodes and Labels Management */
        /* --------------------------------------------------------- */

        /**
         * Add a node to the lookup map if it not already present.
         *
         * @param node
         *            The node to add to the lookup map.
         */
        protected void addToLookupMap(Node node) {
            Tree tree = node.getTree();
            if (tree == null) {
                return;
            }
            if (!treeLookupMap.containsKey(tree)) {
                treeLookupMap.put(tree, node);
            }

            Tree enclosingParens = parenMapping.get(tree);
            while (enclosingParens != null) {
                treeLookupMap.put(enclosingParens, node);
                enclosingParens = parenMapping.get(enclosingParens);
            }
        }

        /**
         * Add a node in the post-conversion lookup map. The node
         * should refer to a Tree and that Tree should already be in
         * the pre-conversion lookup map. This method is used to
         * update the Tree-Node mapping with conversion nodes.
         *
         * @param node
         *            The node to add to the lookup map.
         */
        protected void addToConvertedLookupMap(Node node) {
            Tree tree = node.getTree();
            addToConvertedLookupMap(tree, node);
        }

        /**
         * Add a node in the post-conversion lookup map. The tree
         * argument should already be in the pre-conversion lookup
         * map. This method is used to update the Tree-Node mapping
         * with conversion nodes.
         *
         * @param tree
         *            The tree used as a key in the map.
         * @param node
         *            The node to add to the lookup map.
         */
        protected void addToConvertedLookupMap(Tree tree, Node node) {
            assert tree != null;
            assert treeLookupMap.containsKey(tree);
            convertedTreeLookupMap.put(tree, node);
        }

        /**
         * Extend the list of extended nodes with a node.
         *
         * @param node
         *            The node to add.
         * @return the same node (for convenience)
         */
        protected <T extends Node> T extendWithNode(T node) {
            addToLookupMap(node);
            extendWithExtendedNode(new NodeHolder(node));
            return node;
        }

        /**
         * Extend the list of extended nodes with a node, where
         * {@code node} might throw the exception {@code cause}.
         *
         * @param node
         *            The node to add.
         * @param cause
         *            An exception that the node might throw.
         * @return the node holder
         */
        protected NodeWithExceptionsHolder extendWithNodeWithException(Node node, TypeMirror cause) {
            addToLookupMap(node);
            return extendWithNodeWithExceptions(node, Collections.singleton(cause));
        }

        /**
         * Extend the list of extended nodes with a node, where
         * {@code node} might throw any of the exception in
         * {@code causes}.
         *
         * @param node
         *            The node to add.
         * @param causes
         *            Set of exceptions that the node might throw.
         * @return the node holder
         */
        protected NodeWithExceptionsHolder extendWithNodeWithExceptions(Node node,
                Set<TypeMirror> causes) {
            addToLookupMap(node);
            Map<TypeMirror, Set<Label>> exceptions = new HashMap<>();
            for (TypeMirror cause : causes) {
                exceptions.put(cause, tryStack.possibleLabels(cause));
            }
            NodeWithExceptionsHolder exNode = new NodeWithExceptionsHolder(
                    node, exceptions);
            extendWithExtendedNode(exNode);
            return exNode;
        }

        /**
         * Insert {@code node} after {@code pred} in
         * the list of extended nodes, or append to the list if
         * {@code pred} is not present.
         *
         * @param node
         *            The node to add.
         * @param pred
         *            The desired predecessor of node.
         * @return the node holder
         */
        protected <T extends Node> T insertNodeAfter(T node, Node pred) {
            addToLookupMap(node);
            insertExtendedNodeAfter(new NodeHolder(node), pred);
            return node;
        }

        /**
         * Insert a {@code node} that might throw the exception
         * {@code cause} after {@code pred} in the list of
         * extended nodes, or append to the list if {@code pred}
         * is not present.
         *
         * @param node
         *            The node to add.
         * @param causes
         *            Set of exceptions that the node might throw.
         * @param pred
         *            The desired predecessor of node.
         * @return the node holder
         */
        protected NodeWithExceptionsHolder insertNodeWithExceptionsAfter(Node node,
                Set<TypeMirror> causes, Node pred) {
            addToLookupMap(node);
            Map<TypeMirror, Set<Label>> exceptions = new HashMap<>();
            for (TypeMirror cause : causes) {
                exceptions.put(cause, tryStack.possibleLabels(cause));
            }
            NodeWithExceptionsHolder exNode = new NodeWithExceptionsHolder(
                    node, exceptions);
            insertExtendedNodeAfter(exNode, pred);
            return exNode;
        }

        /**
         * Extend the list of extended nodes with an extended node.
         *
         * @param n
         *            The extended node.
         */
        protected void extendWithExtendedNode(ExtendedNode n) {
            nodeList.add(n);
        }

        /**
         * Insert {@code n} after the node {@code pred} in the
         * list of extended nodes, or append {@code n} if {@code pred}
         * is not present.
         *
         * @param n
         *            The extended node.
         * @param pred
         *            The desired predecessor.
         */
        protected void insertExtendedNodeAfter(ExtendedNode n, Node pred) {
            int index = -1;
            for (int i = 0; i < nodeList.size(); i++) {
                ExtendedNode inList = nodeList.get(i);
                if (inList instanceof NodeHolder ||
                    inList instanceof NodeWithExceptionsHolder) {
                    if (inList.getNode() == pred) {
                        index = i;
                        break;
                    }
                }
            }
            if (index != -1) {
                nodeList.add(index + 1, n);
                // update bindings
                for (Entry<Label, Integer> e : bindings.entrySet()) {
                    if (e.getValue() >= index+1) {
                        bindings.put(e.getKey(), e.getValue() + 1);
                    }
                }
                // update leaders
                Set<Integer> newLeaders = new HashSet<>();
                for (Integer l : leaders) {
                    if (l >= index+1) {
                        newLeaders.add(l+1);
                    } else {
                        newLeaders.add(l);
                    }
                }
                leaders = newLeaders;
            } else {
                nodeList.add(n);
            }
        }

        /**
         * Add the label {@code l} to the extended node that will be placed next
         * in the sequence.
         */
        protected void addLabelForNextNode(Label l) {
            leaders.add(nodeList.size());
            bindings.put(l, nodeList.size());
        }

        /* --------------------------------------------------------- */
        /* Utility Methods */
        /* --------------------------------------------------------- */

        protected long uid = 0;
        protected String uniqueName(String prefix) {
            return prefix + "#num" + uid++;
        }

        /**
         * If the input node is an unboxed primitive type, insert a call to the
         * appropriate valueOf method, otherwise leave it alone.
         *
         * @param node
         *            in input node
         * @return a Node representing the boxed version of the input, which may
         *         simply be the input node
         */
        protected Node box(Node node) {
            // For boxing conversion, see JLS 5.1.7
            if (TypesUtils.isPrimitive(node.getType())) {
                PrimitiveType primitive = types.getPrimitiveType(node.getType()
                        .getKind());
                TypeMirror boxedType = types.getDeclaredType(types
                        .boxedClass(primitive));

                TypeElement boxedElement = (TypeElement)((DeclaredType)boxedType).asElement();
                IdentifierTree classTree = treeBuilder.buildClassUse(boxedElement);
                handleArtificialTree(classTree);
                ClassNameNode className = new ClassNameNode(classTree);
                className.setInSource(false);
                insertNodeAfter(className, node);

                MemberSelectTree valueOfSelect = treeBuilder.buildValueOfMethodAccess(classTree);
                handleArtificialTree(valueOfSelect);
                MethodAccessNode valueOfAccess = new MethodAccessNode(valueOfSelect, className);
                valueOfAccess.setInSource(false);
                insertNodeAfter(valueOfAccess, className);

                MethodInvocationTree valueOfCall =
                    treeBuilder.buildMethodInvocation(valueOfSelect, (ExpressionTree)node.getTree());
                handleArtificialTree(valueOfCall);
                Node boxed = new MethodInvocationNode(valueOfCall, valueOfAccess,
                                                      Collections.singletonList(node),
                                                      getCurrentPath());
                boxed.setInSource(false);
                // Add Throwable to account for unchecked exceptions
                TypeElement throwableElement = elements
                    .getTypeElement("java.lang.Throwable");
                addToConvertedLookupMap(node.getTree(), boxed);
                insertNodeWithExceptionsAfter(boxed,
                        Collections.singleton(throwableElement.asType()), valueOfAccess);
                return boxed;
            } else {
                return node;
            }
        }

        /**
         * If the input node is a boxed type, unbox it, otherwise leave it
         * alone.
         *
         * @param node
         *            in input node
         * @return a Node representing the unboxed version of the input, which
         *         may simply be the input node
         */
        protected Node unbox(Node node) {
            if (TypesUtils.isBoxedPrimitive(node.getType())) {

                MemberSelectTree primValueSelect =
                    treeBuilder.buildPrimValueMethodAccess(node.getTree());
                handleArtificialTree(primValueSelect);
                MethodAccessNode primValueAccess = new MethodAccessNode(primValueSelect, node);
                primValueAccess.setInSource(false);
                // Method access may throw NullPointerException
                TypeElement npeElement = elements
                    .getTypeElement("java.lang.NullPointerException");
                insertNodeWithExceptionsAfter(primValueAccess,
                        Collections.singleton(npeElement.asType()), node);

                MethodInvocationTree primValueCall =
                    treeBuilder.buildMethodInvocation(primValueSelect);
                handleArtificialTree(primValueCall);
                Node unboxed = new MethodInvocationNode(primValueCall, primValueAccess,
                                                        Collections.<Node>emptyList(),
                                                        getCurrentPath());
                unboxed.setInSource(false);

                // Add Throwable to account for unchecked exceptions
                TypeElement throwableElement = elements
                    .getTypeElement("java.lang.Throwable");
                addToConvertedLookupMap(node.getTree(), unboxed);
                insertNodeWithExceptionsAfter(unboxed,
                        Collections.singleton(throwableElement.asType()), primValueAccess);
                return unboxed;
            } else {
                return node;
            }
        }

        private TreeInfo getTreeInfo(Tree tree) {
            final TypeMirror type = InternalUtils.typeOf(tree);
            final boolean boxed = TypesUtils.isBoxedPrimitive(type);
            final TypeMirror unboxedType = boxed ? types.unboxedType(type) : type;

            final boolean bool = TypesUtils.isBooleanType(type);
            final boolean numeric = TypesUtils.isNumeric(unboxedType);

            return new TreeInfo() {
                @Override
                public boolean isNumeric() {
                    return numeric;
                }

                @Override
                public boolean isBoxed() {
                    return boxed;
                }

                @Override
                public boolean isBoolean() {
                    return bool;
                }

                @Override
                public TypeMirror unboxedType() {
                    return unboxedType;
                }
            };
        }

        /**
         * @return the unboxed tree if necessary, as described in JLS 5.1.8
         */
        private Node unboxAsNeeded(Node node, boolean boxed) {
            return boxed ? unbox(node) : node;
        }

        /**
         * Convert the input node to String type, if it isn't already.
         *
         * @param node
         *            an input node
         * @return a Node with the value promoted to String, which may be the
         *         input node
         */
        protected Node stringConversion(Node node) {
            // For string conversion, see JLS 5.1.11
            TypeElement stringElement =
                elements.getTypeElement("java.lang.String");
            if (!TypesUtils.isString(node.getType())) {
                Node converted = new StringConversionNode(node.getTree(), node,
                        stringElement.asType());
                addToConvertedLookupMap(converted);
                insertNodeAfter(converted, node);
                return converted;
            } else {
                return node;
            }
        }

        /**
         * Perform unary numeric promotion on the input node.
         *
         * @param node
         *            a node producing a value of numeric primitive or boxed
         *            type
         * @return a Node with the value promoted to the int, long float or
         *         double, which may be the input node
         */
        protected Node unaryNumericPromotion(Node node) {
            // For unary numeric promotion, see JLS 5.6.1
            node = unbox(node);

            switch (node.getType().getKind()) {
            case BYTE:
            case CHAR:
            case SHORT: {
                TypeMirror intType = types.getPrimitiveType(TypeKind.INT);
                Node widened = new WideningConversionNode(node.getTree(), node, intType);
                addToConvertedLookupMap(widened);
                insertNodeAfter(widened, node);
                return widened;
            }
            default:
                // Nothing to do.
                break;
            }

            return node;
        }

        /**
         * Returns true if the argument type is a numeric primitive or
         * a boxed numeric primitive and false otherwise.
         */
        protected boolean isNumericOrBoxed(TypeMirror type) {
            if (TypesUtils.isBoxedPrimitive(type)) {
                type = types.unboxedType(type);
            }
            return TypesUtils.isNumeric(type);
        }

        /**
         * Compute the type to which two numeric types must be promoted
         * before performing a binary numeric operation on them.  The
         * input types must both be numeric and the output type is primitive.
         *
         * @param left   the type of the left operand
         * @param right  the type of the right operand
         * @return a TypeMirror representing the binary numeric promoted type
         */
        protected TypeMirror binaryPromotedType(TypeMirror left, TypeMirror right) {
            if (TypesUtils.isBoxedPrimitive(left)) {
                left = types.unboxedType(left);
            }
            if (TypesUtils.isBoxedPrimitive(right)) {
                right = types.unboxedType(right);
            }
            TypeKind promotedTypeKind = TypesUtils.widenedNumericType(left, right);
            return types.getPrimitiveType(promotedTypeKind);
        }

        /**
         * Perform binary numeric promotion on the input node to make it match
         * the expression type.
         *
         * @param node
         *            a node producing a value of numeric primitive or boxed
         *            type
         * @param exprType
         *            the type to promote the value to
         * @return a Node with the value promoted to the exprType, which may be
         *         the input node
         */
        protected Node binaryNumericPromotion(Node node, TypeMirror exprType) {
            // For binary numeric promotion, see JLS 5.6.2
            node = unbox(node);

            if (!types.isSameType(node.getType(), exprType)) {
                Node widened = new WideningConversionNode(node.getTree(), node,
                        exprType);
                addToConvertedLookupMap(widened);
                insertNodeAfter(widened, node);
                return widened;
            } else {
                return node;
            }
        }

        /**
         * Perform widening primitive conversion on the input node to make it
         * match the destination type.
         *
         * @param node
         *            a node producing a value of numeric primitive type
         * @param destType
         *            the type to widen the value to
         * @return a Node with the value widened to the exprType, which may be
         *         the input node
         */
        protected Node widen(Node node, TypeMirror destType) {
            // For widening conversion, see JLS 5.1.2
            assert TypesUtils.isPrimitive(node.getType())
                    && TypesUtils.isPrimitive(destType) : "widening must be applied to primitive types";
            if (types.isSubtype(node.getType(), destType)
                    && !types.isSameType(node.getType(), destType)) {
                Node widened = new WideningConversionNode(node.getTree(), node,
                        destType);
                addToConvertedLookupMap(widened);
                insertNodeAfter(widened, node);
                return widened;
            } else {
                return node;
            }
        }

        /**
         * Perform narrowing conversion on the input node to make it match the
         * destination type.
         *
         * @param node
         *            a node producing a value of numeric primitive type
         * @param destType
         *            the type to narrow the value to
         * @return a Node with the value narrowed to the exprType, which may be
         *         the input node
         */
        protected Node narrow(Node node, TypeMirror destType) {
            // For narrowing conversion, see JLS 5.1.3
            assert TypesUtils.isPrimitive(node.getType())
                    && TypesUtils.isPrimitive(destType) : "narrowing must be applied to primitive types";
            if (types.isSubtype(destType, node.getType())
                    && !types.isSameType(destType, node.getType())) {
                Node narrowed = new NarrowingConversionNode(node.getTree(), node,
                        destType);
                addToConvertedLookupMap(narrowed);
                insertNodeAfter(narrowed, node);
                return narrowed;
            } else {
                return node;
            }
        }

        /**
         * Perform narrowing conversion and optionally boxing conversion on the
         * input node to make it match the destination type.
         *
         * @param node
         *            a node producing a value of numeric primitive type
         * @param destType
         *            the type to narrow the value to (possibly boxed)
         * @return a Node with the value narrowed and boxed to the destType,
         *         which may be the input node
         */
        protected Node narrowAndBox(Node node, TypeMirror destType) {
            if (TypesUtils.isBoxedPrimitive(destType)) {
                return box(narrow(node, types.unboxedType(destType)));
            } else {
                return narrow(node, destType);
            }
        }


        /**
         * Return whether a conversion from the type of the node to varType
         * requires narrowing.
         *
         * @param varType  the type of a variable (or general LHS) to be converted to
         * @param node     a node whose value is being converted
         * @return  whether this conversion requires narrowing to succeed
         */
        protected boolean conversionRequiresNarrowing(TypeMirror varType, Node node) {
            // Narrowing is restricted to cases where the left hand side
            // is byte, char, short or Byte, Char, Short and the right
            // hand side is a constant.
            TypeMirror unboxedVarType = TypesUtils.isBoxedPrimitive(varType) ? types
                .unboxedType(varType) : varType;
            TypeKind unboxedVarKind = unboxedVarType.getKind();
            boolean isLeftNarrowableTo = unboxedVarKind == TypeKind.BYTE
                || unboxedVarKind == TypeKind.SHORT
                || unboxedVarKind == TypeKind.CHAR;
            boolean isRightConstant = node instanceof ValueLiteralNode;
            return isLeftNarrowableTo && isRightConstant;
        }


        /**
         * Assignment conversion and method invocation conversion are almost
         * identical, except that assignment conversion allows narrowing. We
         * factor out the common logic here.
         *
         * @param node
         *            a Node producing a value
         * @param varType
         *            the type of a variable
         * @param contextAllowsNarrowing
         *            whether to allow narrowing (for assignment conversion) or
         *            not (for method invocation conversion)
         * @return a Node with the value converted to the type of the variable,
         *         which may be the input node itself
         */
        protected Node commonConvert(Node node, TypeMirror varType,
                boolean contextAllowsNarrowing) {
            // For assignment conversion, see JLS 5.2
            // For method invocation conversion, see JLS 5.3

            // Check for identical types or "identity conversion"
            TypeMirror nodeType = node.getType();
            boolean isSameType = types.isSameType(nodeType, varType);
            if (isSameType) {
                return node;
            }

            boolean isRightNumeric = TypesUtils.isNumeric(nodeType);
            boolean isRightPrimitive = TypesUtils.isPrimitive(nodeType);
            boolean isRightBoxed = TypesUtils.isBoxedPrimitive(nodeType);
            boolean isRightReference = nodeType instanceof ReferenceType;
            boolean isLeftNumeric = TypesUtils.isNumeric(varType);
            boolean isLeftPrimitive = TypesUtils.isPrimitive(varType);
            // boolean isLeftBoxed = TypesUtils.isBoxedPrimitive(varType);
            boolean isLeftReference = varType instanceof ReferenceType;
            boolean isSubtype = types.isSubtype(nodeType, varType);

            if (isRightNumeric && isLeftNumeric && isSubtype) {
                node = widen(node, varType);
                nodeType = node.getType();
            } else if (isRightReference && isLeftReference && isSubtype) {
                // widening reference conversion is a no-op, but if it
                // applies, then later conversions do not.
            } else if (isRightPrimitive && isLeftReference) {
                if (contextAllowsNarrowing && conversionRequiresNarrowing(varType, node)) {
                    node = narrowAndBox(node, varType);
                    nodeType = node.getType();
                } else {
                    node = box(node);
                    nodeType = node.getType();
                }
            } else if (isRightBoxed && isLeftPrimitive) {
                node = unbox(node);
                nodeType = node.getType();

                if (types.isSubtype(nodeType, varType)
                        && !types.isSameType(nodeType, varType)) {
                    node = widen(node, varType);
                    nodeType = node.getType();
                }
            } else if (isRightPrimitive && isLeftPrimitive) {
                if (contextAllowsNarrowing && conversionRequiresNarrowing(varType, node)) {
                    node = narrow(node, varType);
                    nodeType = node.getType();
                }
            }

            // TODO: if checkers need to know about null references of
            // a particular type, add logic for them here.

            return node;
        }

        /**
         * Perform assignment conversion so that it can be assigned to a
         * variable of the given type.
         *
         * @param node
         *            a Node producing a value
         * @param varType
         *            the type of a variable
         * @return a Node with the value converted to the type of the variable,
         *         which may be the input node itself
         */
        protected Node assignConvert(Node node, TypeMirror varType) {
            return commonConvert(node, varType, true);
        }

        /**
         * Perform method invocation conversion so that the node can be passed
         * as a formal parameter of the given type.
         *
         * @param node
         *            a Node producing a value
         * @param formalType
         *            the type of a formal parameter
         * @return a Node with the value converted to the type of the formal,
         *         which may be the input node itself
         */
        protected Node methodInvocationConvert(Node node, TypeMirror formalType) {
            return commonConvert(node, formalType, false);
        }

        /**
         * Given a method element and as list of argument expressions, return a
         * list of {@link Node}s representing the arguments converted for a call
         * of the method. This method applies to both method invocations and
         * constructor calls.
         *
         * @param method
         *            an ExecutableElement representing a method to be called
         * @param actualExprs
         *            a List of argument expressions to a call
         * @return a List of {@link Node}s representing arguments after
         *         conversions required by a call to this method.
         */
        protected List<Node> convertCallArguments(ExecutableElement method,
                List<? extends ExpressionTree> actualExprs) {
            // NOTE: It is important to convert one method argument before
            // generating CFG nodes for the next argument, since label binding
            // expects nodes to be generated in execution order.  Therefore,
            // this method first determines which conversions need to be applied
            // and then iterates over the actual arguments.
            List<? extends VariableElement> formals = method.getParameters();

            ArrayList<Node> convertedNodes = new ArrayList<Node>();

            int numFormals = formals.size();
            int numActuals = actualExprs.size();
            if (method.isVarArgs()) {
                // Create a new array argument if the actuals outnumber
                // the formals, or if the last actual is not assignable
                // to the last formal.
                int lastArgIndex = numFormals - 1;
                TypeMirror lastParamType = formals.get(lastArgIndex).asType();
                List<Node> dimensions = new ArrayList<>();
                List<Node> initializers = new ArrayList<>();

                if (numActuals == numFormals - 1) {
                    // Apply method invocation conversion to all actual
                    // arguments, then create and append an empty array
                    for (int i = 0; i < numActuals; i++) {
                        Node actualVal = scan(actualExprs.get(i), null);
                        convertedNodes.add(methodInvocationConvert(actualVal,
                                formals.get(i).asType()));
                    }

                    Node lastArgument = new ArrayCreationNode(null,
                            lastParamType, dimensions, initializers);
                    extendWithNode(lastArgument);

                    convertedNodes.add(lastArgument);
                } else {
                    TypeMirror actualType = InternalUtils.typeOf(actualExprs
                            .get(lastArgIndex));
                    if (numActuals == numFormals
                            && types.isAssignable(actualType, lastParamType)) {
                        // Normal call with no array creation, apply method
                        // invocation conversion to all arguments.
                        for (int i = 0; i < numActuals; i++) {
                            Node actualVal = scan(actualExprs.get(i), null);
                            convertedNodes.add(methodInvocationConvert(actualVal,
                                    formals.get(i).asType()));
                        }
                    } else {
                        assert lastParamType instanceof ArrayType :
                            "variable argument formal must be an array";
                        // Apply method invocation conversion to lastArgIndex
                        // arguments and use the remaining ones to initialize
                        // an array.
                        for (int i = 0; i < lastArgIndex; i++) {
                            Node actualVal = scan(actualExprs.get(i), null);
                            convertedNodes.add(methodInvocationConvert(actualVal,
                                    formals.get(i).asType()));
                        }

                        TypeMirror elemType =
                            ((ArrayType)lastParamType).getComponentType();
                        for (int i = lastArgIndex; i < numActuals; i++) {
                            Node actualVal = scan(actualExprs.get(i), null);
                            initializers.add(assignConvert(actualVal, elemType));
                        }

                        Node lastArgument = new ArrayCreationNode(null,
                                lastParamType, dimensions, initializers);
                        extendWithNode(lastArgument);
                        convertedNodes.add(lastArgument);
                    }
                }
            } else {
                for (int i = 0; i < numActuals; i++) {
                    Node actualVal = scan(actualExprs.get(i), null);
                    convertedNodes.add(methodInvocationConvert(actualVal,
                            formals.get(i).asType()));
                }
            }

            return convertedNodes;
        }

        /**
         * Convert an operand of a conditional expression to the type of the
         * whole expression.
         *
         * @param node
         *            a node occurring as the second or third operand of
         *            a conditional expression
         * @param destType
         *            the type to promote the value to
         * @return a Node with the value promoted to the destType, which may be
         *         the input node
         */
        protected Node conditionalExprPromotion(Node node, TypeMirror destType) {
            // For rules on converting operands of conditional expressions,
            // JLS 15.25
            TypeMirror nodeType = node.getType();

            // If the operand is already the same type as the whole
            // expression, then do nothing.
            if (types.isSameType(nodeType, destType)) {
                return node;
            }

            // If the operand is a primitive and the whole expression is
            // boxed, then apply boxing.
            if (TypesUtils.isPrimitive(nodeType) &&
                TypesUtils.isBoxedPrimitive(destType)) {
                return box(node);
            }

            // If the operand is byte or Byte and the whole expression is
            // short, then convert to short.
            boolean isBoxedPrimitive = TypesUtils.isBoxedPrimitive(nodeType);
            TypeMirror unboxedNodeType =
                isBoxedPrimitive ? types.unboxedType(nodeType) : nodeType;
            TypeMirror unboxedDestType =
                TypesUtils.isBoxedPrimitive(destType) ?
                types.unboxedType(destType) : destType;
            if (TypesUtils.isNumeric(unboxedNodeType) &&
                TypesUtils.isNumeric(unboxedDestType)) {
                if (unboxedNodeType.getKind() == TypeKind.BYTE &&
                    destType.getKind() == TypeKind.SHORT) {
                    if (isBoxedPrimitive) {
                        node = unbox(node);
                    }
                    return widen(node, destType);
                }

                // If the operand is Byte, Short or Character and the whole expression
                // is the unboxed version of it, then apply unboxing.
                TypeKind destKind = destType.getKind();
                if (destKind == TypeKind.BYTE || destKind == TypeKind.CHAR ||
                    destKind == TypeKind.SHORT) {
                    if (isBoxedPrimitive) {
                        return unbox(node);
                    } else if (nodeType.getKind() == TypeKind.INT) {
                        return narrow(node, destType);
                    }
                }

                return binaryNumericPromotion(node, destType);
            }

            // For the final case in JLS 15.25, apply boxing but not lub.
            if (TypesUtils.isPrimitive(nodeType) &&
                (destType.getKind() == TypeKind.DECLARED ||
                 destType.getKind() == TypeKind.UNION ||
                 destType.getKind() == TypeKind.INTERSECTION)) {
                return box(node);
            }

            return node;
        }

        /**
         * Returns the label {@link Name} of the leaf in the argument path, or
         * null if the leaf is not a labeled statement.
         */
        protected /*@Nullable*/ Name getLabel(TreePath path) {
            if (path.getParentPath() != null) {
                Tree parent = path.getParentPath().getLeaf();
                if (parent.getKind() == Tree.Kind.LABELED_STATEMENT) {
                    return ((LabeledStatementTree) parent).getLabel();
                }
            }
            return null;
        }

        /* --------------------------------------------------------- */
        /* Visitor Methods */
        /* --------------------------------------------------------- */

        @Override
        public Node visitAnnotatedType(AnnotatedTypeTree tree, Void p) {
            return scan(tree.getUnderlyingType(), p);
        }

        @Override
        public Node visitAnnotation(AnnotationTree tree, Void p) {
            assert false : "AnnotationTree is unexpected in AST to CFG translation";
            return null;
        }

        @Override
        public MethodInvocationNode visitMethodInvocation(MethodInvocationTree tree, Void p) {

            // see JLS 15.12.4

            // First, compute the receiver, if any (15.12.4.1)
            // Second, evaluate the actual arguments, left to right and
            // possibly some arguments are stored into an array for variable
            // arguments calls (15.12.4.2)
            // Third, test the receiver, if any, for nullness (15.12.4.4)
            // Fourth, convert the arguments to the type of the formal
            // parameters (15.12.4.5)
            // Fifth, if the method is synchronized, lock the receiving
            // object or class (15.12.4.5)
            ExecutableElement method = TreeUtils.elementFromUse(tree);
            // TODO? Variable wasn't used.
            // boolean isBooleanMethod = TypesUtils.isBooleanType(method.getReturnType());

            ExpressionTree methodSelect = tree.getMethodSelect();
            assert TreeUtils.isMethodAccess(methodSelect) : "Expected a method access, but got: " + methodSelect;

            List<? extends ExpressionTree> actualExprs = tree.getArguments();

            // Look up method to invoke and possibly throw NullPointerException
            Node receiver = getReceiver(methodSelect,
                    TreeUtils.enclosingClass(getCurrentPath()));

            MethodAccessNode target = new MethodAccessNode(methodSelect,
                    receiver);

            ExecutableElement element = TreeUtils.elementFromUse(tree);
            if (ElementUtils.isStatic(element) ||
                receiver instanceof ThisLiteralNode) {
                // No NullPointerException can be thrown, use normal node
                extendWithNode(target);
            } else {
                TypeElement npeElement = elements
                    .getTypeElement("java.lang.NullPointerException");
                extendWithNodeWithException(target, npeElement.asType());
            }

            List<Node> arguments = new ArrayList<>();

            // Don't convert arguments for enum super calls.  The AST contains
            // no actual arguments, while the method element expects two arguments,
            // leading to an exception in convertCallArguments.  Since no actual
            // arguments are present in the AST that is being checked, it shouldn't
            // cause any harm to omit the conversions.
            // See also BaseTypeVisitor.visitMethodInvocation and
            // QualifierPolymorphism.annotate
            if (!TreeUtils.isEnumSuper(tree)) {
                arguments = convertCallArguments(method, actualExprs);
            }

            // TODO: lock the receiver for synchronized methods

            MethodInvocationNode node = new MethodInvocationNode(tree, target, arguments, getCurrentPath());

            Set<TypeMirror> thrownSet = new HashSet<>();
            // Add exceptions explicitly mentioned in the throws clause.
            List<? extends TypeMirror> thrownTypes = element.getThrownTypes();
            thrownSet.addAll(thrownTypes);
            // Add Throwable to account for unchecked exceptions
            TypeElement throwableElement = elements
                    .getTypeElement("java.lang.Throwable");
            thrownSet.add(throwableElement.asType());

            ExtendedNode extendedNode = extendWithNodeWithExceptions(node, thrownSet);

            /* Check for the TerminatesExecution annotation. */
            Element methodElement = InternalUtils.symbol(tree);
            boolean terminatesExecution = annotationProvider.getDeclAnnotation(
                    methodElement, TerminatesExecution.class) != null;
            if (terminatesExecution) {
                extendedNode.setTerminatesExecution(true);
            }

            return node;
        }

        @Override
        public Node visitAssert(AssertTree tree, Void p) {

            // see JLS 14.10

            // If assertions are enabled, then we can just translate the
            // assertion.
            if (assumeAssertionsEnabled || assumeAssertionsEnabledFor(tree)) {
                translateAssertWithAssertionsEnabled(tree);
                return null;
            }

            // If assertions are disabled, then nothing is executed.
            if (assumeAssertionsDisabled) {
                return null;
            }


            // Otherwise, we don't know if assertions are enabled, so we use a
            // variable "ea" and case-split on it. One branch does execute the
            // assertion, while the other assumes assertions are disabled.
            VariableTree ea = getAssertionsEnabledVariable();

            // all necessary labels
            Label assertionEnabled = new Label();
            Label assertionDisabled = new Label();

            extendWithNode(new LocalVariableNode(ea));
            extendWithExtendedNode(new ConditionalJump(assertionEnabled,
                    assertionDisabled));

            // 'then' branch (i.e. check the assertion)
            addLabelForNextNode(assertionEnabled);

            translateAssertWithAssertionsEnabled(tree);

            // 'else' branch
            addLabelForNextNode(assertionDisabled);

            return null;
        }

        /**
         * Should assertions be assumed to be executed for a given
         * {@link AssertTree}? False by default.
         */
        protected boolean assumeAssertionsEnabledFor(AssertTree tree) {
            return false;
        }

        /**
         * The {@link VariableTree} that indicates whether assertions are
         * enabled or not.
         */
        protected VariableTree ea = null;

        /**
         * Get a synthetic {@link VariableTree} that indicates whether assertions are
         * enabled or not.
         */
        protected VariableTree getAssertionsEnabledVariable() {
            if (ea == null) {
                String name = uniqueName("assertionsEnabled");
                MethodTree enclosingMethod = TreeUtils
                        .enclosingMethod(getCurrentPath());
                Element owner;
                if (enclosingMethod != null) {
                    owner = TreeUtils.elementFromDeclaration(enclosingMethod);
                } else {
                    ClassTree enclosingClass = TreeUtils.enclosingClass(getCurrentPath());
                    owner = TreeUtils.elementFromDeclaration(enclosingClass);
                }
                ExpressionTree initializer = null;
                ea = treeBuilder.buildVariableDecl(
                        types.getPrimitiveType(TypeKind.BOOLEAN), name, owner,
                        initializer);
            }
            return ea;
        }

        /**
         * Translates an assertion statement to the correct CFG nodes. The
         * translation assumes that assertions are enabled.
         */
        protected void translateAssertWithAssertionsEnabled(AssertTree tree) {

            // all necessary labels
            Label assertEnd = new Label();
            Label elseEntry = new Label();

            // basic block for the condition
            Node condition = unbox(scan(tree.getCondition(), null));
            ConditionalJump cjump = new ConditionalJump(assertEnd, elseEntry);
            extendWithExtendedNode(cjump);

            // else branch
            Node detail = null;
            addLabelForNextNode(elseEntry);
            if (tree.getDetail() != null) {
                detail = scan(tree.getDetail(), null);
            }
            TypeElement assertException = elements
                    .getTypeElement("java.lang.AssertionError");
            AssertionErrorNode assertNode = new AssertionErrorNode(tree,
                    condition, detail, assertException.asType());
            extendWithNode(assertNode);
            NodeWithExceptionsHolder exNode = extendWithNodeWithException(
                    new ThrowNode(null, assertNode, env.getTypeUtils()), assertException.asType());
            exNode.setTerminatesExecution(true);

            // then branch (nothing happens)
            addLabelForNextNode(assertEnd);
        }

        @Override
        public Node visitAssignment(AssignmentTree tree, Void p) {

            // see JLS 15.26.1

            AssignmentNode assignmentNode;
            ExpressionTree variable = tree.getVariable();
            TypeMirror varType = InternalUtils.typeOf(variable);

            // case 1: field access
            if (TreeUtils.isFieldAccess(variable)) {
                // visit receiver
                Node receiver = getReceiver(variable,
                        TreeUtils.enclosingClass(getCurrentPath()));

                // visit expression
                Node expression = scan(tree.getExpression(), p);
                expression = assignConvert(expression, varType);

                // visit field access (throws null-pointer exception)
                FieldAccessNode target = new FieldAccessNode(variable, receiver);
                target.setLValue();

                Element element = TreeUtils.elementFromUse(variable);
                if (ElementUtils.isStatic(element) ||
                    receiver instanceof ThisLiteralNode) {
                    // No NullPointerException can be thrown, use normal node
                    extendWithNode(target);
                } else {
                    TypeElement npeElement = elements
                            .getTypeElement("java.lang.NullPointerException");
                    extendWithNodeWithException(target, npeElement.asType());
                }

                // add assignment node
                assignmentNode = new AssignmentNode(tree,
                        target, expression);
                extendWithNode(assignmentNode);
            }

            // case 2: other cases
            else {
                Node target = scan(variable, p);
                target.setLValue();

                assignmentNode = translateAssignment(tree, target,
                        tree.getExpression());
            }

            return assignmentNode;
        }

        /**
         * Translate an assignment.
         */
        protected AssignmentNode translateAssignment(Tree tree, Node target,
                ExpressionTree rhs) {
            Node expression = scan(rhs, null);
            return translateAssignment(tree, target, expression);
        }

        /**
         * Translate an assignment where the RHS has already been scanned.
         */
        protected AssignmentNode translateAssignment(Tree tree, Node target,
                Node expression) {
            assert tree instanceof AssignmentTree
                    || tree instanceof VariableTree;
            target.setLValue();
            expression = assignConvert(expression, target.getType());
            AssignmentNode assignmentNode = new AssignmentNode(tree, target,
                    expression);
            extendWithNode(assignmentNode);
            return assignmentNode;
        }

        /**
         * Note 1: Requires {@code tree} to be a field or method access
         * tree.
         * <p>
         * Note 2: Visits the receiver and adds all necessary blocks to the CFG.
         *
         * @param tree
         *            the field access tree containing the receiver
         * @param classTree
         *            the ClassTree enclosing the field access
         * @return the receiver of the field access
         */
        private Node getReceiver(Tree tree, ClassTree classTree) {
            assert TreeUtils.isFieldAccess(tree)
                    || TreeUtils.isMethodAccess(tree);
            if (tree.getKind().equals(Tree.Kind.MEMBER_SELECT)) {
                MemberSelectTree mtree = (MemberSelectTree) tree;
                return scan(mtree.getExpression(), null);
            } else {
                TypeMirror classType = InternalUtils.typeOf(classTree);
                Node node = new ImplicitThisLiteralNode(classType);
                extendWithNode(node);
                return node;
            }
        }

        /**
         * Map an operation with assignment to the corresponding operation
         * without assignment.
         *
         * @param kind  a Tree.Kind representing an operation with assignment
         * @return the Tree.Kind for the same operation without assignment
         */
        protected Tree.Kind withoutAssignment(Tree.Kind kind) {
            switch (kind) {
            case DIVIDE_ASSIGNMENT:
                return Tree.Kind.DIVIDE;
            case MULTIPLY_ASSIGNMENT:
                return Tree.Kind.MULTIPLY;
            case REMAINDER_ASSIGNMENT:
                return Tree.Kind.REMAINDER;
            case MINUS_ASSIGNMENT:
                return Tree.Kind.MINUS;
            case PLUS_ASSIGNMENT:
                return Tree.Kind.PLUS;
            case LEFT_SHIFT_ASSIGNMENT:
                return Tree.Kind.LEFT_SHIFT;
            case RIGHT_SHIFT_ASSIGNMENT:
                return Tree.Kind.RIGHT_SHIFT;
            case UNSIGNED_RIGHT_SHIFT_ASSIGNMENT:
                return Tree.Kind.UNSIGNED_RIGHT_SHIFT;
            case AND_ASSIGNMENT:
                return Tree.Kind.AND;
            case OR_ASSIGNMENT:
                return Tree.Kind.OR;
            case XOR_ASSIGNMENT:
                return Tree.Kind.XOR;
            default:
                return Tree.Kind.ERRONEOUS;
            }
        }


        @Override
        public Node visitCompoundAssignment(CompoundAssignmentTree tree, Void p) {
            // According the JLS 15.26.2, E1 op= E2 is equivalent to
            // E1 = (T) ((E1) op (E2)), where T is the type of E1,
            // except that E1 is evaluated only once.
            //

            Tree.Kind kind = tree.getKind();
            switch (kind) {
            case DIVIDE_ASSIGNMENT:
            case MULTIPLY_ASSIGNMENT:
            case REMAINDER_ASSIGNMENT: {
                // see JLS 15.17 and 15.26.2
                Node targetLHS = scan(tree.getVariable(), p);
                Node value = scan(tree.getExpression(), p);

                TypeMirror exprType = InternalUtils.typeOf(tree);
                TypeMirror leftType = InternalUtils.typeOf(tree.getVariable());
                TypeMirror rightType = InternalUtils.typeOf(tree.getExpression());
                TypeMirror promotedType = binaryPromotedType(leftType, rightType);
                Node targetRHS = binaryNumericPromotion(targetLHS, promotedType);
                value = binaryNumericPromotion(value, promotedType);

                BinaryTree operTree = treeBuilder.buildBinary(promotedType, withoutAssignment(kind),
                        tree.getVariable(), tree.getExpression());
                handleArtificialTree(operTree);
                Node operNode;
                if (kind == Tree.Kind.MULTIPLY_ASSIGNMENT) {
                    operNode = new NumericalMultiplicationNode(operTree, targetRHS, value);
                } else if (kind == Tree.Kind.DIVIDE_ASSIGNMENT) {
                    if (TypesUtils.isIntegral(exprType)) {
                        operNode = new IntegerDivisionNode(operTree, targetRHS, value);
                    } else {
                        operNode = new FloatingDivisionNode(operTree, targetRHS, value);
                    }
                } else {
                    assert kind == Kind.REMAINDER_ASSIGNMENT;
                    if (TypesUtils.isIntegral(exprType)) {
                        operNode = new IntegerRemainderNode(operTree, targetRHS, value);
                    } else {
                        operNode = new FloatingRemainderNode(operTree, targetRHS, value);
                    }
                }
                extendWithNode(operNode);

                TypeCastTree castTree = treeBuilder.buildTypeCast(leftType, operTree);
                handleArtificialTree(castTree);
                TypeCastNode castNode = new TypeCastNode(castTree, operNode, leftType);
                castNode.setInSource(false);
                extendWithNode(castNode);

                AssignmentNode assignNode = new AssignmentNode(tree, targetLHS, castNode);
                extendWithNode(assignNode);
                return assignNode;
            }

            case MINUS_ASSIGNMENT:
            case PLUS_ASSIGNMENT: {
                // see JLS 15.18 and 15.26.2

                Node targetLHS = scan(tree.getVariable(), p);
                Node value = scan(tree.getExpression(), p);

                TypeMirror leftType = InternalUtils.typeOf(tree.getVariable());
                TypeMirror rightType = InternalUtils.typeOf(tree.getExpression());

                if (TypesUtils.isString(leftType) || TypesUtils.isString(rightType)) {
                    assert (kind == Tree.Kind.PLUS_ASSIGNMENT);
                    Node targetRHS = stringConversion(targetLHS);
                    value = stringConversion(value);
                    Node r = new StringConcatenateAssignmentNode(tree, targetRHS, value);
                    extendWithNode(r);
                    return r;
                } else {
                    TypeMirror promotedType = binaryPromotedType(leftType, rightType);
                    Node targetRHS = binaryNumericPromotion(targetLHS, promotedType);
                    value = binaryNumericPromotion(value, promotedType);

                    BinaryTree operTree = treeBuilder.buildBinary(promotedType, withoutAssignment(kind),
                            tree.getVariable(), tree.getExpression());
                    handleArtificialTree(operTree);
                    Node operNode;
                    if (kind == Tree.Kind.PLUS_ASSIGNMENT) {
                        operNode = new NumericalAdditionNode(operTree, targetRHS, value);
                    } else {
                        assert kind == Kind.MINUS_ASSIGNMENT;
                        operNode = new NumericalSubtractionNode(operTree, targetRHS, value);
                    }
                    extendWithNode(operNode);

                    TypeCastTree castTree = treeBuilder.buildTypeCast(leftType, operTree);
                    handleArtificialTree(castTree);
                    TypeCastNode castNode = new TypeCastNode(castTree, operNode, leftType);
                    castNode.setInSource(false);
                    extendWithNode(castNode);

                    // Map the compound assignment tree to an assignment node, which
                    // will have the correct type.
                    AssignmentNode assignNode = new AssignmentNode(tree, targetLHS, castNode);
                    extendWithNode(assignNode);
                    return assignNode;
                }
            }

            case LEFT_SHIFT_ASSIGNMENT:
            case RIGHT_SHIFT_ASSIGNMENT:
            case UNSIGNED_RIGHT_SHIFT_ASSIGNMENT: {
                // see JLS 15.19 and 15.26.2
                Node targetLHS = scan(tree.getVariable(), p);
                Node value = scan(tree.getExpression(), p);

                TypeMirror leftType = InternalUtils.typeOf(tree.getVariable());

                Node targetRHS = unaryNumericPromotion(targetLHS);
                value = unaryNumericPromotion(value);

                BinaryTree operTree = treeBuilder.buildBinary(leftType, withoutAssignment(kind),
                        tree.getVariable(), tree.getExpression());
                handleArtificialTree(operTree);
                Node operNode;
                if (kind == Tree.Kind.LEFT_SHIFT_ASSIGNMENT) {
                    operNode = new LeftShiftNode(operTree, targetRHS, value);
                } else if (kind == Tree.Kind.RIGHT_SHIFT_ASSIGNMENT) {
                    operNode = new SignedRightShiftNode(operTree, targetRHS, value);
                } else {
                    assert kind == Kind.UNSIGNED_RIGHT_SHIFT_ASSIGNMENT;
                    operNode = new UnsignedRightShiftNode(operTree, targetRHS, value);
                }
                extendWithNode(operNode);

                TypeCastTree castTree = treeBuilder.buildTypeCast(leftType, operTree);
                handleArtificialTree(castTree);
                TypeCastNode castNode = new TypeCastNode(castTree, operNode, leftType);
                castNode.setInSource(false);
                extendWithNode(castNode);

                AssignmentNode assignNode = new AssignmentNode(tree, targetLHS, castNode);
                extendWithNode(assignNode);
                return assignNode;
            }

            case AND_ASSIGNMENT:
            case OR_ASSIGNMENT:
            case XOR_ASSIGNMENT:
                // see JLS 15.22
                Node targetLHS = scan(tree.getVariable(), p);
                Node value = scan(tree.getExpression(), p);

                TypeMirror leftType = InternalUtils.typeOf(tree.getVariable());
                TypeMirror rightType = InternalUtils.typeOf(tree.getExpression());

                Node targetRHS = null;
                if (isNumericOrBoxed(leftType) && isNumericOrBoxed(rightType)) {
                    TypeMirror promotedType = binaryPromotedType(leftType, rightType);
                    targetRHS = binaryNumericPromotion(targetLHS, promotedType);
                    value = binaryNumericPromotion(value, promotedType);
                } else if (TypesUtils.isBooleanType(leftType) &&
                           TypesUtils.isBooleanType(rightType)) {
                    targetRHS = unbox(targetLHS);
                    value = unbox(value);
                } else {
                    assert false :
                        "Both argument to logical operation must be numeric or boolean";
                }

                BinaryTree operTree = treeBuilder.buildBinary(leftType, withoutAssignment(kind),
                        tree.getVariable(), tree.getExpression());
                handleArtificialTree(operTree);
                Node operNode;
                if (kind == Tree.Kind.AND_ASSIGNMENT) {
                    operNode = new BitwiseAndNode(operTree, targetRHS, value);
                } else if (kind == Tree.Kind.OR_ASSIGNMENT) {
                    operNode = new BitwiseOrNode(operTree, targetRHS, value);
                } else {
                    assert kind == Kind.XOR_ASSIGNMENT;
                    operNode = new BitwiseXorNode(operTree, targetRHS, value);
                }
                extendWithNode(operNode);

                TypeCastTree castTree = treeBuilder.buildTypeCast(leftType, operTree);
                handleArtificialTree(castTree);
                TypeCastNode castNode = new TypeCastNode(castTree, operNode, leftType);
                castNode.setInSource(false);
                extendWithNode(castNode);

                AssignmentNode assignNode = new AssignmentNode(tree, targetLHS, castNode);
                extendWithNode(assignNode);
                return assignNode;
            default:
                assert false : "unexpected compound assignment type";
                break;
            }
            assert false : "unexpected compound assignment type";
            return null;
        }

        @Override
        public Node visitBinary(BinaryTree tree, Void p) {
            // Note that for binary operations it is important to perform any required
            // promotion on the left operand before generating any Nodes for the right
            // operand, because labels must be inserted AFTER ALL preceding Nodes and
            // BEFORE ALL following Nodes.
            Node r = null;
            Tree leftTree = tree.getLeftOperand();
            Tree rightTree = tree.getRightOperand();

            Tree.Kind kind = tree.getKind();
            switch (kind) {
            case DIVIDE:
            case MULTIPLY:
            case REMAINDER: {
                // see JLS 15.17

                TypeMirror exprType = InternalUtils.typeOf(tree);
                TypeMirror leftType = InternalUtils.typeOf(leftTree);
                TypeMirror rightType = InternalUtils.typeOf(rightTree);
                TypeMirror promotedType = binaryPromotedType(leftType, rightType);

                Node left = binaryNumericPromotion(scan(leftTree, p), promotedType);
                Node right = binaryNumericPromotion(scan(rightTree, p), promotedType);

                if (kind == Tree.Kind.MULTIPLY) {
                    r = new NumericalMultiplicationNode(tree, left, right);
                } else if (kind == Tree.Kind.DIVIDE) {
                    if (TypesUtils.isIntegral(exprType)) {
                        r = new IntegerDivisionNode(tree, left, right);
                    } else {
                        r = new FloatingDivisionNode(tree, left, right);
                    }
                } else {
                    assert kind == Kind.REMAINDER;
                    if (TypesUtils.isIntegral(exprType)) {
                        r = new IntegerRemainderNode(tree, left, right);
                    } else {
                        r = new FloatingRemainderNode(tree, left, right);
                    }
                }
                break;
            }

            case MINUS:
            case PLUS: {
                // see JLS 15.18

                // TypeMirror exprType = InternalUtils.typeOf(tree);
                TypeMirror leftType = InternalUtils.typeOf(leftTree);
                TypeMirror rightType = InternalUtils.typeOf(rightTree);

                if (TypesUtils.isString(leftType) || TypesUtils.isString(rightType)) {
                    assert (kind == Tree.Kind.PLUS);
                    Node left = stringConversion(scan(leftTree, p));
                    Node right = stringConversion(scan(rightTree, p));
                    r = new StringConcatenateNode(tree, left, right);
                } else {
                    TypeMirror promotedType = binaryPromotedType(leftType, rightType);
                    Node left = binaryNumericPromotion(scan(leftTree, p), promotedType);
                    Node right = binaryNumericPromotion(scan(rightTree, p), promotedType);

                    // TODO: Decide whether to deal with floating-point value
                    // set conversion.
                    if (kind == Tree.Kind.PLUS) {
                        r = new NumericalAdditionNode(tree, left, right);
                    } else {
                        assert kind == Kind.MINUS;
                        r = new NumericalSubtractionNode(tree, left, right);
                    }
                }
                break;
            }

            case LEFT_SHIFT:
            case RIGHT_SHIFT:
            case UNSIGNED_RIGHT_SHIFT: {
                // see JLS 15.19

                Node left = unaryNumericPromotion(scan(leftTree, p));
                Node right = unaryNumericPromotion(scan(rightTree, p));

                if (kind == Tree.Kind.LEFT_SHIFT) {
                    r = new LeftShiftNode(tree, left, right);
                } else if (kind == Tree.Kind.RIGHT_SHIFT) {
                    r = new SignedRightShiftNode(tree, left, right);
                } else {
                    assert kind == Kind.UNSIGNED_RIGHT_SHIFT;
                    r = new UnsignedRightShiftNode(tree, left, right);
                }
                break;
            }

            case GREATER_THAN:
            case GREATER_THAN_EQUAL:
            case LESS_THAN:
            case LESS_THAN_EQUAL: {
                // see JLS 15.20.1
                TypeMirror leftType = InternalUtils.typeOf(leftTree);
                if (TypesUtils.isBoxedPrimitive(leftType)) {
                    leftType = types.unboxedType(leftType);
                }

                TypeMirror rightType = InternalUtils.typeOf(rightTree);
                if (TypesUtils.isBoxedPrimitive(rightType)) {
                    rightType = types.unboxedType(rightType);
                }

                TypeMirror promotedType = binaryPromotedType(leftType, rightType);
                Node left = binaryNumericPromotion(scan(leftTree, p), promotedType);
                Node right = binaryNumericPromotion(scan(rightTree, p), promotedType);

                Node node;
                if (kind == Tree.Kind.GREATER_THAN) {
                    node = new GreaterThanNode(tree, left, right);
                } else if (kind == Tree.Kind.GREATER_THAN_EQUAL) {
                    node = new GreaterThanOrEqualNode(tree, left, right);
                } else if (kind == Tree.Kind.LESS_THAN) {
                    node = new LessThanNode(tree, left, right);
                } else {
                    assert kind == Tree.Kind.LESS_THAN_EQUAL;
                    node = new LessThanOrEqualNode(tree, left, right);
                }

                extendWithNode(node);

                return node;
            }

            case EQUAL_TO:
            case NOT_EQUAL_TO: {
                // see JLS 15.21
                TreeInfo leftInfo = getTreeInfo(leftTree);
                TreeInfo rightInfo = getTreeInfo(rightTree);
                Node left = scan(leftTree, p);
                Node right = scan(rightTree, p);

                if (leftInfo.isNumeric() && rightInfo.isNumeric() &&
                   !(leftInfo.isBoxed() && rightInfo.isBoxed())) {
                    // JLS 15.21.1 numerical equality
                    TypeMirror promotedType = binaryPromotedType(leftInfo.unboxedType(),
                                                                 rightInfo.unboxedType());
                    left  = binaryNumericPromotion(left,  promotedType);
                    right = binaryNumericPromotion(right, promotedType);
                } else if (leftInfo.isBoolean() && rightInfo.isBoolean() &&
                          !(leftInfo.isBoxed() && rightInfo.isBoxed())) {
                    // JSL 15.21.2 boolean equality
                    left  = unboxAsNeeded(left,  leftInfo.isBoxed());
                    right = unboxAsNeeded(right, rightInfo.isBoxed());
                }

                Node node;
                if (kind == Tree.Kind.EQUAL_TO) {
                    node = new EqualToNode(tree, left, right);
                } else {
                    assert kind == Kind.NOT_EQUAL_TO;
                    node = new NotEqualNode(tree, left, right);
                }
                extendWithNode(node);

                return node;
            }

            case AND:
            case OR:
            case XOR: {
                // see JLS 15.22
                TypeMirror leftType = InternalUtils.typeOf(leftTree);
                TypeMirror rightType = InternalUtils.typeOf(rightTree);
                boolean isBooleanOp = TypesUtils.isBooleanType(leftType) &&
                    TypesUtils.isBooleanType(rightType);

                Node left;
                Node right;

                if (isBooleanOp) {
                    left = unbox(scan(leftTree, p));
                    right = unbox(scan(rightTree, p));
                } else if (isNumericOrBoxed(leftType) && isNumericOrBoxed(rightType)) {
                    TypeMirror promotedType = binaryPromotedType(leftType, rightType);
                    left = binaryNumericPromotion(scan(leftTree, p), promotedType);
                    right = binaryNumericPromotion(scan(rightTree, p), promotedType);
                } else {
                    left = unbox(scan(leftTree, p));
                    right = unbox(scan(rightTree, p));
                }

                Node node;
                if (kind == Tree.Kind.AND) {
                    node = new BitwiseAndNode(tree, left, right);
                } else if (kind == Tree.Kind.OR) {
                    node = new BitwiseOrNode(tree, left, right);
                } else {
                    assert kind == Kind.XOR;
                    node = new BitwiseXorNode(tree, left, right);
                }

                extendWithNode(node);

                return node;
            }

            case CONDITIONAL_AND:
            case CONDITIONAL_OR: {
                // see JLS 15.23 and 15.24

                // all necessary labels
                Label rightStartL = new Label();
                Label shortCircuitL = new Label();

                // left-hand side
                Node left = scan(leftTree, p);

                ConditionalJump cjump;
                if (kind == Tree.Kind.CONDITIONAL_AND) {
                    cjump = new ConditionalJump(rightStartL, shortCircuitL);
                    cjump.setFalseFlowRule(Store.FlowRule.ELSE_TO_ELSE);
                } else {
                    cjump = new ConditionalJump(shortCircuitL, rightStartL);
                    cjump.setTrueFlowRule(Store.FlowRule.THEN_TO_THEN);
                }
                extendWithExtendedNode(cjump);

                // right-hand side
                addLabelForNextNode(rightStartL);
                Node right = scan(rightTree, p);

                // conditional expression itself
                addLabelForNextNode(shortCircuitL);
                Node node;
                if (kind == Tree.Kind.CONDITIONAL_AND) {
                    node = new ConditionalAndNode(tree, left, right);
                } else {
                    node = new ConditionalOrNode(tree, left, right);
                }
                extendWithNode(node);
                return node;
            }
            default:
                assert false : "unexpected binary tree: " + kind;
                break;
            }
            assert r != null : "unexpected binary tree";
            return extendWithNode(r);
        }

        @Override
        public Node visitBlock(BlockTree tree, Void p) {
            for (StatementTree n : tree.getStatements()) {
                scan(n, null);
            }
            return null;
        }

        @Override
        public Node visitBreak(BreakTree tree, Void p) {
            Name label = tree.getLabel();
            if (label == null) {
                assert breakTargetL != null : "no target for break statement";

                extendWithExtendedNode(new UnconditionalJump(breakTargetL));
            } else {
                assert breakLabels.containsKey(label);

                extendWithExtendedNode(new UnconditionalJump(
                        breakLabels.get(label)));
            }

            return null;
        }

        @Override
        public Node visitSwitch(SwitchTree tree, Void p) {
            SwitchBuilder builder = new SwitchBuilder(tree, p);
            builder.build();
            return null;
        }

        private class SwitchBuilder {
            final private SwitchTree switchTree;
            final private Label[] caseBodyLabels;
            final private Void p;
            private Node switchExpr;

            private SwitchBuilder(SwitchTree tree, Void p) {
                this.switchTree = tree;
                this.caseBodyLabels = new Label[switchTree.getCases().size() + 1];
                this.p = p;
            }

            public void build() {
                Label oldBreakTargetL = breakTargetL;
                breakTargetL = new Label();
                int cases = caseBodyLabels.length-1;
                for (int i = 0; i < cases; ++i) {
                    caseBodyLabels[i] = new Label();
                }
                caseBodyLabels[cases] = breakTargetL;

                switchExpr = unbox(scan(switchTree.getExpression(), p));
                extendWithNode(new MarkerNode(switchTree, "start of switch statement", env.getTypeUtils()));

                Integer defaultIndex = null;
                for (int i = 0; i < cases; ++i) {
                    CaseTree caseTree = switchTree.getCases().get(i);
                    if (caseTree.getExpression() == null) {
                        defaultIndex = i;
                    } else {
                        buildCase(caseTree, i);
                    }
                }
                if (defaultIndex != null) {
                    // the checks of all cases must happen before the default case,
                    // therefore we build the default case last.
                    // fallthrough is still handled correctly with the caseBodyLabels.
                    buildCase(switchTree.getCases().get(defaultIndex), defaultIndex);
                }

                addLabelForNextNode(breakTargetL);
                breakTargetL = oldBreakTargetL;
            }

            private void buildCase(CaseTree tree, int index) {
                final Label thisBodyL = caseBodyLabels[index];
                final Label nextBodyL = caseBodyLabels[index+1];
                final Label nextCaseL = new Label();

                ExpressionTree exprTree = tree.getExpression();
                if (exprTree != null) {
                    Node expr = scan(exprTree, p);
                    CaseNode test = new CaseNode(tree, switchExpr, expr, env.getTypeUtils());
                    extendWithNode(test);
                    extendWithExtendedNode(new ConditionalJump(thisBodyL, nextCaseL));
                }
                addLabelForNextNode(thisBodyL);
                for (StatementTree stmt : tree.getStatements()) {
                    scan(stmt, p);
                }
                extendWithExtendedNode(new UnconditionalJump(nextBodyL));
                addLabelForNextNode(nextCaseL);
            }
        }

        @Override
        public Node visitCase(CaseTree tree, Void p) {
            throw new AssertionError("case visitor is implemented in SwitchBuilder");
        }

        @Override
        public Node visitCatch(CatchTree tree, Void p) {
            scan(tree.getParameter(), p);
            scan(tree.getBlock(), p);
            return null;
        }

        @Override
        public Node visitClass(ClassTree tree, Void p) {
            declaredClasses.add(tree);
            return null;
        }

        @Override
        public Node visitConditionalExpression(ConditionalExpressionTree tree,
                Void p) {
            // see JLS 15.25
            TypeMirror exprType = InternalUtils.typeOf(tree);

            Label trueStart = new Label();
            Label falseStart = new Label();
            Label merge = new Label();

            Node condition = unbox(scan(tree.getCondition(), p));
            ConditionalJump cjump = new ConditionalJump(trueStart, falseStart);
            extendWithExtendedNode(cjump);

            addLabelForNextNode(trueStart);
            Node trueExpr = scan(tree.getTrueExpression(), p);
            trueExpr = conditionalExprPromotion(trueExpr, exprType);
            extendWithExtendedNode(new UnconditionalJump(merge));

            addLabelForNextNode(falseStart);
            Node falseExpr = scan(tree.getFalseExpression(), p);
            falseExpr = conditionalExprPromotion(falseExpr, exprType);

            addLabelForNextNode(merge);
            Node node = new TernaryExpressionNode(tree, condition, trueExpr, falseExpr);
            extendWithNode(node);

            return node;
        }

        @Override
        public Node visitContinue(ContinueTree tree, Void p) {
            Name label = tree.getLabel();
            if (label == null) {
                assert continueTargetL != null : "no target for continue statement";

                extendWithExtendedNode(new UnconditionalJump(continueTargetL));
            } else {
                assert continueLabels.containsKey(label);

                extendWithExtendedNode(new UnconditionalJump(
                        continueLabels.get(label)));
            }

            return null;
        }

        @Override
        public Node visitDoWhileLoop(DoWhileLoopTree tree, Void p) {
            Name parentLabel = getLabel(getCurrentPath());

            Label loopEntry = new Label();
            Label loopExit = new Label();

            // If the loop is a labeled statement, then its continue
            // target is identical for continues with no label and
            // continues with the loop's label.
            Label conditionStart;
            if (parentLabel != null) {
                conditionStart = continueLabels.get(parentLabel);
            } else {
                conditionStart = new Label();
            }

            Label oldBreakTargetL = breakTargetL;
            breakTargetL = loopExit;

            Label oldContinueTargetL = continueTargetL;
            continueTargetL = conditionStart;

            // Loop body
            addLabelForNextNode(loopEntry);
            if (tree.getStatement() != null) {
                scan(tree.getStatement(), p);
            }

            // Condition
            addLabelForNextNode(conditionStart);
            if (tree.getCondition() != null) {
                unbox(scan(tree.getCondition(), p));
                ConditionalJump cjump = new ConditionalJump(loopEntry, loopExit);
                extendWithExtendedNode(cjump);
            }

            // Loop exit
            addLabelForNextNode(loopExit);

            breakTargetL = oldBreakTargetL;
            continueTargetL = oldContinueTargetL;

            return null;
        }

        @Override
        public Node visitErroneous(ErroneousTree tree, Void p) {
            assert false : "ErroneousTree is unexpected in AST to CFG translation";
            return null;
        }

        @Override
        public Node visitExpressionStatement(ExpressionStatementTree tree,
                Void p) {
            return scan(tree.getExpression(), p);
        }

        @Override
        public Node visitEnhancedForLoop(EnhancedForLoopTree tree, Void p) {
          // see JLS 14.14.2
          Name parentLabel = getLabel(getCurrentPath());

          Label conditionStart = new Label();
          Label loopEntry = new Label();
          Label loopExit = new Label();

          // If the loop is a labeled statement, then its continue
          // target is identical for continues with no label and
          // continues with the loop's label.
          Label updateStart;
          if (parentLabel != null) {
              updateStart = continueLabels.get(parentLabel);
          } else {
              updateStart = new Label();
          }

          Label oldBreakTargetL = breakTargetL;
          breakTargetL = loopExit;

          Label oldContinueTargetL = continueTargetL;
          continueTargetL = updateStart;

          // Distinguish loops over Iterables from loops over arrays.

          TypeElement iterableElement = elements.getTypeElement("java.lang.Iterable");
          TypeMirror iterableType = types.erasure(iterableElement.asType());

          VariableTree variable = tree.getVariable();
          VariableElement variableElement =
              TreeUtils.elementFromDeclaration(variable);
          ExpressionTree expression = tree.getExpression();
          StatementTree statement = tree.getStatement();

          TypeMirror exprType = InternalUtils.typeOf(expression);

          if (types.isSubtype(exprType, iterableType)) {
              // Take the upper bound of a type variable or wildcard
              exprType = TypesUtils.upperBound(exprType);

              assert (exprType instanceof DeclaredType) : "an Iterable must be a DeclaredType";
              DeclaredType declaredExprType = (DeclaredType) exprType;
              declaredExprType.getTypeArguments();

              MemberSelectTree iteratorSelect =
                  treeBuilder.buildIteratorMethodAccess(expression);
              handleArtificialTree(iteratorSelect);

              MethodInvocationTree iteratorCall =
                  treeBuilder.buildMethodInvocation(iteratorSelect);
              handleArtificialTree(iteratorCall);

              VariableTree iteratorVariable = createEnhancedForLoopIteratorVariable(iteratorCall, variableElement);
              handleArtificialTree(iteratorVariable);

              VariableDeclarationNode iteratorVariableDecl =
                  new VariableDeclarationNode(iteratorVariable);
              iteratorVariableDecl.setInSource(false);

              extendWithNode(iteratorVariableDecl);

              Node expressionNode = scan(expression, p);

              MethodAccessNode iteratorAccessNode =
                  new MethodAccessNode(iteratorSelect, expressionNode);
              iteratorAccessNode.setInSource(false);
              extendWithNode(iteratorAccessNode);
              MethodInvocationNode iteratorCallNode =
                  new MethodInvocationNode(iteratorCall, iteratorAccessNode,
                                           Collections.<Node>emptyList(), getCurrentPath());
              iteratorCallNode.setInSource(false);
              extendWithNode(iteratorCallNode);

              translateAssignment(iteratorVariable,
                                  new LocalVariableNode(iteratorVariable),
                                  iteratorCallNode);

              // Test the loop ending condition
              addLabelForNextNode(conditionStart);
              IdentifierTree iteratorUse1 =
                  treeBuilder.buildVariableUse(iteratorVariable);
              handleArtificialTree(iteratorUse1);

              LocalVariableNode iteratorReceiverNode =
                  new LocalVariableNode(iteratorUse1);
              iteratorReceiverNode.setInSource(false);
              extendWithNode(iteratorReceiverNode);

              MemberSelectTree hasNextSelect =
                  treeBuilder.buildHasNextMethodAccess(iteratorUse1);
              handleArtificialTree(hasNextSelect);

              MethodAccessNode hasNextAccessNode =
                  new MethodAccessNode(hasNextSelect, iteratorReceiverNode);
              hasNextAccessNode.setInSource(false);
              extendWithNode(hasNextAccessNode);

              MethodInvocationTree hasNextCall =
                  treeBuilder.buildMethodInvocation(hasNextSelect);
              handleArtificialTree(hasNextCall);

              MethodInvocationNode hasNextCallNode =
                  new MethodInvocationNode(hasNextCall, hasNextAccessNode,
                                           Collections.<Node>emptyList(), getCurrentPath());
              hasNextCallNode.setInSource(false);
              extendWithNode(hasNextCallNode);
              extendWithExtendedNode(new ConditionalJump(loopEntry, loopExit));

              // Loop body, starting with declaration of the loop iteration variable
              addLabelForNextNode(loopEntry);
              extendWithNode(new VariableDeclarationNode(variable));

              IdentifierTree iteratorUse2 =
                  treeBuilder.buildVariableUse(iteratorVariable);
              handleArtificialTree(iteratorUse2);

              LocalVariableNode iteratorReceiverNode2 =
                  new LocalVariableNode(iteratorUse2);
              iteratorReceiverNode2.setInSource(false);
              extendWithNode(iteratorReceiverNode2);

              MemberSelectTree nextSelect =
                  treeBuilder.buildNextMethodAccess(iteratorUse2);
              handleArtificialTree(nextSelect);

              MethodAccessNode nextAccessNode =
                  new MethodAccessNode(nextSelect, iteratorReceiverNode2);
              nextAccessNode.setInSource(false);
              extendWithNode(nextAccessNode);

              MethodInvocationTree nextCall =
                  treeBuilder.buildMethodInvocation(nextSelect);
              handleArtificialTree(nextCall);

              MethodInvocationNode nextCallNode =
                  new MethodInvocationNode(nextCall, nextAccessNode,
                                           Collections.<Node>emptyList(), getCurrentPath());
              nextCallNode.setInSource(false);
              extendWithNode(nextCallNode);

              translateAssignment(variable,
                                  new LocalVariableNode(variable),
                                  nextCall);

              if (statement != null) {
                  scan(statement, p);
              }

              // Loop back edge
              addLabelForNextNode(updateStart);
              extendWithExtendedNode(new UnconditionalJump(conditionStart));

          } else {
              // TODO: Shift any labels after the initialization of the
              // temporary array variable.

              VariableTree arrayVariable = createEnhancedForLoopArrayVariable(expression, variableElement);
              handleArtificialTree(arrayVariable);

              VariableDeclarationNode arrayVariableNode =
                  new VariableDeclarationNode(arrayVariable);
              arrayVariableNode.setInSource(false);
              extendWithNode(arrayVariableNode);
              Node expressionNode = scan(expression, p);

              translateAssignment(arrayVariable,
                                  new LocalVariableNode(arrayVariable),
                                  expressionNode);

              // Declare and initialize the loop index variable
              TypeMirror intType = types.getPrimitiveType(TypeKind.INT);

              LiteralTree zero =
                  treeBuilder.buildLiteral(new Integer(0));
              handleArtificialTree(zero);

              VariableTree indexVariable =
                  treeBuilder.buildVariableDecl(intType,
                                                uniqueName("index"),
                                                variableElement.getEnclosingElement(),
                                                zero);
              handleArtificialTree(indexVariable);
              VariableDeclarationNode indexVariableNode =
                  new VariableDeclarationNode(indexVariable);
              indexVariableNode.setInSource(false);
              extendWithNode(indexVariableNode);
              IntegerLiteralNode zeroNode =
                  extendWithNode(new IntegerLiteralNode(zero));

              translateAssignment(indexVariable,
                                  new LocalVariableNode(indexVariable),
                                  zeroNode);

              // Compare index to array length
              addLabelForNextNode(conditionStart);
              IdentifierTree indexUse1 =
                  treeBuilder.buildVariableUse(indexVariable);
              handleArtificialTree(indexUse1);
              LocalVariableNode indexNode1 =
                  new LocalVariableNode(indexUse1);
              indexNode1.setInSource(false);
              extendWithNode(indexNode1);

              IdentifierTree arrayUse1 =
                  treeBuilder.buildVariableUse(arrayVariable);
              handleArtificialTree(arrayUse1);
              LocalVariableNode arrayNode1 =
                  extendWithNode(new LocalVariableNode(arrayUse1));

              MemberSelectTree lengthSelect =
                  treeBuilder.buildArrayLengthAccess(arrayUse1);
              handleArtificialTree(lengthSelect);
              FieldAccessNode lengthAccessNode =
                  new FieldAccessNode(lengthSelect, arrayNode1);
              lengthAccessNode.setInSource(false);
              extendWithNode(lengthAccessNode);

              BinaryTree lessThan =
                  treeBuilder.buildLessThan(indexUse1, lengthSelect);
              handleArtificialTree(lessThan);

              LessThanNode lessThanNode =
                  new LessThanNode(lessThan, indexNode1, lengthAccessNode);
              lessThanNode.setInSource(false);
              extendWithNode(lessThanNode);
              extendWithExtendedNode(new ConditionalJump(loopEntry, loopExit));

              // Loop body, starting with declaration of the loop iteration variable
              addLabelForNextNode(loopEntry);
              extendWithNode(new VariableDeclarationNode(variable));

              IdentifierTree arrayUse2 =
                  treeBuilder.buildVariableUse(arrayVariable);
              handleArtificialTree(arrayUse2);
              LocalVariableNode arrayNode2 =
                  new LocalVariableNode(arrayUse2);
              arrayNode2.setInSource(false);
              extendWithNode(arrayNode2);

              IdentifierTree indexUse2 =
                  treeBuilder.buildVariableUse(indexVariable);
              handleArtificialTree(indexUse2);
              LocalVariableNode indexNode2 =
                  new LocalVariableNode(indexUse2);
              indexNode2.setInSource(false);
              extendWithNode(indexNode2);

              ArrayAccessTree arrayAccess =
                  treeBuilder.buildArrayAccess(arrayUse2, indexUse2);
              handleArtificialTree(arrayAccess);
              ArrayAccessNode arrayAccessNode =
                  new ArrayAccessNode(arrayAccess, arrayNode2, indexNode2);
              arrayAccessNode.setInSource(false);
              extendWithNode(arrayAccessNode);
              translateAssignment(variable,
                                  new LocalVariableNode(variable),
                                  arrayAccessNode);

              if (statement != null) {
                  scan(statement, p);
              }

              // Loop back edge
              addLabelForNextNode(updateStart);

              IdentifierTree indexUse3 =
                  treeBuilder.buildVariableUse(indexVariable);
              handleArtificialTree(indexUse3);
              LocalVariableNode indexNode3 =
                  new LocalVariableNode(indexUse3);
              indexNode3.setInSource(false);
              extendWithNode(indexNode3);

              LiteralTree oneTree = treeBuilder.buildLiteral(Integer.valueOf(1));
              handleArtificialTree(oneTree);
              Node one = new IntegerLiteralNode(oneTree);
              one.setInSource(false);
              extendWithNode(one);

              BinaryTree addOneTree = treeBuilder.buildBinary(intType, Tree.Kind.PLUS,
                      indexUse3, oneTree);
              handleArtificialTree(addOneTree);
              Node addOneNode = new NumericalAdditionNode(addOneTree, indexNode3, one);
              addOneNode.setInSource(false);
              extendWithNode(addOneNode);

              AssignmentTree assignTree = treeBuilder.buildAssignment(indexUse3, addOneTree);
              handleArtificialTree(assignTree);
              Node assignNode = new AssignmentNode(assignTree, indexNode3, addOneNode);
              assignNode.setInSource(false);
              extendWithNode(assignNode);

              extendWithExtendedNode(new UnconditionalJump(conditionStart));
          }

          // Loop exit
          addLabelForNextNode(loopExit);

          breakTargetL = oldBreakTargetL;
          continueTargetL = oldContinueTargetL;

          return null;
        }

        protected VariableTree createEnhancedForLoopIteratorVariable(MethodInvocationTree iteratorCall, VariableElement variableElement) {
            TypeMirror iteratorType = InternalUtils.typeOf(iteratorCall);

            // Declare and initialize a new, unique iterator variable
            VariableTree iteratorVariable =
                treeBuilder.buildVariableDecl(iteratorType, // annotatedIteratorTypeTree,
                                              uniqueName("iter"),
                                              variableElement.getEnclosingElement(),
                                              iteratorCall);
            return iteratorVariable;
        }

        protected VariableTree createEnhancedForLoopArrayVariable(ExpressionTree expression, VariableElement variableElement) {
            TypeMirror arrayType = InternalUtils.typeOf(expression);

            // Declare and initialize a temporary array variable
            VariableTree arrayVariable =
                    treeBuilder.buildVariableDecl(arrayType,
                            uniqueName("array"),
                            variableElement.getEnclosingElement(),
                            expression);
            return arrayVariable;
        }

        @Override
        public Node visitForLoop(ForLoopTree tree, Void p) {
            Name parentLabel = getLabel(getCurrentPath());

            Label conditionStart = new Label();
            Label loopEntry = new Label();
            Label loopExit = new Label();

            // If the loop is a labeled statement, then its continue
            // target is identical for continues with no label and
            // continues with the loop's label.
            Label updateStart;
            if (parentLabel != null) {
                updateStart = continueLabels.get(parentLabel);
            } else {
                updateStart = new Label();
            }

            Label oldBreakTargetL = breakTargetL;
            breakTargetL = loopExit;

            Label oldContinueTargetL = continueTargetL;
            continueTargetL = updateStart;

            // Initializer
            for (StatementTree init : tree.getInitializer()) {
                scan(init, p);
            }

            // Condition
            addLabelForNextNode(conditionStart);
            if (tree.getCondition() != null) {
                unbox(scan(tree.getCondition(), p));
                ConditionalJump cjump = new ConditionalJump(loopEntry, loopExit);
                extendWithExtendedNode(cjump);
            }

            // Loop body
            addLabelForNextNode(loopEntry);
            if (tree.getStatement() != null) {
                scan(tree.getStatement(), p);
            }

            // Update
            addLabelForNextNode(updateStart);
            for (ExpressionStatementTree update : tree.getUpdate()) {
                scan(update, p);
            }

            extendWithExtendedNode(new UnconditionalJump(conditionStart));

            // Loop exit
            addLabelForNextNode(loopExit);

            breakTargetL = oldBreakTargetL;
            continueTargetL = oldContinueTargetL;

            return null;
        }

        @Override
        public Node visitIdentifier(IdentifierTree tree, Void p) {
            Node node;
            if (TreeUtils.isFieldAccess(tree)) {
                Node receiver = getReceiver(tree,
                        TreeUtils.enclosingClass(getCurrentPath()));
                node = new FieldAccessNode(tree, receiver);
            } else {
                Element element = TreeUtils.elementFromUse(tree);
                switch (element.getKind()) {
                case ANNOTATION_TYPE:
                case CLASS:
                case ENUM:
                case INTERFACE:
                case TYPE_PARAMETER:
                    node = new ClassNameNode(tree);
                    break;
                case FIELD:
                    // Note that "this"/"super" is a field, but not a field access.
                    if (element.getSimpleName().contentEquals("this")) {
                        node = new ExplicitThisLiteralNode(tree);
                    } else {
                        node = new SuperNode(tree);
                    }
                    break;
                case EXCEPTION_PARAMETER:
                case LOCAL_VARIABLE:
                case RESOURCE_VARIABLE:
                case PARAMETER:
                    node = new LocalVariableNode(tree);
                    break;
                case PACKAGE:
                    node = new PackageNameNode(tree);
                    break;
                default:
                    throw new IllegalArgumentException(
                            "unexpected element kind : " + element.getKind());
                }
            }
            extendWithNode(node);
            return node;
        }

        @Override
        public Node visitIf(IfTree tree, Void p) {
            // all necessary labels
            Label thenEntry = new Label();
            Label elseEntry = new Label();
            Label endIf = new Label();

            // basic block for the condition
            unbox(scan(tree.getCondition(), p));

            ConditionalJump cjump = new ConditionalJump(thenEntry, elseEntry);
            extendWithExtendedNode(cjump);

            // then branch
            addLabelForNextNode(thenEntry);
            StatementTree thenStatement = tree.getThenStatement();
            scan(thenStatement, p);
            extendWithExtendedNode(new UnconditionalJump(endIf));

            // else branch
            addLabelForNextNode(elseEntry);
            StatementTree elseStatement = tree.getElseStatement();
            if (elseStatement != null) {
                scan(elseStatement, p);
            }

            // label the end of the if statement
            addLabelForNextNode(endIf);

            return null;
        }

        @Override
        public Node visitImport(ImportTree tree, Void p) {
            assert false : "ImportTree is unexpected in AST to CFG translation";
            return null;
        }

        @Override
        public Node visitArrayAccess(ArrayAccessTree tree, Void p) {
            Node array = scan(tree.getExpression(), p);
            Node index = unaryNumericPromotion(scan(tree.getIndex(), p));
            return extendWithNode(new ArrayAccessNode(tree, array, index));
        }

        @Override
        public Node visitLabeledStatement(LabeledStatementTree tree, Void p) {
            // This method can set the break target after generating all Nodes
            // in the contained statement, but it can't set the continue target,
            // which may be in the middle of a sequence of nodes. Labeled loops
            // must look up and use the continue Labels.
            Name labelName = tree.getLabel();

            Label breakL = new Label(labelName + "_break");
            Label continueL = new Label(labelName + "_continue");

            breakLabels.put(labelName, breakL);
            continueLabels.put(labelName, continueL);

            scan(tree.getStatement(), p);

            addLabelForNextNode(breakL);

            breakLabels.remove(labelName);
            continueLabels.remove(labelName);

            return null;
        }

        @Override
        public Node visitLiteral(LiteralTree tree, Void p) {
            Node r = null;
            switch (tree.getKind()) {
            case BOOLEAN_LITERAL:
                r = new BooleanLiteralNode(tree);
                break;
            case CHAR_LITERAL:
                r = new CharacterLiteralNode(tree);
                break;
            case DOUBLE_LITERAL:
                r = new DoubleLiteralNode(tree);
                break;
            case FLOAT_LITERAL:
                r = new FloatLiteralNode(tree);
                break;
            case INT_LITERAL:
                r = new IntegerLiteralNode(tree);
                break;
            case LONG_LITERAL:
                r = new LongLiteralNode(tree);
                break;
            case NULL_LITERAL:
                r = new NullLiteralNode(tree);
                break;
            case STRING_LITERAL:
                r = new StringLiteralNode(tree);
                break;
            default:
                assert false : "unexpected literal tree";
                break;
            }
            assert r != null : "unexpected literal tree";
            Node result = extendWithNode(r);
            return result;
        }

        @Override
        public Node visitMethod(MethodTree tree, Void p) {
            assert false : "MethodTree is unexpected in AST to CFG translation";
            return null;
        }

        @Override
        public Node visitModifiers(ModifiersTree tree, Void p) {
            assert false : "ModifiersTree is unexpected in AST to CFG translation";
            return null;
        }

        @Override
        public Node visitNewArray(NewArrayTree tree, Void p) {
            // see JLS 15.10

            ArrayType type = (ArrayType)InternalUtils.typeOf(tree);
            TypeMirror elemType = type.getComponentType();

            List<? extends ExpressionTree> dimensions = tree.getDimensions();
            List<? extends ExpressionTree> initializers = tree
                    .getInitializers();

            List<Node> dimensionNodes = new ArrayList<Node>();
            if (dimensions != null) {
                for (ExpressionTree dim : dimensions) {
                    dimensionNodes.add(unaryNumericPromotion(scan(dim, p)));
                }
            }

            List<Node> initializerNodes = new ArrayList<Node>();
            if (initializers != null) {
                for (ExpressionTree init : initializers) {
                    initializerNodes.add(assignConvert(scan(init, p), elemType));
                }
            }

            Node node = new ArrayCreationNode(tree, type, dimensionNodes,
                    initializerNodes);
            return extendWithNode(node);
        }

        @Override
        public Node visitNewClass(NewClassTree tree, Void p) {
            // see JLS 15.9

            Tree enclosingExpr = tree.getEnclosingExpression();
            if (enclosingExpr != null) {
                scan(enclosingExpr, p);
            }

            // We ignore any class body because its methods should
            // be visited separately.

            // Convert constructor arguments
            ExecutableElement constructor = TreeUtils.elementFromUse(tree);

            List<? extends ExpressionTree> actualExprs = tree.getArguments();

            List<Node> arguments = convertCallArguments(constructor,
                    actualExprs);

            Node constructorNode = scan(tree.getIdentifier(), p);

            Node node = new ObjectCreationNode(tree, constructorNode, arguments);

            Set<TypeMirror> thrownSet = new HashSet<>();
            // Add exceptions explicitly mentioned in the throws clause.
            List<? extends TypeMirror> thrownTypes = constructor.getThrownTypes();
            thrownSet.addAll(thrownTypes);
            // Add Throwable to account for unchecked exceptions
            TypeElement throwableElement = elements
                    .getTypeElement("java.lang.Throwable");
            thrownSet.add(throwableElement.asType());

            extendWithNodeWithExceptions(node, thrownSet);

            return node;
        }

        /**
         * Maps a {@code Tree} its directly enclosing {@code ParenthesizedTree} if one exists.
         *
         * This map is used by {@link CFGTranslationPhaseOne#addToLookupMap(Node)} to
         * associate a {@code ParenthesizedTree} with the dataflow {@code Node} that was used
         * during inference. This map is necessary because dataflow does
         * not create a {@code Node} for a {@code ParenthesizedTree.}
         */
        private final Map<Tree, ParenthesizedTree> parenMapping = new HashMap<>();

        @Override
        public Node visitParenthesized(ParenthesizedTree tree, Void p) {
            parenMapping.put(tree.getExpression(), tree);
            return scan(tree.getExpression(), p);
        }

        @Override
        public Node visitReturn(ReturnTree tree, Void p) {
            ExpressionTree ret = tree.getExpression();
            // TODO: also have a return-node if nothing is returned
            ReturnNode result = null;
            if (ret != null) {
                Node node = scan(ret, p);
                Tree enclosing = TreeUtils.enclosingOfKind(getCurrentPath(), new HashSet<Kind>(Arrays.asList(Kind.METHOD, Kind.LAMBDA_EXPRESSION)));
                if (enclosing.getKind() == Kind.LAMBDA_EXPRESSION) {
                    LambdaExpressionTree lambdaTree = (LambdaExpressionTree) enclosing;
                    TreePath lambdaTreePath = TreePath.getPath(getCurrentPath().getCompilationUnit(), lambdaTree);
                    Context ctx = ((JavacProcessingEnvironment)env).getContext();
                    Element overriddenElement = com.sun.tools.javac.code.Types.instance(ctx).findDescriptorSymbol(
                            ((Type)trees.getTypeMirror(lambdaTreePath)).tsym);

                    result = new ReturnNode(tree, node, env.getTypeUtils(), lambdaTree, (MethodSymbol)overriddenElement);
                } else {
                    result = new ReturnNode(tree, node, env.getTypeUtils(), (MethodTree)enclosing);
                }
                returnNodes.add(result);
                extendWithNode(result);
            }
            extendWithExtendedNode(new UnconditionalJump(regularExitLabel));
            // TODO: return statements should also flow to an enclosing finally block
            return result;
        }

        @Override
        public Node visitMemberSelect(MemberSelectTree tree, Void p) {
            Node expr = scan(tree.getExpression(), p);
            if (!TreeUtils.isFieldAccess(tree)) {
                // Could be a selector of a class or package
                Node result = null;
                Element element = TreeUtils.elementFromUse(tree);
                switch (element.getKind()) {
                case ANNOTATION_TYPE:
                case CLASS:
                case ENUM:
                case INTERFACE:
                    result = extendWithNode(new ClassNameNode(tree, expr));
                    break;
                case PACKAGE:
                    result = extendWithNode(new PackageNameNode(tree, (PackageNameNode) expr));
                    break;
                default:
                    assert false : "Unexpected element kind: " + element.getKind();
                    return null;
                }
                return result;
            }

            Node node = new FieldAccessNode(tree, expr);

            Element element = TreeUtils.elementFromUse(tree);
            if (ElementUtils.isStatic(element) ||
                expr instanceof ImplicitThisLiteralNode ||
                expr instanceof ExplicitThisLiteralNode) {
                // No NullPointerException can be thrown, use normal node
                extendWithNode(node);
            } else {
                TypeElement npeElement = elements
                    .getTypeElement("java.lang.NullPointerException");
                extendWithNodeWithException(node, npeElement.asType());
            }

            return node;
        }

        @Override
        public Node visitEmptyStatement(EmptyStatementTree tree, Void p) {
            return null;
        }

        @Override
        public Node visitSynchronized(SynchronizedTree tree, Void p) {
            // see JLS 14.19

            Node synchronizedExpr = scan(tree.getExpression(), p);
            SynchronizedNode synchronizedStartNode = new SynchronizedNode(tree, synchronizedExpr, true, env.getTypeUtils());
            extendWithNode(synchronizedStartNode);
            scan(tree.getBlock(), p);
            SynchronizedNode synchronizedEndNode = new SynchronizedNode(tree, synchronizedExpr, false, env.getTypeUtils());
            extendWithNode(synchronizedEndNode);

            return null;
        }

        @Override
        public Node visitThrow(ThrowTree tree, Void p) {
            Node expression = scan(tree.getExpression(), p);
            TypeMirror exception = expression.getType();
            ThrowNode throwsNode = new ThrowNode(tree, expression, env.getTypeUtils());
            NodeWithExceptionsHolder exNode = extendWithNodeWithException(
                    throwsNode, exception);
            exNode.setTerminatesExecution(true);
            return throwsNode;
        }

        @Override
        public Node visitCompilationUnit(CompilationUnitTree tree, Void p) {
            assert false : "CompilationUnitTree is unexpected in AST to CFG translation";
            return null;
        }

        @Override
        public Node visitTry(TryTree tree, Void p) {
            List<? extends CatchTree> catches = tree.getCatches();
            BlockTree finallyBlock = tree.getFinallyBlock();

            extendWithNode(new MarkerNode(tree, "start of try statement", env.getTypeUtils()));

            // TODO: Should we handle try-with-resources blocks by also generating code
            // for automatically closing the resources?
            List<? extends Tree> resources = tree.getResources();
            for (Tree resource : resources) {
                scan(resource, p);
            }

            List<Pair<TypeMirror, Label>> catchLabels = new ArrayList<>();
            for (CatchTree c : catches) {
                TypeMirror type = InternalUtils.typeOf(c.getParameter().getType());
                assert type != null : "exception parameters must have a type";
                catchLabels.add(Pair.of(type, new Label()));
            }

            Label finallyLabel = null;
            if (finallyBlock != null) {
                finallyLabel = new Label();
                tryStack.pushFrame(new TryFinallyFrame(finallyLabel));
            }

            Label doneLabel = new Label();

            tryStack.pushFrame(new TryCatchFrame(types, catchLabels));

            scan(tree.getBlock(), p);
            extendWithExtendedNode(new UnconditionalJump(firstNonNull(finallyLabel, doneLabel)));

            tryStack.popFrame();

            int catchIndex = 0;
            for (CatchTree c : catches) {
                addLabelForNextNode(catchLabels.get(catchIndex).second);
                scan(c, p);
                catchIndex++;
                extendWithExtendedNode(new UnconditionalJump(firstNonNull(finallyLabel, doneLabel)));
            }

            if (finallyLabel != null) {
                tryStack.popFrame();
                addLabelForNextNode(finallyLabel);
                scan(finallyBlock, p);

                TypeMirror throwableType =
                    elements.getTypeElement("java.lang.Throwable").asType();
                extendWithNodeWithException(new MarkerNode(tree, "end of finally block", env.getTypeUtils()),
                                            throwableType);
            }

            addLabelForNextNode(doneLabel);

            return null;
        }

        @Override
        public Node visitParameterizedType(ParameterizedTypeTree tree, Void p) {
            return extendWithNode(new ParameterizedTypeNode(tree));
        }

        @Override
        public Node visitUnionType(UnionTypeTree tree, Void p) {
            assert false : "UnionTypeTree is unexpected in AST to CFG translation";
            return null;
        }

        @Override
        public Node visitArrayType(ArrayTypeTree tree, Void p) {
            return extendWithNode(new ArrayTypeNode(tree));
        }

        @Override
        public Node visitTypeCast(TypeCastTree tree, Void p) {
            final Node operand = scan(tree.getExpression(), p);
            final TypeMirror type = InternalUtils.typeOf(tree.getType());
            final Node node = new TypeCastNode(tree, operand, type);
            final TypeElement cceElement = elements.getTypeElement("java.lang.ClassCastException");

            extendWithNodeWithException(node, cceElement.asType());
            return node;
        }

        @Override
        public Node visitPrimitiveType(PrimitiveTypeTree tree, Void p) {
            return extendWithNode(new PrimitiveTypeNode(tree));
        }

        @Override
        public Node visitTypeParameter(TypeParameterTree tree, Void p) {
            assert false : "TypeParameterTree is unexpected in AST to CFG translation";
            return null;
        }

        @Override
        public Node visitInstanceOf(InstanceOfTree tree, Void p) {
            Node operand = scan(tree.getExpression(), p);
            TypeMirror refType = InternalUtils.typeOf(tree.getType());
            InstanceOfNode node = new InstanceOfNode(tree, operand, refType,
                    types);
            extendWithNode(node);
            return node;
        }

        @Override
        public Node visitUnary(UnaryTree tree, Void p) {
            Node result = null;
            Tree.Kind kind = tree.getKind();
            switch (kind) {
            case BITWISE_COMPLEMENT:
            case UNARY_MINUS:
            case UNARY_PLUS: {
                // see JLS 15.14 and 15.15
                Node expr = scan(tree.getExpression(), p);
                expr = unaryNumericPromotion(expr);

                // TypeMirror exprType = InternalUtils.typeOf(tree);

                switch (kind) {
                case BITWISE_COMPLEMENT:
                    result = extendWithNode(new BitwiseComplementNode(tree,
                            expr));
                    break;
                case UNARY_MINUS:
                    result = extendWithNode(new NumericalMinusNode(tree, expr));
                    break;
                case UNARY_PLUS:
                    result = extendWithNode(new NumericalPlusNode(tree, expr));
                    break;
                default:
                    assert false;
                    break;
                }
                break;
            }

            case LOGICAL_COMPLEMENT: {
                // see JLS 15.15.6
                Node expr = scan(tree.getExpression(), p);
                result = extendWithNode(new ConditionalNotNode(tree,
                        unbox(expr)));
                break;
            }

            case POSTFIX_DECREMENT:
            case POSTFIX_INCREMENT: {
                ExpressionTree exprTree = tree.getExpression();
                TypeMirror exprType = InternalUtils.typeOf(exprTree);
                TypeMirror oneType = types.getPrimitiveType(TypeKind.INT);
                Node expr = scan(exprTree, p);

                TypeMirror promotedType = binaryPromotedType(exprType, oneType);

                LiteralTree oneTree = treeBuilder.buildLiteral(Integer.valueOf(1));
                handleArtificialTree(oneTree);

                Node exprRHS = binaryNumericPromotion(expr, promotedType);
                Node one = new IntegerLiteralNode(oneTree);
                one.setInSource(false);
                extendWithNode(one);
                one = binaryNumericPromotion(one, promotedType);

                BinaryTree operTree = treeBuilder.buildBinary(promotedType,
                        (kind == Tree.Kind.POSTFIX_INCREMENT ? Tree.Kind.PLUS : Tree.Kind.MINUS),
                        exprTree, oneTree);
                handleArtificialTree(operTree);
                Node operNode;
                if (kind == Tree.Kind.POSTFIX_INCREMENT) {
                    operNode = new NumericalAdditionNode(operTree, exprRHS, one);
                } else {
                    assert kind == Tree.Kind.POSTFIX_DECREMENT;
                    operNode = new NumericalSubtractionNode(operTree, exprRHS, one);
                }
                extendWithNode(operNode);

                Node narrowed = narrowAndBox(operNode, exprType);
                // TODO: By using the assignment as the result of the expression, we
                // act like a pre-increment/decrement.  Fix this by saving the initial
                // value of the expression in a temporary.
                AssignmentNode assignNode = new AssignmentNode(tree, expr, narrowed);
                extendWithNode(assignNode);
                result = assignNode;
                break;
            }
            case PREFIX_DECREMENT:
            case PREFIX_INCREMENT: {
                ExpressionTree exprTree = tree.getExpression();
                TypeMirror exprType = InternalUtils.typeOf(exprTree);
                TypeMirror oneType = types.getPrimitiveType(TypeKind.INT);
                Node expr = scan(exprTree, p);

                TypeMirror promotedType = binaryPromotedType(exprType, oneType);

                LiteralTree oneTree = treeBuilder.buildLiteral(Integer.valueOf(1));
                handleArtificialTree(oneTree);

                Node exprRHS = binaryNumericPromotion(expr, promotedType);
                Node one = new IntegerLiteralNode(oneTree);
                one.setInSource(false);
                extendWithNode(one);
                one = binaryNumericPromotion(one, promotedType);

                BinaryTree operTree = treeBuilder.buildBinary(promotedType,
                        (kind == Tree.Kind.PREFIX_INCREMENT ? Tree.Kind.PLUS : Tree.Kind.MINUS),
                        exprTree, oneTree);
                handleArtificialTree(operTree);
                Node operNode;
                if (kind == Tree.Kind.PREFIX_INCREMENT) {
                    operNode = new NumericalAdditionNode(operTree, exprRHS, one);
                } else {
                    assert kind == Tree.Kind.PREFIX_DECREMENT;
                    operNode = new NumericalSubtractionNode(operTree, exprRHS, one);
                }
                extendWithNode(operNode);

                Node narrowed = narrowAndBox(operNode, exprType);
                AssignmentNode assignNode = new AssignmentNode(tree, expr, narrowed);
                extendWithNode(assignNode);
                result = assignNode;
                break;
            }

            case OTHER:
            default:
                // special node NLLCHK
                if (tree.toString().startsWith("<*nullchk*>")) {
                    Node expr = scan(tree.getExpression(), p);
                    result = extendWithNode(new NullChkNode(tree, expr));
                    break;
                }

                assert false : "Unknown kind (" + kind
                        + ") of unary expression: " + tree;
            }

            return result;
        }

        @Override
        public Node visitVariable(VariableTree tree, Void p) {

            // see JLS 14.4

            boolean isField = getCurrentPath().getParentPath() != null
                    && getCurrentPath().getParentPath().getLeaf().getKind() == Kind.CLASS;
            Node node = null;

            ClassTree enclosingClass = TreeUtils
                    .enclosingClass(getCurrentPath());
            TypeElement classElem = TreeUtils
                    .elementFromDeclaration(enclosingClass);
            Node receiver = new ImplicitThisLiteralNode(classElem.asType());

            if (isField) {
                ExpressionTree initializer = tree.getInitializer();
                assert initializer != null;
                node = translateAssignment(
                        tree,
                        new FieldAccessNode(tree, TreeUtils.elementFromDeclaration(tree), receiver),
                        initializer);
            } else {
                // local variable definition
                VariableDeclarationNode decl = new VariableDeclarationNode(tree);
                extendWithNode(decl);

                // initializer

                ExpressionTree initializer = tree.getInitializer();
                if (initializer != null) {
                    node = translateAssignment(tree, new LocalVariableNode(tree, receiver),
                            initializer);
                }
            }

            return node;
        }

        @Override
        public Node visitWhileLoop(WhileLoopTree tree, Void p) {
            Name parentLabel = getLabel(getCurrentPath());

            Label loopEntry = new Label();
            Label loopExit = new Label();

            // If the loop is a labeled statement, then its continue
            // target is identical for continues with no label and
            // continues with the loop's label.
            Label conditionStart;
            if (parentLabel != null) {
                conditionStart = continueLabels.get(parentLabel);
            } else {
                conditionStart = new Label();
            }

            Label oldBreakTargetL = breakTargetL;
            breakTargetL = loopExit;

            Label oldContinueTargetL = continueTargetL;
            continueTargetL = conditionStart;

            // Condition
            addLabelForNextNode(conditionStart);
            if (tree.getCondition() != null) {
                unbox(scan(tree.getCondition(), p));
                ConditionalJump cjump = new ConditionalJump(loopEntry, loopExit);
                extendWithExtendedNode(cjump);
            }

            // Loop body
            addLabelForNextNode(loopEntry);
            if (tree.getStatement() != null) {
                scan(tree.getStatement(), p);
            }
            extendWithExtendedNode(new UnconditionalJump(conditionStart));

            // Loop exit
            addLabelForNextNode(loopExit);

            breakTargetL = oldBreakTargetL;
            continueTargetL = oldContinueTargetL;

            return null;
        }

        @Override
        public Node visitLambdaExpression(LambdaExpressionTree tree, Void p) {
            declaredLambdas.add(tree);
            Node node = new FunctionalInterfaceNode(tree);
            extendWithNode(node);
            return node;
        }

        @Override
        public Node visitMemberReference(MemberReferenceTree tree, Void p) {
            Tree enclosingExpr = tree.getQualifierExpression();
            if (enclosingExpr != null) {
                scan(enclosingExpr, p);
            }

            Node node = new FunctionalInterfaceNode(tree);
            extendWithNode(node);

            return node;
        }

        @Override
        public Node visitWildcard(WildcardTree tree, Void p) {
            assert false : "WildcardTree is unexpected in AST to CFG translation";
            return null;
        }

        @Override
        public Node visitOther(Tree tree, Void p) {
            assert false : "Unknown AST element encountered in AST to CFG translation.";
            return null;
        }
    }

    /**
     * A tuple with 4 named elements.
     */
    private interface TreeInfo {
        boolean isBoxed();
        boolean isNumeric();
        boolean isBoolean();
        TypeMirror unboxedType();
    }

    private static <A> A firstNonNull(A first, A second) {
        if (first != null) {
            return first;
        } else if (second != null) {
            return second;
        } else {
            throw new NullPointerException();
        }
    }

    /* --------------------------------------------------------- */
    /* Utility routines for debugging CFG building */
    /* --------------------------------------------------------- */

    /**
     * Print a set of {@link Block}s and the edges between them. This is useful
     * for examining the results of phase two.
     */
    protected static void printBlocks(Set<Block> blocks) {
        for (Block b : blocks) {
            System.out.print(b.hashCode() + ": " + b);
            switch (b.getType()) {
            case REGULAR_BLOCK:
            case SPECIAL_BLOCK: {
                Block succ = ((SingleSuccessorBlockImpl) b).getSuccessor();
                System.out.println(" -> "
                        + (succ != null ? succ.hashCode() : "||"));
                break;
            }
            case EXCEPTION_BLOCK: {
                Block succ = ((SingleSuccessorBlockImpl) b).getSuccessor();
                System.out.print(" -> "
                                 + (succ != null ? succ.hashCode() : "||") + " {");
                for (Map.Entry<TypeMirror, Set<Block>> entry : ((ExceptionBlockImpl) b).getExceptionalSuccessors().entrySet()) {
                    System.out.print(entry.getKey() + " : " + entry.getValue() + ", ");
                }
                System.out.println("}");
                break;
            }
            case CONDITIONAL_BLOCK: {
                Block tSucc = ((ConditionalBlockImpl) b).getThenSuccessor();
                Block eSucc = ((ConditionalBlockImpl) b).getElseSuccessor();
                System.out.println(" -> T "
                        + (tSucc != null ? tSucc.hashCode() : "||") + " F "
                        + (eSucc != null ? eSucc.hashCode() : "||"));
                break;
            }
            }
        }
    }
}
