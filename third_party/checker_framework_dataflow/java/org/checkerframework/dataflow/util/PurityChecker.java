package org.checkerframework.dataflow.util;

/*>>>
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.checker.compilermsgs.qual.CompilerMessageKey;
*/

import org.checkerframework.dataflow.qual.Deterministic;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.Pure.Kind;
import org.checkerframework.dataflow.qual.SideEffectFree;

import org.checkerframework.javacutil.AnnotationProvider;
import org.checkerframework.javacutil.InternalUtils;
import org.checkerframework.javacutil.Pair;
import org.checkerframework.javacutil.TreeUtils;

import java.util.ArrayList;
import java.util.Collection;
import java.util.EnumSet;
import java.util.List;

import javax.lang.model.element.Element;

import com.sun.source.tree.ArrayAccessTree;
import com.sun.source.tree.AssertTree;
import com.sun.source.tree.AssignmentTree;
import com.sun.source.tree.BinaryTree;
import com.sun.source.tree.BlockTree;
import com.sun.source.tree.BreakTree;
import com.sun.source.tree.CaseTree;
import com.sun.source.tree.CatchTree;
import com.sun.source.tree.ClassTree;
import com.sun.source.tree.CompoundAssignmentTree;
import com.sun.source.tree.ConditionalExpressionTree;
import com.sun.source.tree.ContinueTree;
import com.sun.source.tree.DoWhileLoopTree;
import com.sun.source.tree.EmptyStatementTree;
import com.sun.source.tree.EnhancedForLoopTree;
import com.sun.source.tree.ExpressionStatementTree;
import com.sun.source.tree.ExpressionTree;
import com.sun.source.tree.ForLoopTree;
import com.sun.source.tree.IdentifierTree;
import com.sun.source.tree.IfTree;
import com.sun.source.tree.InstanceOfTree;
import com.sun.source.tree.LabeledStatementTree;
import com.sun.source.tree.LambdaExpressionTree;
import com.sun.source.tree.LiteralTree;
import com.sun.source.tree.MemberReferenceTree;
import com.sun.source.tree.MemberSelectTree;
import com.sun.source.tree.MethodInvocationTree;
import com.sun.source.tree.NewArrayTree;
import com.sun.source.tree.NewClassTree;
import com.sun.source.tree.ParenthesizedTree;
import com.sun.source.tree.ReturnTree;
import com.sun.source.tree.SwitchTree;
import com.sun.source.tree.SynchronizedTree;
import com.sun.source.tree.ThrowTree;
import com.sun.source.tree.Tree;
import com.sun.source.tree.TryTree;
import com.sun.source.tree.TypeCastTree;
import com.sun.source.tree.UnaryTree;
import com.sun.source.tree.VariableTree;
import com.sun.source.tree.WhileLoopTree;
import com.sun.source.util.SimpleTreeVisitor;
import com.sun.tools.javac.tree.TreeScanner;

/**
 * A visitor that determines the purity (as defined by {@link
 * org.checkerframework.dataflow.qual.SideEffectFree}, {@link org.checkerframework.dataflow.qual.Deterministic},
 * and {@link org.checkerframework.dataflow.qual.Pure}) of a statement or expression.  The
 * entry point is method {@link #checkPurity}.
 *
 * @see SideEffectFree
 * @see Deterministic
 * @see Pure
 *
 * @author Stefan Heule
 *
 */
public class PurityChecker {

    /**
     * Compute whether the given statement is
     * side-effect-free, deterministic, or both.
     * Returns a result that can be queried.
     */
    public static PurityResult checkPurity(Tree statement,
            AnnotationProvider annoProvider, boolean assumeSideEffectFree) {
        PurityCheckerHelper helper = new PurityCheckerHelper(annoProvider, assumeSideEffectFree);
        PurityResult res = helper.scan(statement, new PurityResult());
        return res;
    }

    /**
     * Result of the {@link PurityChecker}.
     * Can be queried queried regarding whether a given tree was
     * side-effect-free, deterministic, or both; also gives reasons if
     * the answer is "no".
     */
    public static class PurityResult {

        protected final List<Pair<Tree, String>> notSeFreeReasons;
        protected final List<Pair<Tree, String>> notDetReasons;
        protected final List<Pair<Tree, String>> notBothReasons;
        protected EnumSet<Pure.Kind> types;

        public PurityResult() {
            notSeFreeReasons = new ArrayList<>();
            notDetReasons = new ArrayList<>();
            notBothReasons = new ArrayList<>();
            types = EnumSet.allOf(Pure.Kind.class);
        }

        public EnumSet<Pure.Kind> getTypes() {
            return types;
        }

        /** Is the method pure w.r.t. a given set of types? */
        public boolean isPure(Collection<Kind> kinds) {
            return types.containsAll(kinds);
        }

        /**
         * Get the {@code reason}s why the method is not side-effect-free.
         */
        public List<Pair<Tree, String>> getNotSeFreeReasons() {
            return notSeFreeReasons;
        }

        /**
         * Add {@code reason} as a reason why the method is not side-effect
         * free.
         */
        public void addNotSeFreeReason(Tree t, String msgId) {
            notSeFreeReasons.add(Pair.of(t, msgId));
            types.remove(Kind.SIDE_EFFECT_FREE);
        }

        /**
         * Get the {@code reason}s why the method is not deterministic.
         */
        public List<Pair<Tree, String>> getNotDetReasons() {
            return notDetReasons;
        }

        /**
         * Add {@code reason} as a reason why the method is not deterministic.
         */
        public void addNotDetReason(Tree t, String msgId) {
            notDetReasons.add(Pair.of(t, msgId));
            types.remove(Kind.DETERMINISTIC);
        }

        /**
         * Get the {@code reason}s why the method is not both side-effect-free
         * and deterministic.
         */
        public List<Pair<Tree, String>> getNotBothReasons() {
            return notBothReasons;
        }

        /**
         * Add {@code reason} as a reason why the method is not both side-effect
         * free and deterministic.
         */
        public void addNotBothReason(Tree t, String msgId) {
            notBothReasons.add(Pair.of(t, msgId));
            types.remove(Kind.DETERMINISTIC);
            types.remove(Kind.SIDE_EFFECT_FREE);
        }
    }

    /**
     * Helper class to keep {@link PurityChecker}'s interface clean. The
     * implementation is heavily based on {@link TreeScanner}, but some parts of
     * the AST are skipped (such as types or modifiers). Furthermore, scanning
     * works differently in that the input parameter (usually named {@code p})
     * gets "threaded through", instead of using {@code reduce}.
     */
    protected static class PurityCheckerHelper
            extends SimpleTreeVisitor<PurityResult, PurityResult> {

        protected final AnnotationProvider annoProvider;
        /** True if all methods should be assumed to be @SideEffectFree,
         * for the purposes of org.checkerframework.dataflow analysis. */
        private final boolean assumeSideEffectFree;
        protected /*@Nullable*/ List<Element> methodParameter;

        public PurityCheckerHelper(AnnotationProvider annoProvider, boolean assumeSideEffectFree) {
            this.annoProvider = annoProvider;
            this.assumeSideEffectFree = assumeSideEffectFree;
        }

        /**
         * Scan a single node.
         */
        public PurityResult scan(Tree node, PurityResult p) {
            return node == null ? p : node.accept(this, p);
        }

        /**
         * Scan a list of nodes.
         */
        public PurityResult scan(Iterable<? extends Tree> nodes, PurityResult p) {
            PurityResult r = p;
            if (nodes != null) {
                for (Tree node : nodes) {
                    r = scan(node, r);
                }
            }
            return r;
        }

        @Override
        protected PurityResult defaultAction(Tree node, PurityResult p) {
            assert false : "this type of tree is unexpected here";
            return null;
        }

        @Override
        public PurityResult visitClass(ClassTree node, PurityResult p) {
            return p;
        }

        @Override
        public PurityResult visitVariable(VariableTree node, PurityResult p) {
            return scan(node.getInitializer(), p);
        }

        @Override
        public PurityResult visitEmptyStatement(EmptyStatementTree node,
                PurityResult p) {
            return p;
        }

        @Override
        public PurityResult visitBlock(BlockTree node, PurityResult p) {
            return scan(node.getStatements(), p);
        }

        @Override
        public PurityResult visitDoWhileLoop(DoWhileLoopTree node,
                PurityResult p) {
            PurityResult r = scan(node.getStatement(), p);
            r = scan(node.getCondition(), r);
            return r;
        }

        @Override
        public PurityResult visitWhileLoop(WhileLoopTree node, PurityResult p) {
            PurityResult r = scan(node.getCondition(), p);
            r = scan(node.getStatement(), r);
            return r;
        }

        @Override
        public PurityResult visitForLoop(ForLoopTree node, PurityResult p) {
            PurityResult r = scan(node.getInitializer(), p);
            r = scan(node.getCondition(), r);
            r = scan(node.getUpdate(), r);
            r = scan(node.getStatement(), r);
            return r;
        }

        @Override
        public PurityResult visitEnhancedForLoop(EnhancedForLoopTree node,
                PurityResult p) {
            PurityResult r = scan(node.getVariable(), p);
            r = scan(node.getExpression(), r);
            r = scan(node.getStatement(), r);
            return r;
        }

        @Override
        public PurityResult visitLabeledStatement(LabeledStatementTree node,
                PurityResult p) {
            return scan(node.getStatement(), p);
        }

        @Override
        public PurityResult visitSwitch(SwitchTree node, PurityResult p) {
            PurityResult r = scan(node.getExpression(), p);
            r = scan(node.getCases(), r);
            return r;
        }

        @Override
        public PurityResult visitCase(CaseTree node, PurityResult p) {
            PurityResult r = scan(node.getExpression(), p);
            r = scan(node.getStatements(), r);
            return r;
        }

        @Override
        public PurityResult visitSynchronized(SynchronizedTree node,
                PurityResult p) {
            PurityResult r = scan(node.getExpression(), p);
            r = scan(node.getBlock(), r);
            return r;
        }

        @Override
        public PurityResult visitTry(TryTree node, PurityResult p) {
            PurityResult r = scan(node.getResources(), p);
            r = scan(node.getBlock(), r);
            r = scan(node.getCatches(), r);
            r = scan(node.getFinallyBlock(), r);
            return r;
        }

        @Override
        public PurityResult visitCatch(CatchTree node, PurityResult p) {
            p.addNotDetReason(node, "catch");
            PurityResult r = scan(node.getParameter(), p);
            r = scan(node.getBlock(), r);
            return r;
        }

        @Override
        public PurityResult visitConditionalExpression(
                ConditionalExpressionTree node, PurityResult p) {
            PurityResult r = scan(node.getCondition(), p);
            r = scan(node.getTrueExpression(), r);
            r = scan(node.getFalseExpression(), r);
            return r;
        }

        @Override
        public PurityResult visitIf(IfTree node, PurityResult p) {
            PurityResult r = scan(node.getCondition(), p);
            r = scan(node.getThenStatement(), r);
            r = scan(node.getElseStatement(), r);
            return r;
        }

        @Override
        public PurityResult visitExpressionStatement(
                ExpressionStatementTree node, PurityResult p) {
            return scan(node.getExpression(), p);
        }

        @Override
        public PurityResult visitBreak(BreakTree node, PurityResult p) {
            return p;
        }

        @Override
        public PurityResult visitContinue(ContinueTree node, PurityResult p) {
            return p;
        }

        @Override
        public PurityResult visitReturn(ReturnTree node, PurityResult p) {
            return scan(node.getExpression(), p);
        }

        @Override
        public PurityResult visitThrow(ThrowTree node, PurityResult p) {
            return scan(node.getExpression(), p);
        }

        @Override
        public PurityResult visitAssert(AssertTree node, PurityResult p) {
            PurityResult r = scan(node.getCondition(), p);
            r = scan(node.getDetail(), r);
            return r;
        }

        @Override
        public PurityResult visitMethodInvocation(MethodInvocationTree node,
                PurityResult p) {
            Element elt = TreeUtils.elementFromUse(node);
            String reason = "call";
            if (!PurityUtils.hasPurityAnnotation(annoProvider, elt)) {
                p.addNotBothReason(node, reason);
            } else {
                boolean det = PurityUtils.isDeterministic(annoProvider, elt);
                boolean seFree = (assumeSideEffectFree
                                  || PurityUtils.isSideEffectFree(annoProvider,
                                                                  elt));
                if (!det && !seFree) {
                    p.addNotBothReason(node, reason);
                } else if (!det) {
                    p.addNotDetReason(node, reason);
                } else if (!seFree) {
                    p.addNotSeFreeReason(node, reason);
                }
            }
            PurityResult r = scan(node.getMethodSelect(), p);
            r = scan(node.getArguments(), r);
            return r;
        }

        @Override
        public PurityResult visitNewClass(NewClassTree node, PurityResult p) {
            Element methodElement = InternalUtils.symbol(node);
            boolean sideEffectFree = (assumeSideEffectFree
                                      || PurityUtils.isSideEffectFree(annoProvider,
                                                                      methodElement));
            if (sideEffectFree) {
                p.addNotDetReason(node, "object.creation");
            } else {
                p.addNotBothReason(node, "object.creation");
            }
            PurityResult r = scan(node.getEnclosingExpression(), p);
            r = scan(node.getArguments(), r);
            r = scan(node.getClassBody(), r);
            return r;
        }

        @Override
        public PurityResult visitNewArray(NewArrayTree node, PurityResult p) {
            PurityResult r = scan(node.getDimensions(), p);
            r = scan(node.getInitializers(), r);
            return r;
        }

        @Override
        public PurityResult visitLambdaExpression(LambdaExpressionTree node,
                PurityResult p) {
            PurityResult r = scan(node.getParameters(), p);
            r = scan(node.getBody(), r);
            return r;
        }

        @Override
        public PurityResult visitParenthesized(ParenthesizedTree node,
                PurityResult p) {
            return scan(node.getExpression(), p);
        }

        @Override
        public PurityResult visitAssignment(AssignmentTree node, PurityResult p) {
            ExpressionTree variable = node.getVariable();
            p = assignmentCheck(p, variable);
            PurityResult r = scan(variable, p);
            r = scan(node.getExpression(), r);
            return r;
        }

        protected PurityResult assignmentCheck(PurityResult p,
                ExpressionTree variable) {
            if (TreeUtils.isFieldAccess(variable)) {
                // rhs is a field access
                p.addNotBothReason(variable, "assign.field");
            } else if (variable instanceof ArrayAccessTree) {
                // rhs is array access
                p.addNotBothReason(variable, "assign.array");
            } else {
                // rhs is a local variable
                assert isLocalVariable(variable);
            }
            return p;
        }

        protected boolean isLocalVariable(ExpressionTree variable) {
            return variable instanceof IdentifierTree
                    && !TreeUtils.isFieldAccess(variable);
        }

        @Override
        public PurityResult visitCompoundAssignment(
                CompoundAssignmentTree node, PurityResult p) {
            ExpressionTree variable = node.getVariable();
            p = assignmentCheck(p, variable);
            PurityResult r = scan(variable, p);
            r = scan(node.getExpression(), r);
            return r;
        }

        @Override
        public PurityResult visitUnary(UnaryTree node, PurityResult p) {
            return scan(node.getExpression(), p);
        }

        @Override
        public PurityResult visitBinary(BinaryTree node, PurityResult p) {
            PurityResult r = scan(node.getLeftOperand(), p);
            r = scan(node.getRightOperand(), r);
            return r;
        }

        @Override
        public PurityResult visitTypeCast(TypeCastTree node, PurityResult p) {
            PurityResult r = scan(node.getExpression(), p);
            return r;
        }

        @Override
        public PurityResult visitInstanceOf(InstanceOfTree node, PurityResult p) {
            PurityResult r = scan(node.getExpression(), p);
            return r;
        }

        @Override
        public PurityResult visitArrayAccess(ArrayAccessTree node,
                PurityResult p) {
            PurityResult r = scan(node.getExpression(), p);
            r = scan(node.getIndex(), r);
            return r;
        }

        @Override
        public PurityResult visitMemberSelect(MemberSelectTree node,
                PurityResult p) {
            return scan(node.getExpression(), p);
        }

        @Override
        public PurityResult visitMemberReference(MemberReferenceTree node,
                PurityResult p) {
            assert false : "this type of tree is unexpected here";
            return null;
        }

        @Override
        public PurityResult visitIdentifier(IdentifierTree node, PurityResult p) {
            return p;
        }

        @Override
        public PurityResult visitLiteral(LiteralTree node, PurityResult p) {
            return p;
        }
    }

}
