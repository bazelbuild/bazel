package org.checkerframework.dataflow.cfg.node;

import javax.lang.model.element.Element;
import javax.lang.model.element.ExecutableElement;

import org.checkerframework.javacutil.TreeUtils;

import com.sun.source.tree.ExpressionTree;
import com.sun.source.tree.MethodTree;
import com.sun.source.tree.Tree;
import com.sun.source.tree.VariableTree;

/**
 * An assignment context for a node, which represents the place to which the
 * node with this context is 'assigned' to. An 'assignment' (as we use the term
 * here) can occur for Java assignments, method calls (for all the actual
 * parameters which get assigned to their formal parameters) or method return
 * statements.
 *
 * <p>
 * The main use of {@link AssignmentContext} is to be able to get the declared
 * type of the left-hand side of the assignment for proper type-refinement.
 *
 * @author Stefan Heule
 */
public abstract class AssignmentContext {

    /**
     * An assignment context for an assignment 'lhs = rhs'.
     */
    public static class AssignmentLhsContext extends AssignmentContext {

        protected final Node node;

        public AssignmentLhsContext(Node node) {
            this.node = node;
        }

        @Override
        public Element getElementForType() {
            Tree tree = node.getTree();
            if (tree == null) {
                return null;
            } else if (tree instanceof ExpressionTree) {
                return TreeUtils.elementFromUse((ExpressionTree) tree);
            } else if (tree instanceof VariableTree) {
                return TreeUtils.elementFromDeclaration((VariableTree) tree);
            } else {
                assert false : "unexpected tree";
                return null;
            }
        }

        @Override
        public Tree getContextTree() {
            return node.getTree();
        }
    }

    /**
     * An assignment context for a method parameter.
     */
    public static class MethodParameterContext extends AssignmentContext {

        protected final ExecutableElement method;
        protected final int paramNum;

        public MethodParameterContext(ExecutableElement method, int paramNum) {
            this.method = method;
            this.paramNum = paramNum;
        }

        @Override
        public Element getElementForType() {
            return method.getParameters().get(paramNum);
        }

        @Override
        public Tree getContextTree() {
            // TODO: what is the right assignment context? We might not have
            // a tree for the invoked method.
            return null;
        }
    }

    /**
     * An assignment context for method return statements.
     */
    public static class MethodReturnContext extends AssignmentContext {

        protected final ExecutableElement method;
        protected final Tree ret;

        public MethodReturnContext(MethodTree method) {
            this.method = TreeUtils.elementFromDeclaration(method);
            this.ret = method.getReturnType();
        }

        @Override
        public Element getElementForType() {
            return method;
        }

        @Override
        public Tree getContextTree() {
            return ret;
        }
    }

    /**
     * An assignment context for lambda return statements.
     */
    public static class LambdaReturnContext extends AssignmentContext {

        protected final ExecutableElement method;

        public LambdaReturnContext(ExecutableElement method) {
            this.method = method;
        }

        @Override
        public Element getElementForType() {
            return method;
        }

        @Override
        public Tree getContextTree() {
            // TODO: what is the right assignment context? We might not have
            // a tree for the invoked method.
            return null;
        }
    }

    /**
     * Returns an {@link Element} that has the type of this assignment context.
     */
    public abstract Element getElementForType();

    public abstract Tree getContextTree();
}
