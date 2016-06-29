package org.checkerframework.dataflow.cfg;

import com.sun.source.tree.ClassTree;
import com.sun.source.tree.LambdaExpressionTree;
import com.sun.source.tree.MethodTree;
import com.sun.source.tree.Tree;

/**
 * Represents an abstract syntax tree of type {@link Tree} that underlies a
 * given control flow graph.
 *
 * @author Stefan Heule
 *
 */
public abstract class UnderlyingAST {
    public enum Kind {
        /** The underlying code is a whole method */
        METHOD,
        /** The underlying code is a lambda expression */
        LAMBDA,

        /**
         * The underlying code is an arbitrary Java statement or expression
         */
        ARBITRARY_CODE,
    }

    protected final Kind kind;

    public UnderlyingAST(Kind kind) {
        this.kind = kind;
    }

    /**
     * @return the code that corresponds to the CFG
     */
    abstract public Tree getCode();

    public Kind getKind() {
        return kind;
    }

    /**
     * If the underlying AST is a method.
     */
    public static class CFGMethod extends UnderlyingAST {

        /** The method declaration */
        protected final MethodTree method;

        /** The class tree this method belongs to. */
        protected final ClassTree classTree;

        public CFGMethod(MethodTree method, ClassTree classTree) {
            super(Kind.METHOD);
            this.method = method;
            this.classTree = classTree;
        }

        @Override
        public Tree getCode() {
            return method.getBody();
        }

        public MethodTree getMethod() {
            return method;
        }

        public ClassTree getClassTree() {
            return classTree;
        }

        @Override
        public String toString() {
            return "CFGMethod(\n" + method + "\n)";
        }
    }

    /**
     * If the underlying AST is a lambda.
     */
    public static class CFGLambda extends UnderlyingAST {

        private final LambdaExpressionTree lambda;

        public CFGLambda(LambdaExpressionTree lambda) {
            super(Kind.LAMBDA);
            this.lambda = lambda;
        }

        @Override
        public Tree getCode() {
            return lambda.getBody();
        }

        public LambdaExpressionTree getLambdaTree() {
            return lambda;
        }

        @Override
        public String toString() {
            return "CFGLambda(\n" + lambda + "\n)";
        }
    }

    /**
     * If the underlying AST is a statement or expression.
     */
    public static class CFGStatement extends UnderlyingAST {

        protected final Tree code;

        /** The class tree this method belongs to. */
        protected final ClassTree classTree;

        public CFGStatement(Tree code, ClassTree classTree) {
            super(Kind.ARBITRARY_CODE);
            this.code = code;
            this.classTree = classTree;
        }

        @Override
        public Tree getCode() {
            return code;
        }

        public ClassTree getClassTree() {
            return classTree;
        }

        @Override
        public String toString() {
            return "CFGStatement(\n" + code + "\n)";
        }
    }
}
