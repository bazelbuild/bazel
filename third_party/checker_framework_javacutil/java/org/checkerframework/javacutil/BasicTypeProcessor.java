package org.checkerframework.javacutil;

import javax.lang.model.element.TypeElement;

import com.sun.source.tree.CompilationUnitTree;

import com.sun.source.util.TreePath;
import com.sun.source.util.TreePathScanner;

/**
 * Process the types in an AST in a trivial manner, with hooks for derived classes
 * to actually do something.
 */
public abstract class BasicTypeProcessor extends AbstractTypeProcessor {
    /** The source tree that's being scanned. */
    protected CompilationUnitTree currentRoot;

    /**
     * Create a TreePathScanner at the given root.
     */
    protected abstract TreePathScanner<?, ?> createTreePathScanner(CompilationUnitTree root);

    /**
     * Visit the tree path for the type element.
     */
    @Override
    public void typeProcess(TypeElement e, TreePath p) {
        currentRoot = p.getCompilationUnit();

        TreePathScanner<?, ?> scanner = null;
        try {
            scanner = createTreePathScanner(currentRoot);
            scanner.scan(p, null);
        } catch (Throwable t) {
            System.err.println("BasicTypeProcessor.typeProcess: unexpected Throwable (" +
                    t.getClass().getSimpleName() + ")  when processing "
                    + currentRoot.getSourceFile().getName() +
                    (t.getMessage()!=null ? "; message: " + t.getMessage() : ""));
        }
    }

}
