package org.checkerframework.dataflow.util;

import org.checkerframework.javacutil.TypesUtils;

import org.checkerframework.dataflow.cfg.node.ConditionalOrNode;
import org.checkerframework.dataflow.cfg.node.Node;

import com.sun.source.tree.Tree;
import com.sun.tools.javac.code.Type;
import com.sun.tools.javac.tree.JCTree;

/**
 * A utility class to operate on a given {@link Node}.
 *
 * @author Stefan Heule
 *
 */
public class NodeUtils {

    /**
     * @return true iff {@code node} corresponds to a boolean typed
     *         expression (either the primitive type {@code boolean}, or
     *         class type {@link java.lang.Boolean})
     */
    public static boolean isBooleanTypeNode(Node node) {

        if (node instanceof ConditionalOrNode) {
            return true;
        }

        // not all nodes have an associated tree, but those are all not of a
        // boolean type.
        Tree tree = node.getTree();
        if (tree == null) {
            return false;
        }

        Type type = ((JCTree) tree).type;
        if (TypesUtils.isBooleanType(type)) {
            return true;
        }

        return false;
    }
}
