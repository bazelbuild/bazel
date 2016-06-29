package org.checkerframework.dataflow.cfg.node;

/*>>>
import org.checkerframework.checker.nullness.qual.Nullable;
*/

import com.sun.source.tree.LambdaExpressionTree;
import com.sun.tools.javac.code.Symbol.MethodSymbol;
import org.checkerframework.dataflow.cfg.node.AssignmentContext.LambdaReturnContext;
import org.checkerframework.dataflow.cfg.node.AssignmentContext.MethodReturnContext;
import org.checkerframework.dataflow.util.HashCodeUtils;

import java.util.Collection;
import java.util.Collections;

import javax.lang.model.type.TypeKind;
import javax.lang.model.util.Types;

import com.sun.source.tree.MethodTree;
import com.sun.source.tree.ReturnTree;

/**
 * A node for a return statement:
 *
 * <pre>
 *   return
 *   return <em>expression</em>
 * </pre>
 *
 * @author Stefan Heule
 *
 */
public class ReturnNode extends Node {

    protected ReturnTree tree;
    protected /*@Nullable*/ Node result;

    public ReturnNode(ReturnTree t, /*@Nullable*/ Node result, Types types, MethodTree methodTree) {
        super(types.getNoType(TypeKind.NONE));
        this.result = result;
        tree = t;
        result.setAssignmentContext(new MethodReturnContext(methodTree));
    }

    public ReturnNode(ReturnTree t, /*@Nullable*/ Node result, Types types, LambdaExpressionTree lambda, MethodSymbol methodSymbol) {
        super(types.getNoType(TypeKind.NONE));
        this.result = result;
        tree = t;
        result.setAssignmentContext(new LambdaReturnContext(methodSymbol));
    }


    public Node getResult() {
        return result;
    }

    @Override
    public ReturnTree getTree() {
        return tree;
    }

    @Override
    public <R, P> R accept(NodeVisitor<R, P> visitor, P p) {
        return visitor.visitReturn(this, p);
    }

    @Override
    public String toString() {
        if (result != null) {
            return "return " + result;
        }
        return "return";
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == null || !(obj instanceof ReturnNode)) {
            return false;
        }
        ReturnNode other = (ReturnNode) obj;
        if ((result == null) != (other.result == null)) {
            return false;
        }
        return (result == null || result.equals(other.result));
    }

    @Override
    public int hashCode() {
        return HashCodeUtils.hash(result);
    }

    @Override
    public Collection<Node> getOperands() {
        if (result == null) {
            return Collections.emptyList();
        } else {
            return Collections.singletonList(result);
        }
    }

}
