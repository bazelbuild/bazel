package org.checkerframework.dataflow.cfg.node;

import com.sun.source.tree.LiteralTree;
import java.util.Collection;
import java.util.Collections;
import org.checkerframework.dataflow.util.HashCodeUtils;
import org.checkerframework.javacutil.InternalUtils;

/*>>>
import org.checkerframework.checker.nullness.qual.Nullable;
*/

/**
 * A node for a literals that have some form of value:
 *
 * <ul>
 *   <li>integer literal
 *   <li>long literal
 *   <li>char literal
 *   <li>string literal
 *   <li>float literal
 *   <li>double literal
 *   <li>boolean literal
 *   <li>null literal
 * </ul>
 *
 * @author Stefan Heule
 */
public abstract class ValueLiteralNode extends Node {

    protected final LiteralTree tree;

    /** @return the value of the literal */
    public abstract /*@Nullable*/ Object getValue();

    public ValueLiteralNode(LiteralTree tree) {
        super(InternalUtils.typeOf(tree));
        this.tree = tree;
    }

    @Override
    public LiteralTree getTree() {
        return tree;
    }

    @Override
    public String toString() {
        return String.valueOf(getValue());
    }

    /** Compare the value of this nodes. */
    @Override
    public boolean equals(Object obj) {
        if (obj == null || !(obj instanceof ValueLiteralNode)) {
            return false;
        }
        ValueLiteralNode other = (ValueLiteralNode) obj;
        Object val = getValue();
        Object otherVal = other.getValue();
        return ((val == null || otherVal == null) && val == otherVal) || val.equals(otherVal);
    }

    @Override
    public int hashCode() {
        return HashCodeUtils.hash(getValue());
    }

    @Override
    public Collection<Node> getOperands() {
        return Collections.emptyList();
    }
}
