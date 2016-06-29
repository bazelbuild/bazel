package org.checkerframework.dataflow.cfg.node;

import java.util.Collection;
import java.util.Collections;

import javax.lang.model.type.TypeMirror;

import org.checkerframework.dataflow.util.HashCodeUtils;

/**
 * A node for a reference to 'this', either implicit or explicit.
 *
 * <pre>
 *   <em>this</em>
 * </pre>
 *
 * @author Stefan Heule
 * @author Charlie Garrett
 *
 */
public abstract class ThisLiteralNode extends Node {

    public ThisLiteralNode(TypeMirror type) {
        super(type);
    }

    public String getName() {
        return "this";
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == null || !(obj instanceof ThisLiteralNode)) {
            return false;
        }
        return true;
    }

    @Override
    public int hashCode() {
        return HashCodeUtils.hash(getName());
    }

    @Override
    public Collection<Node> getOperands() {
        return Collections.emptyList();
    }
}
