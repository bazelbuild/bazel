package org.checkerframework.dataflow.analysis;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import javax.lang.model.element.Element;
import javax.lang.model.element.ExecutableElement;
import javax.lang.model.element.VariableElement;
import javax.lang.model.type.TypeKind;
import javax.lang.model.type.TypeMirror;

import org.checkerframework.dataflow.cfg.node.ArrayAccessNode;
import org.checkerframework.dataflow.cfg.node.ArrayCreationNode;
import org.checkerframework.dataflow.cfg.node.ClassNameNode;
import org.checkerframework.dataflow.cfg.node.ExplicitThisLiteralNode;
import org.checkerframework.dataflow.cfg.node.FieldAccessNode;
import org.checkerframework.dataflow.cfg.node.LocalVariableNode;
import org.checkerframework.dataflow.cfg.node.MethodInvocationNode;
import org.checkerframework.dataflow.cfg.node.NarrowingConversionNode;
import org.checkerframework.dataflow.cfg.node.Node;
import org.checkerframework.dataflow.cfg.node.StringConversionNode;
import org.checkerframework.dataflow.cfg.node.SuperNode;
import org.checkerframework.dataflow.cfg.node.ThisLiteralNode;
import org.checkerframework.dataflow.cfg.node.ValueLiteralNode;
import org.checkerframework.dataflow.cfg.node.WideningConversionNode;
import org.checkerframework.dataflow.util.HashCodeUtils;
import org.checkerframework.dataflow.util.PurityUtils;
import org.checkerframework.javacutil.AnnotationProvider;
import org.checkerframework.javacutil.ElementUtils;
import org.checkerframework.javacutil.TreeUtils;
import org.checkerframework.javacutil.TypesUtils;

import com.sun.tools.javac.code.Symbol.VarSymbol;

/**
 * Collection of classes and helper functions to represent Java expressions
 * about which the org.checkerframework.dataflow analysis can possibly infer facts. Expressions
 * include:
 * <ul>
 * <li>Field accesses (e.g., <em>o.f</em>)</li>
 * <li>Local variables (e.g., <em>l</em>)</li>
 * <li>This reference (e.g., <em>this</em>)</li>
 * <li>Pure method calls (e.g., <em>o.m()</em>)</li>
 * <li>Unknown other expressions to mark that something else was present.</li>
 * </ul>
 *
 * @author Stefan Heule
 *
 */
public class FlowExpressions {

    /**
     * @return the internal representation (as {@link FieldAccess}) of a
     *         {@link FieldAccessNode}. Can contain {@link Unknown} as receiver.
     */
    public static FieldAccess internalReprOfFieldAccess(
            AnnotationProvider provider, FieldAccessNode node) {
        Receiver receiver;
        Node receiverNode = node.getReceiver();
        if (node.isStatic()) {
            receiver = new ClassName(receiverNode.getType());
        } else {
            receiver = internalReprOf(provider, receiverNode);
        }
        return new FieldAccess(receiver, node);
    }

    /**
     * @return the internal representation (as {@link FieldAccess}) of a
     *         {@link FieldAccessNode}. Can contain {@link Unknown} as receiver.
     */
    public static ArrayAccess internalReprOfArrayAccess(
            AnnotationProvider provider, ArrayAccessNode node) {
        Receiver receiver = internalReprOf(provider, node.getArray());
        Receiver index = internalReprOf(provider, node.getIndex());
        return new ArrayAccess(node.getType(), receiver, index);
    }

    /**
     * We ignore operations such as widening and
     * narrowing when computing the internal representation.
     *
     * @return the internal representation (as {@link Receiver}) of any
     *         {@link Node}. Might contain {@link Unknown}.
     */
    public static Receiver internalReprOf(AnnotationProvider provider,
            Node receiverNode) {
        return internalReprOf(provider, receiverNode, false);
    }

    /**
     * We ignore operations such as widening and
     * narrowing when computing the internal representation.
     *
     * @return the internal representation (as {@link Receiver}) of any
     *         {@link Node}. Might contain {@link Unknown}.
     */
    public static Receiver internalReprOf(AnnotationProvider provider,
            Node receiverNode, boolean allowNonDeterministic) {
        Receiver receiver = null;
        if (receiverNode instanceof FieldAccessNode) {
            FieldAccessNode fan = (FieldAccessNode) receiverNode;

            if (fan.getFieldName().equals("this")) {
                // For some reason, "className.this" is considered a field access.
                // We right this wrong here.
                receiver = new ThisReference(fan.getReceiver().getType());
            } else if (fan.getFieldName().equals("class")) {
                // "className.class" is considered a field access. This makes sense,
                // since .class is similar to a field access which is the equivalent
                // of a call to getClass(). However for the purposes of dataflow
                // analysis, and value stores, this is the equivalent of a ClassNameNode.
                receiver = new ClassName(fan.getReceiver().getType());
            }  else {
                receiver = internalReprOfFieldAccess(provider, fan);
            }
        } else if (receiverNode instanceof ExplicitThisLiteralNode) {
            receiver = new ThisReference(receiverNode.getType());
        } else if (receiverNode instanceof ThisLiteralNode) {
            receiver = new ThisReference(receiverNode.getType());
        } else if (receiverNode instanceof SuperNode) {
            receiver = new ThisReference(receiverNode.getType());
        } else if (receiverNode instanceof LocalVariableNode) {
            LocalVariableNode lv = (LocalVariableNode) receiverNode;
            receiver = new LocalVariable(lv);
        } else if (receiverNode instanceof ArrayAccessNode) {
            ArrayAccessNode a = (ArrayAccessNode) receiverNode;
            receiver = internalReprOfArrayAccess(provider, a);
        } else if (receiverNode instanceof StringConversionNode) {
            // ignore string conversion
            return internalReprOf(provider,
                    ((StringConversionNode) receiverNode).getOperand());
        } else if (receiverNode instanceof WideningConversionNode) {
            // ignore widening
            return internalReprOf(provider,
                    ((WideningConversionNode) receiverNode).getOperand());
        } else if (receiverNode instanceof NarrowingConversionNode) {
            // ignore narrowing
            return internalReprOf(provider,
                    ((NarrowingConversionNode) receiverNode).getOperand());
        } else if (receiverNode instanceof ClassNameNode) {
            ClassNameNode cn = (ClassNameNode) receiverNode;
            receiver = new ClassName(cn.getType());
        } else if (receiverNode instanceof ValueLiteralNode) {
            ValueLiteralNode vn = (ValueLiteralNode) receiverNode;
            receiver = new ValueLiteral(vn.getType(), vn);
        } else if (receiverNode instanceof ArrayCreationNode) {
            ArrayCreationNode an = (ArrayCreationNode)receiverNode;
            receiver = new ArrayCreation(an.getType(), an.getDimensions(), an.getInitializers());
        } else if (receiverNode instanceof MethodInvocationNode) {
            MethodInvocationNode mn = (MethodInvocationNode) receiverNode;
            ExecutableElement invokedMethod = TreeUtils.elementFromUse(mn
                    .getTree());

            // check if this represents a boxing operation of a constant, in which
            // case we treat the method call as deterministic, because there is no way
            // to behave differently in two executions where two constants are being used.
            boolean considerDeterministic = false;
            if (invokedMethod.toString().equals("valueOf(long)")
                    && mn.getTarget().getReceiver().toString().equals("Long")) {
                Node arg = mn.getArgument(0);
                if (arg instanceof ValueLiteralNode) {
                    considerDeterministic = true;
                }
            }

            if (PurityUtils.isDeterministic(provider, invokedMethod) || allowNonDeterministic || considerDeterministic) {
                List<Receiver> parameters = new ArrayList<>();
                for (Node p : mn.getArguments()) {
                    parameters.add(internalReprOf(provider, p));
                }
                Receiver methodReceiver;
                if (ElementUtils.isStatic(invokedMethod)) {
                    methodReceiver = new ClassName(mn.getTarget().getReceiver()
                            .getType());
                } else {
                    methodReceiver = internalReprOf(provider, mn.getTarget()
                            .getReceiver());
                }
                receiver = new MethodCall(mn.getType(), invokedMethod,
                        methodReceiver, parameters);
            }
        }

        if (receiver == null) {
            receiver = new Unknown(receiverNode.getType());
        }
        return receiver;
    }

    public static abstract class Receiver {
        protected final TypeMirror type;

        public Receiver(TypeMirror type) {
            assert type != null;
            this.type = type;
        }

        public TypeMirror getType() {
            return type;
        }

        public abstract boolean containsOfClass(Class<? extends FlowExpressions.Receiver> clazz);

        public boolean containsUnknown() {
            return containsOfClass(Unknown.class);
        }

        /**
         * Returns true if and only if the value this expression stands for
         * cannot be changed by a method call. This is the case for local
         * variables, the self reference as well as final field accesses for
         * whose receiver {@link #isUnmodifiableByOtherCode} is true.
         */
        public abstract boolean isUnmodifiableByOtherCode();

        /**
         * @return true if and only if the two receiver are syntactically
         *         identical
         */
        public boolean syntacticEquals(Receiver other) {
            return other == this;
        }

        /**
         * @return true if and only if this receiver contains a receiver that is
         *         syntactically equal to {@code other}.
         */
        public boolean containsSyntacticEqualReceiver(Receiver other) {
            return syntacticEquals(other);
        }

        /**
         * Returns true if and only if {@code other} appears anywhere in this
         * receiver or an expression appears in this receiver such that
         * {@code other} might alias this expression, and that expression is
         * modifiable.
         *
         * <p>
         * This is always true, except for cases where the Java type information
         * prevents aliasing and none of the subexpressions can alias 'other'.
         */
        public boolean containsModifiableAliasOf(Store<?> store, Receiver other) {
            return this.equals(other) || store.canAlias(this, other);
        }
    }

    public static class FieldAccess extends Receiver {
        protected Receiver receiver;
        protected VariableElement field;

        public Receiver getReceiver() {
            return receiver;
        }

        public VariableElement getField() {
            return field;
        }

        public FieldAccess(Receiver receiver, FieldAccessNode node) {
            super(node.getType());
            this.receiver = receiver;
            this.field = node.getElement();
        }

        public FieldAccess(Receiver receiver, TypeMirror type,
                VariableElement fieldElement) {
            super(type);
            this.receiver = receiver;
            this.field = fieldElement;
        }

        public boolean isFinal() {
            return ElementUtils.isFinal(field);
        }

        public boolean isStatic() {
            return ElementUtils.isStatic(field);
        }

        @Override
        public boolean equals(Object obj) {
            if (obj == null || !(obj instanceof FieldAccess)) {
                return false;
            }
            FieldAccess fa = (FieldAccess) obj;
            return fa.getField().equals(getField())
                    && fa.getReceiver().equals(getReceiver());
        }

        @Override
        public int hashCode() {
            return HashCodeUtils.hash(getField(), getReceiver());
        }

        @Override
        public boolean containsModifiableAliasOf(Store<?> store, Receiver other) {
            return super.containsModifiableAliasOf(store, other)
                    || receiver.containsModifiableAliasOf(store, other);
        }

        @Override
        public boolean containsSyntacticEqualReceiver(Receiver other) {
            return syntacticEquals(other)
                    || receiver.containsSyntacticEqualReceiver(other);
        }

        @Override
        public boolean syntacticEquals(Receiver other) {
            if (!(other instanceof FieldAccess)) {
                return false;
            }
            FieldAccess fa = (FieldAccess) other;
            return super.syntacticEquals(other)
                    || fa.getField().equals(getField())
                    && fa.getReceiver().syntacticEquals(getReceiver());
        }

        @Override
        public String toString() {
            return receiver + "." + field;
        }

        @Override
        public boolean containsOfClass(Class<? extends FlowExpressions.Receiver> clazz) {
            return getClass().equals(clazz) || receiver.containsOfClass(clazz);
        }

        @Override
        public boolean isUnmodifiableByOtherCode() {
            return isFinal() && getReceiver().isUnmodifiableByOtherCode();
        }
    }

    public static class ThisReference extends Receiver {
        public ThisReference(TypeMirror type) {
            super(type);
        }

        @Override
        public boolean equals(Object obj) {
            return obj != null && obj instanceof ThisReference;
        }

        @Override
        public int hashCode() {
            return HashCodeUtils.hash(0);
        }

        @Override
        public String toString() {
            return "this";
        }

        @Override
        public boolean containsOfClass(Class<? extends FlowExpressions.Receiver> clazz) {
            return getClass().equals(clazz);
        }

        @Override
        public boolean syntacticEquals(Receiver other) {
            return other instanceof ThisReference;
        }

        @Override
        public boolean isUnmodifiableByOtherCode() {
            return true;
        }

        @Override
        public boolean containsModifiableAliasOf(Store<?> store, Receiver other) {
            return false; // 'this' is not modifiable
        }
    }

    /**
     * A ClassName represents the occurrence of a class as part of a static
     * field access or method invocation.
     */
    public static class ClassName extends Receiver {
        public ClassName(TypeMirror type) {
            super(type);
        }

        @Override
        public boolean equals(Object obj) {
            if (obj == null || !(obj instanceof ClassName)) {
                return false;
            }
            ClassName other = (ClassName) obj;
            return getType().toString().equals(other.getType().toString());
        }

        @Override
        public int hashCode() {
            return HashCodeUtils.hash(getType().toString());
        }

        @Override
        public String toString() {
            return getType().toString();
        }

        @Override
        public boolean containsOfClass(Class<? extends FlowExpressions.Receiver> clazz) {
            return getClass().equals(clazz);
        }

        @Override
        public boolean syntacticEquals(Receiver other) {
            return this.equals(other);
        }

        @Override
        public boolean isUnmodifiableByOtherCode() {
            return true;
        }

        @Override
        public boolean containsModifiableAliasOf(Store<?> store, Receiver other) {
            return false; // not modifiable
        }
    }

    public static class Unknown extends Receiver {
        public Unknown(TypeMirror type) {
            super(type);
        }

        @Override
        public boolean equals(Object obj) {
            return obj == this;
        }

        @Override
        public int hashCode() {
            return System.identityHashCode(this);
        }

        @Override
        public String toString() {
            return "?";
        }

        @Override
        public boolean containsModifiableAliasOf(Store<?> store, Receiver other) {
            return true;
        }

        @Override
        public boolean containsOfClass(Class<? extends FlowExpressions.Receiver> clazz) {
            return getClass().equals(clazz);
        }

        @Override
        public boolean isUnmodifiableByOtherCode() {
            return false;
        }

    }

    public static class LocalVariable extends Receiver {
        protected Element element;

        public LocalVariable(LocalVariableNode localVar) {
            super(localVar.getType());
            this.element = localVar.getElement();
        }

        public LocalVariable(Element elem) {
            super(ElementUtils.getType(elem));
            this.element = elem;
        }

        @Override
        public boolean equals(Object obj) {
            if (obj == null || !(obj instanceof LocalVariable)) {
                return false;
            }
            LocalVariable other = (LocalVariable) obj;
            VarSymbol vs = (VarSymbol) element;
            VarSymbol vsother = (VarSymbol) other.element;
            // Use type.unannotatedType().toString().equals(...) instead of Types.isSameType(...)
            // because Types requires a processing environment, and FlowExpressions is
            // designed to be independent of processing environment.  See also
            // calls to getType().toString() in FlowExpressions.
            return vsother.name.contentEquals(vs.name) &&
                   vsother.type.unannotatedType().toString().equals(vs.type.unannotatedType().toString()) &&
                   vsother.owner.toString().equals(vs.owner.toString());
        }

        public Element getElement() {
            return element;
        }

        @Override
        public int hashCode() {
            VarSymbol vs = (VarSymbol) element;
            return HashCodeUtils.hash(vs.name.toString(),
                    vs.type.unannotatedType().toString(),
                    vs.owner.toString());
        }

        @Override
        public String toString() {
            return element.toString();
        }

        @Override
        public boolean containsOfClass(Class<? extends FlowExpressions.Receiver> clazz) {
            return getClass().equals(clazz);
        }

        @Override
        public boolean syntacticEquals(Receiver other) {
            if (!(other instanceof LocalVariable)) {
                return false;
            }
            LocalVariable l = (LocalVariable) other;
            return l.equals(this);
        }

        @Override
        public boolean containsSyntacticEqualReceiver(Receiver other) {
            return syntacticEquals(other);
        }

        @Override
        public boolean isUnmodifiableByOtherCode() {
            return true;
        }
    }

    public static class ValueLiteral extends Receiver {

        protected final Object value;

        public ValueLiteral(TypeMirror type, ValueLiteralNode node) {
            super(type);
            value = node.getValue();
        }

        public ValueLiteral(TypeMirror type, Object value) {
            super(type);
            this.value = value;
        }

        @Override
        public boolean containsOfClass(Class<? extends FlowExpressions.Receiver> clazz) {
            return getClass().equals(clazz);
        }

        @Override
        public boolean isUnmodifiableByOtherCode() {
            return true;
        }

        @Override
        public boolean equals(Object obj) {
            if (obj == null || !(obj instanceof ValueLiteral)) {
                return false;
            }
            ValueLiteral other = (ValueLiteral) obj;
            if (value == null) {
                return type.toString().equals(other.type.toString())
                        && other.value == null;
            }
            return type.toString().equals(other.type.toString())
                    && value.equals(other.value);
        }

        @Override
        public String toString() {
            if (TypesUtils.isString(type)) {
                return "\"" + value + "\"";
            } else if (type.getKind() == TypeKind.LONG) {
                return value.toString() + "L";
            }
            return value == null ? "null" : value.toString();
        }

        @Override
        public int hashCode() {
            return HashCodeUtils.hash(value, type.toString());
        }

        @Override
        public boolean syntacticEquals(Receiver other) {
            return this.equals(other);
        }

        @Override
        public boolean containsModifiableAliasOf(Store<?> store, Receiver other) {
            return false; // not modifiable
        }
    }

    /**
     * A method call.
     */
    public static class MethodCall extends Receiver {

        protected final Receiver receiver;
        protected final List<Receiver> parameters;
        protected final ExecutableElement method;

        public MethodCall(TypeMirror type, ExecutableElement method,
                Receiver receiver, List<Receiver> parameters) {
            super(type);
            this.receiver = receiver;
            this.parameters = parameters;
            this.method = method;
        }

        @Override
        public boolean containsOfClass(Class<? extends FlowExpressions.Receiver> clazz) {
            if (getClass().equals(clazz)) {
                return true;
            }
            if (receiver.containsOfClass(clazz)) {
                return true;
            }
            for (Receiver p : parameters) {
                if (p.containsOfClass(clazz)) {
                    return true;
                }
            }
            return false;
        }

        /**
         * @return the method call receiver (for inspection only - do not modify)
         */
        public Receiver getReceiver() {
            return receiver;
        }

        /**
         * @return the method call parameters (for inspection only - do not modify any of the parameters)
         */
        public List<Receiver> getParameters() {
            return Collections.unmodifiableList(parameters);
        }

        /**
         * @return the ExecutableElement for the method call
         */
        public ExecutableElement getElement() {
            return method;
        }

        @Override
        public boolean isUnmodifiableByOtherCode() {
            return false;
        }

        @Override
        public boolean containsSyntacticEqualReceiver(Receiver other) {
            return syntacticEquals(other) || receiver.syntacticEquals(other);
        }

        @Override
        public boolean syntacticEquals(Receiver other) {
            if (!(other instanceof MethodCall)) {
                return false;
            }
            MethodCall otherMethod = (MethodCall) other;
            if (!receiver.syntacticEquals(otherMethod.receiver)) {
                return false;
            }
            if (parameters.size() != otherMethod.parameters.size()) {
                return false;
            }
            int i = 0;
            for (Receiver p : parameters) {
                if (!p.syntacticEquals(otherMethod.parameters.get(i))) {
                    return false;
                }
                i++;
            }
            return method.equals(otherMethod.method);
        }

        public boolean containsSyntacticEqualParameter(LocalVariable var) {
            for (Receiver p : parameters) {
                if (p.containsSyntacticEqualReceiver(var)) {
                    return true;
                }
            }
            return false;
        }

        @Override
        public boolean containsModifiableAliasOf(Store<?> store, Receiver other) {
            if (receiver.containsModifiableAliasOf(store, other)) {
                return true;
            }
            for (Receiver p : parameters) {
                if (p.containsModifiableAliasOf(store, other)) {
                    return true;
                }
            }
            return false; // the method call itself is not modifiable
        }

        @Override
        public boolean equals(Object obj) {
            if (obj == null || !(obj instanceof MethodCall)) {
                return false;
            }
            MethodCall other = (MethodCall) obj;
            int i = 0;
            for (Receiver p : parameters) {
                if (!p.equals(other.parameters.get(i))) {
                    return false;
                }
                i++;
            }
            return receiver.equals(other.receiver)
                    && method.equals(other.method);
        }

        @Override
        public int hashCode() {
            int hash = HashCodeUtils.hash(method, receiver);
            for (Receiver p : parameters) {
                hash = HashCodeUtils.hash(hash, p);
            }
            return hash;
        }

        @Override
        public String toString() {
            StringBuilder result = new StringBuilder();
            result.append(receiver.toString());
            result.append(".");
            String methodName = method.getSimpleName().toString();
            result.append(methodName);
            result.append("(");
            boolean first = true;
            for (Receiver p : parameters) {
                if (!first) {
                    result.append(", ");
                }
                result.append(p.toString());
                first = false;
            }
            result.append(")");
            return result.toString();
        }
    }

    /**
     * A deterministic method call.
     */
    public static class ArrayAccess extends Receiver {

        protected final Receiver receiver;
        protected final Receiver index;

        public ArrayAccess(TypeMirror type, Receiver receiver, Receiver index) {
            super(type);
            this.receiver = receiver;
            this.index = index;
        }

        @Override
        public boolean containsOfClass(Class<? extends FlowExpressions.Receiver> clazz) {
            if (getClass().equals(clazz)) {
                return true;
            }
            if (receiver.containsOfClass(clazz)) {
                return true;
            }
            return index.containsOfClass(clazz);
        }

        public Receiver getReceiver() {
            return receiver;
        }

        public Receiver getIndex() {
            return index;
        }

        @Override
        public boolean isUnmodifiableByOtherCode() {
            return false;
        }

        @Override
        public boolean containsSyntacticEqualReceiver(Receiver other) {
            return syntacticEquals(other) || receiver.syntacticEquals(other)
                    || index.syntacticEquals(other);
        }

        @Override
        public boolean syntacticEquals(Receiver other) {
            if (!(other instanceof ArrayAccess)) {
                return false;
            }
            ArrayAccess otherArrayAccess = (ArrayAccess) other;
            if (!receiver.syntacticEquals(otherArrayAccess.receiver)) {
                return false;
            }
            return index.syntacticEquals(otherArrayAccess.index);
        }

        @Override
        public boolean containsModifiableAliasOf(Store<?> store, Receiver other) {
            if (receiver.containsModifiableAliasOf(store, other)) {
                return true;
            }
            return index.containsModifiableAliasOf(store, other);
        }

        @Override
        public boolean equals(Object obj) {
            if (obj == null || !(obj instanceof ArrayAccess)) {
                return false;
            }
            ArrayAccess other = (ArrayAccess) obj;
            return receiver.equals(other.receiver) && index.equals(other.index);
        }

        @Override
        public int hashCode() {
            return HashCodeUtils.hash(receiver, index);
        }

        @Override
        public String toString() {
            StringBuilder result = new StringBuilder();
            result.append(receiver.toString());
            result.append("[");
            result.append(index.toString());
            result.append("]");
            return result.toString();
        }
    }

    public static class ArrayCreation extends Receiver {

        protected List<Node> dimensions;
        protected List<Node> initializers;

        public ArrayCreation(TypeMirror type, List<Node> dimensions, List<Node> initializers) {
            super(type);
            this.dimensions = dimensions;
            this.initializers = initializers;
        }

        public List<Node> getDimensions() {
            return dimensions;
        }

        public List<Node> getInitializers() {
            return initializers;
        }

        @Override
        public boolean containsOfClass(Class<? extends Receiver> clazz) {
            for (Node n : dimensions) {
                if (n.getClass().equals(clazz)) return true;
            }
            for (Node n : initializers) {
                if (n.getClass().equals(clazz)) return true;
            }
            return false;
        }

        @Override
        public boolean isUnmodifiableByOtherCode() {
            return false;
        }

        @Override
        public int hashCode() {
            final int prime = 31;
            int result = 1;
            result = prime * result + ((dimensions == null) ? 0 : dimensions.hashCode());
            result = prime * result + ((initializers == null) ? 0 : initializers.hashCode());
            result = prime * result + HashCodeUtils.hash(getType().toString());
            return result;
        }

        @Override
        public boolean equals(Object obj) {
            if (obj == null || !(obj instanceof ArrayCreation)) {
                return false;
            }
            ArrayCreation other = (ArrayCreation) obj;
            return this.dimensions.equals(other.getDimensions())
                    && this.initializers.equals(other.getInitializers())
                    && getType().toString().equals(other.getType().toString());
        }

        @Override
        public boolean syntacticEquals(Receiver other) {
            return this.equals(other);
        }

        @Override
        public boolean containsSyntacticEqualReceiver(Receiver other) {
            return syntacticEquals(other);
        }

        @Override
        public String toString() {
            StringBuffer sb = new StringBuffer();
            sb.append("new " + type);
            if (!dimensions.isEmpty()) {
                boolean needComma = false;
                sb.append(" (");
                for (Node dim : dimensions) {
                    if (needComma) {
                        sb.append(", ");
                    }
                    sb.append(dim);
                    needComma = true;
                }
                sb.append(")");
            }
            if (!initializers.isEmpty()) {
                boolean needComma = false;
                sb.append(" = {");
                for (Node init : initializers) {
                    if (needComma) {
                        sb.append(", ");
                    }
                    sb.append(init);
                    needComma = true;
                }
                sb.append("}");
            }
            return sb.toString();
        }
    }
}
