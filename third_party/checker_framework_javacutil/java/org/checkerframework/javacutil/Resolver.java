package org.checkerframework.javacutil;

import static com.sun.tools.javac.code.Kinds.VAR;

import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;

import javax.annotation.processing.ProcessingEnvironment;
import javax.lang.model.element.Element;
import javax.lang.model.element.ElementKind;
import javax.lang.model.element.VariableElement;
import javax.lang.model.type.TypeMirror;

import com.sun.source.util.TreePath;
import com.sun.source.util.Trees;
import com.sun.tools.javac.api.JavacScope;
import com.sun.tools.javac.code.Symbol;
import com.sun.tools.javac.code.Symbol.TypeSymbol;
import com.sun.tools.javac.code.Type;
import com.sun.tools.javac.comp.AttrContext;
import com.sun.tools.javac.comp.DeferredAttr;
import com.sun.tools.javac.comp.Env;
import com.sun.tools.javac.comp.Resolve;
import com.sun.tools.javac.processing.JavacProcessingEnvironment;
import com.sun.tools.javac.util.Context;
import com.sun.tools.javac.util.List;
import com.sun.tools.javac.util.Log;
import com.sun.tools.javac.util.Name;
import com.sun.tools.javac.util.Names;

/**
 * A Utility class to find symbols corresponding to string references.
 */
public class Resolver {
    private final Resolve resolve;
    private final Names names;
    private final Trees trees;
    private final Log log;

    private final Method FIND_METHOD;
    private final Method FIND_IDENT_IN_TYPE;
    private final Method FIND_IDENT_IN_PACKAGE;
    private final Method FIND_TYPE;

    public Resolver(ProcessingEnvironment env) {
        Context context = ((JavacProcessingEnvironment) env).getContext();
        this.resolve = Resolve.instance(context);
        this.names = Names.instance(context);
        this.trees = Trees.instance(env);
        this.log = Log.instance(context);

        try {
            FIND_METHOD = Resolve.class.getDeclaredMethod("findMethod",
                    Env.class, Type.class, Name.class, List.class, List.class,
                    boolean.class, boolean.class, boolean.class);
            FIND_METHOD.setAccessible(true);

            FIND_IDENT_IN_TYPE = Resolve.class.getDeclaredMethod(
                    "findIdentInType", Env.class, Type.class, Name.class,
                    int.class);
            FIND_IDENT_IN_TYPE.setAccessible(true);

            FIND_IDENT_IN_PACKAGE = Resolve.class.getDeclaredMethod(
                    "findIdentInPackage", Env.class, TypeSymbol.class, Name.class,
                    int.class);
            FIND_IDENT_IN_PACKAGE.setAccessible(true);

            FIND_TYPE = Resolve.class.getDeclaredMethod(
                    "findType", Env.class, Name.class);
            FIND_TYPE.setAccessible(true);
        } catch (Exception e) {
            Error err = new AssertionError(
                    "Compiler 'Resolve' class doesn't contain required 'find' method");
            err.initCause(e);
            throw err;
        }
    }

    /**
     * Finds the field with name {@code name} in a given type.
     *
     * <p>
     * The method adheres to all the rules of Java's scoping (while also
     * considering the imports) for name resolution.
     *
     * @param name
     *            The name of the field.
     * @param type
     *            The type of the receiver (i.e., the type in which to look for
     *            the field).
     * @param path
     *            The tree path to the local scope.
     * @return The element for the field.
     */
    public VariableElement findField(String name, TypeMirror type, TreePath path) {
        Log.DiagnosticHandler discardDiagnosticHandler =
            new Log.DiscardDiagnosticHandler(log);
        try {
            JavacScope scope = (JavacScope) trees.getScope(path);
            Env<AttrContext> env = scope.getEnv();
            Element res = wrapInvocation(FIND_IDENT_IN_TYPE, env, type,
                    names.fromString(name), VAR);
            if (res.getKind() == ElementKind.FIELD) {
                return (VariableElement) res;
            } else {
                // Most likely didn't find the field and the Element is a SymbolNotFoundError
                return null;
            }
        } finally {
            log.popDiagnosticHandler(discardDiagnosticHandler);
        }
    }

    /**
     * Finds the class literal with name {@code name}.
     *
     * <p>
     * The method adheres to all the rules of Java's scoping (while also
     * considering the imports) for name resolution.
     *
     * @param name
     *            The name of the class.
     * @param path
     *            The tree path to the local scope.
     * @return The element for the class.
     */
    public Element findClass(String name, TreePath path) {
        Log.DiagnosticHandler discardDiagnosticHandler =
            new Log.DiscardDiagnosticHandler(log);
        try {
            JavacScope scope = (JavacScope) trees.getScope(path);
            Env<AttrContext> env = scope.getEnv();
            return wrapInvocation(FIND_TYPE, env, names.fromString(name));
        } finally {
            log.popDiagnosticHandler(discardDiagnosticHandler);
        }
    }

    /**
     * Finds the method element for a given name and list of expected parameter
     * types.
     *
     * <p>
     * The method adheres to all the rules of Java's scoping (while also
     * considering the imports) for name resolution.
     *
     * @param methodName
     *            Name of the method to find.
     * @param receiverType
     *            Type of the receiver of the method
     * @param path
     *            Tree path.
     * @return The method element (if found).
     */
    public Element findMethod(String methodName, TypeMirror receiverType,
            TreePath path, java.util.List<TypeMirror> argumentTypes) {
        Log.DiagnosticHandler discardDiagnosticHandler =
            new Log.DiscardDiagnosticHandler(log);
        try {
            JavacScope scope = (JavacScope) trees.getScope(path);
            Env<AttrContext> env = scope.getEnv();

            Type site = (Type) receiverType;
            Name name = names.fromString(methodName);
            List<Type> argtypes = List.nil();
            for (TypeMirror a : argumentTypes) {
                argtypes = argtypes.append((Type) a);
            }
            List<Type> typeargtypes = List.nil();
            boolean allowBoxing = true;
            boolean useVarargs = false;
            boolean operator = true;

            try {
                // For some reason we have to set our own method context, which is rather ugly.
                // TODO: find a nicer way to do this.
                Object methodContext = buildMethodContext();
                Object oldContext = getField(resolve, "currentResolutionContext");
                setField(resolve, "currentResolutionContext", methodContext);
                Element result = wrapInvocation(FIND_METHOD, env, site, name, argtypes,
                    typeargtypes, allowBoxing, useVarargs, operator);
                setField(resolve, "currentResolutionContext", oldContext);
                return result;
            } catch (Throwable t) {
                Error err = new AssertionError("Unexpected Reflection error");
                err.initCause(t);
                throw err;
            }
        } finally {
            log.popDiagnosticHandler(discardDiagnosticHandler);
        }
    }

    /**
     * Build an instance of {@code Resolve$MethodResolutionContext}.
     */
    protected Object buildMethodContext() throws ClassNotFoundException,
            InstantiationException, IllegalAccessException,
            InvocationTargetException, NoSuchFieldException {
        // Class is not accessible, instantiate reflectively.
        Class<?> methCtxClss = Class.forName("com.sun.tools.javac.comp.Resolve$MethodResolutionContext");
        Constructor<?> constructor = methCtxClss.getDeclaredConstructors()[0];
        constructor.setAccessible(true);
        Object methodContext = constructor.newInstance(resolve);
        // we need to also initialize the fields attrMode and step
        setField(methodContext, "attrMode", DeferredAttr.AttrMode.CHECK);
        @SuppressWarnings("rawtypes")
        List<?> phases = (List) getField(resolve, "methodResolutionSteps");
        setField(methodContext, "step", phases.get(1));
        return methodContext;
    }

    /** Reflectively set a field. */
    private void setField(Object receiver, String fieldName,
            Object value) throws NoSuchFieldException,
            IllegalAccessException {
        Field f = receiver.getClass().getDeclaredField(fieldName);
        f.setAccessible(true);
        f.set(receiver, value);
    }

    /** Reflectively get the value of a field. */
    private Object getField(Object receiver, String fieldName) throws NoSuchFieldException,
            IllegalAccessException {
        Field f = receiver.getClass().getDeclaredField(fieldName);
        f.setAccessible(true);
        return f.get(receiver);
    }

    private Symbol wrapInvocation(Method method, Object... args) {
        try {
            return (Symbol) method.invoke(resolve, args);
        } catch (IllegalAccessException e) {
            Error err = new AssertionError("Unexpected Reflection error");
            err.initCause(e);
            throw err;
        } catch (IllegalArgumentException e) {
            Error err = new AssertionError("Unexpected Reflection error");
            err.initCause(e);
            throw err;
        } catch (InvocationTargetException e) {
            Error err = new AssertionError("Unexpected Reflection error");
            err.initCause(e);
            throw err;
        }
    }
}
