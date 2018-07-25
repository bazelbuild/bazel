package com.google.devtools.build.lib.syntax;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import java.lang.reflect.Method;
import java.util.Arrays;

/**
 * A value class to store Methods with their corresponding {@link SkylarkCallable} annotation
 * metadata. This is needed because the annotation is sometimes in a superclass.
 *
 * <p>The annotation metadata is duplicated in this class to avoid usage of Java dynamic proxies
 * which are ~7X slower.
 */
public final class MethodDescriptor {
  private final Method method;
  private final SkylarkCallable annotation;

  private final String name;
  private final String doc;
  private final boolean documented;
  private final boolean structField;
  private final ImmutableList<ParamDescriptor> parameters;
  private final ParamDescriptor extraPositionals;
  private final ParamDescriptor extraKeywords;
  private final boolean selfCall;
  private final boolean allowReturnNones;
  private final boolean useLocation;
  private final boolean useAst;
  private final boolean useEnvironment;
  private final boolean useSkylarkSemantics;

  private MethodDescriptor(
      Method method,
      SkylarkCallable annotation,
      String name,
      String doc,
      boolean documented,
      boolean structField,
      ImmutableList<ParamDescriptor> parameters,
      ParamDescriptor extraPositionals,
      ParamDescriptor extraKeywords,
      boolean selfCall,
      boolean allowReturnNones,
      boolean useLocation,
      boolean useAst,
      boolean useEnvironment,
      boolean useSkylarkSemantics) {
    this.method = method;
    this.annotation = annotation;
    this.name = name;
    this.doc = doc;
    this.documented = documented;
    this.structField = structField;
    this.parameters = parameters;
    this.extraPositionals = extraPositionals;
    this.extraKeywords = extraKeywords;
    this.selfCall = selfCall;
    this.allowReturnNones = allowReturnNones;
    this.useLocation = useLocation;
    this.useAst = useAst;
    this.useEnvironment = useEnvironment;
    this.useSkylarkSemantics = useSkylarkSemantics;
  }

  Method getMethod() {
    return method;
  }

  /** Returns the SkylarkCallable annotation corresponding to this method. */
  public SkylarkCallable getAnnotation() {
    return annotation;
  }

  /** @return Skylark method descriptor for provided Java method and signature annotation. */
  public static MethodDescriptor of(Method method, SkylarkCallable annotation) {
    return new MethodDescriptor(
        method,
        annotation,
        annotation.name(),
        annotation.doc(),
        annotation.documented(),
        annotation.structField(),
        Arrays.stream(annotation.parameters())
            .map(ParamDescriptor::of)
            .collect(ImmutableList.toImmutableList()),
        ParamDescriptor.of(annotation.extraPositionals()),
        ParamDescriptor.of(annotation.extraKeywords()),
        annotation.selfCall(),
        annotation.allowReturnNones(),
        annotation.useLocation(),
        annotation.useAst(),
        annotation.useEnvironment(),
        annotation.useSkylarkSemantics());
  }

  /** @see SkylarkCallable#name() */
  public String getName() {
    return name;
  }

  /** @see SkylarkCallable#structField() */
  public boolean isStructField() {
    return structField;
  }

  /** @see SkylarkCallable#useEnvironment() */
  public boolean isUseEnvironment() {
    return useEnvironment;
  }

  /** @see SkylarkCallable#useSkylarkSemantics() */
  public boolean isUseSkylarkSemantics() {
    return useSkylarkSemantics;
  }

  /** @see SkylarkCallable#useLocation() */
  public boolean isUseLocation() {
    return useLocation;
  }

  /** @see SkylarkCallable#allowReturnNones() */
  public boolean isAllowReturnNones() {
    return allowReturnNones;
  }

  /** @see SkylarkCallable#useAst() */
  public boolean isUseAst() {
    return useAst;
  }

  /** @see SkylarkCallable#extraPositionals() */
  public ParamDescriptor getExtraPositionals() {
    return extraPositionals;
  }

  public ParamDescriptor getExtraKeywords() {
    return extraKeywords;
  }

  /** @return {@code true} if this method accepts extra arguments ({@code *args}) */
  public boolean isAcceptsExtraArgs() {
    return !getExtraPositionals().getName().isEmpty();
  }

  /** @see SkylarkCallable#extraKeywords() */
  public boolean isAcceptsExtraKwargs() {
    return !getExtraKeywords().getName().isEmpty();
  }

  /** @see SkylarkCallable#parameters() */
  public ImmutableList<ParamDescriptor> getParameters() {
    return parameters;
  }

  /** @see SkylarkCallable#documented() */
  public boolean isDocumented() {
    return documented;
  }

  /** @see SkylarkCallable#doc() */
  public String getDoc() {
    return doc;
  }

  /** @see SkylarkCallable#selfCall() */
  public boolean isSelfCall() {
    return selfCall;
  }
}
