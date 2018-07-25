package com.google.devtools.build.lib.syntax;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.skylarkinterface.Param;
import java.util.Arrays;

/** A value class for storing {@link Param} metadata to avoid using Java proxies. */
public final class ParamDescriptor {

  private String name;
  private final String doc;
  private final String defaultValue;
  private final Class<?> type;
  private final ImmutableList<ParamTypeDescriptor> allowedTypes;
  private final Class<?> generic1;
  private final boolean callbackEnabled;
  private final boolean noneable;
  private final boolean named;
  private final boolean legacyNamed;
  private final boolean positional;

  private ParamDescriptor(
      String name,
      String doc,
      String defaultValue,
      Class<?> type,
      ImmutableList<ParamTypeDescriptor> allowedTypes,
      Class<?> generic1,
      boolean callbackEnabled,
      boolean noneable,
      boolean named,
      boolean legacyNamed,
      boolean positional) {
    this.name = name;
    this.doc = doc;
    this.defaultValue = defaultValue;
    this.type = type;
    this.allowedTypes = allowedTypes;
    this.generic1 = generic1;
    this.callbackEnabled = callbackEnabled;
    this.noneable = noneable;
    this.named = named;
    this.legacyNamed = legacyNamed;
    this.positional = positional;
  }

  static ParamDescriptor of(Param param) {
    return new ParamDescriptor(
        param.name(),
        param.doc(),
        param.defaultValue(),
        param.type(),
        Arrays.stream(param.allowedTypes())
            .map(ParamTypeDescriptor::of)
            .collect(ImmutableList.toImmutableList()),
        param.generic1(),
        param.callbackEnabled(),
        param.noneable(),
        param.named(),
        param.legacyNamed(),
        param.positional());
  }

  /** @see Param#name() */
  public String getName() {
    return name;
  }

  /** @see Param#allowedTypes() */
  public ImmutableList<ParamTypeDescriptor> getAllowedTypes() {
    return allowedTypes;
  }

  /** @see Param#type() */
  public Class<?> getType() {
    return type;
  }

  /** @see Param#generic1() */
  public Class<?> getGeneric1() {
    return generic1;
  }

  /** @see Param#noneable() */
  public boolean isNoneable() {
    return noneable;
  }

  /** @see Param#positional() */
  public boolean isPositional() {
    return positional;
  }

  /** @see Param#named() */
  public boolean isNamed() {
    return named;
  }

  /** @see Param#legacyNamed() */
  public boolean isLegacyNamed() {
    return legacyNamed;
  }

  /** @see Param#defaultValue() */
  public String getDefaultValue() {
    return defaultValue;
  }
}
