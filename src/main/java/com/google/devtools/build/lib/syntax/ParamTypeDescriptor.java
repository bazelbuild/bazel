package com.google.devtools.build.lib.syntax;

import com.google.devtools.build.lib.skylarkinterface.ParamType;

/** A value class to store {@link ParamType} metadata to avoid using Java proxies. */
public final class ParamTypeDescriptor {

  private final Class<?> type;
  private final Class<?> generic1;

  private ParamTypeDescriptor(Class<?> type, Class<?> generic1) {
    this.type = type;
    this.generic1 = generic1;
  }

  /** @see ParamType#type() */
  public Class<?> getType() {
    return type;
  }

  /** @see ParamType#generic1() */
  public Class<?> getGeneric1() {
    return generic1;
  }

  static ParamTypeDescriptor of(ParamType paramType) {
    return new ParamTypeDescriptor(paramType.type(), paramType.generic1());
  }
}
