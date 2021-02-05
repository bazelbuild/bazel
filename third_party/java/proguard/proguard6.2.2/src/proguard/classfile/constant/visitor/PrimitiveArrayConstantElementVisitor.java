/*
 * ProGuard -- shrinking, optimization, obfuscation, and preverification
 *             of Java bytecode.
 *
 * Copyright (c) 2002-2019 Guardsquare NV
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option)
 * any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 */
package proguard.classfile.constant.visitor;

import proguard.classfile.Clazz;
import proguard.classfile.constant.PrimitiveArrayConstant;

/**
 * This interface specifies the methods for a visitor of primitive elements
 * of the array of a PrimitiveArrayConstant.
 *
 * @author Eric Lafortune
 */
public interface PrimitiveArrayConstantElementVisitor
{
    public void visitBooleanArrayConstantElement(Clazz clazz, PrimitiveArrayConstant primitiveArrayConstant, int index, boolean value);
    public void visitByteArrayConstantElement(Clazz clazz, PrimitiveArrayConstant primitiveArrayConstant, int index, byte value);
    public void visitCharArrayConstantElement(Clazz clazz, PrimitiveArrayConstant primitiveArrayConstant, int index, char value);
    public void visitShortArrayConstantElement(Clazz clazz, PrimitiveArrayConstant primitiveArrayConstant, int index, short value);
    public void visitIntArrayConstantElement(Clazz clazz, PrimitiveArrayConstant primitiveArrayConstant, int index, int value);
    public void visitFloatArrayConstantElement(Clazz clazz, PrimitiveArrayConstant primitiveArrayConstant, int index, float value);
    public void visitLongArrayConstantElement(Clazz clazz, PrimitiveArrayConstant primitiveArrayConstant, int index, long value);
    public void visitDoubleArrayConstantElement(Clazz clazz, PrimitiveArrayConstant primitiveArrayConstant, int index, double value);
}
