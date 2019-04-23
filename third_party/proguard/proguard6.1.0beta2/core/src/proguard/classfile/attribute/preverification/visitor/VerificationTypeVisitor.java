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
package proguard.classfile.attribute.preverification.visitor;

import proguard.classfile.*;
import proguard.classfile.attribute.CodeAttribute;
import proguard.classfile.attribute.preverification.*;

/**
 * This interface specifies the methods for a visitor of
 * <code>VerificationType</code> objects. There a methods for stack entries
 * and methods for variable entries.
 *
 * @author Eric Lafortune
 */
public interface VerificationTypeVisitor
{
    public void visitIntegerType(          Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, IntegerType           integerType);
    public void visitFloatType(            Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, FloatType             floatType);
    public void visitLongType(             Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, LongType              longType);
    public void visitDoubleType(           Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, DoubleType            doubleType);
    public void visitTopType(              Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, TopType               topType);
    public void visitObjectType(           Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, ObjectType            objectType);
    public void visitNullType(             Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, NullType              nullType);
    public void visitUninitializedType(    Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, UninitializedType     uninitializedType);
    public void visitUninitializedThisType(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, UninitializedThisType uninitializedThisType);

    public void visitStackIntegerType(          Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, int index, IntegerType           integerType);
    public void visitStackFloatType(            Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, int index, FloatType             floatType);
    public void visitStackLongType(             Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, int index, LongType              longType);
    public void visitStackDoubleType(           Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, int index, DoubleType            doubleType);
    public void visitStackTopType(              Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, int index, TopType               topType);
    public void visitStackObjectType(           Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, int index, ObjectType            objectType);
    public void visitStackNullType(             Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, int index, NullType              nullType);
    public void visitStackUninitializedType(    Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, int index, UninitializedType     uninitializedType);
    public void visitStackUninitializedThisType(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, int index, UninitializedThisType uninitializedThisType);

    public void visitVariablesIntegerType(          Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, int index, IntegerType           integerType);
    public void visitVariablesFloatType(            Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, int index, FloatType             floatType);
    public void visitVariablesLongType(             Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, int index, LongType              longType);
    public void visitVariablesDoubleType(           Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, int index, DoubleType            doubleType);
    public void visitVariablesTopType(              Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, int index, TopType               topType);
    public void visitVariablesObjectType(           Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, int index, ObjectType            objectType);
    public void visitVariablesNullType(             Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, int index, NullType              nullType);
    public void visitVariablesUninitializedType(    Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, int index, UninitializedType     uninitializedType);
    public void visitVariablesUninitializedThisType(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, int index, UninitializedThisType uninitializedThisType);
}
