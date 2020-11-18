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
import proguard.classfile.constant.*;


/**
 * This interface specifies the methods for a visitor of <code>Constant</code>
 * objects.
 *
 * @author Eric Lafortune
 */
public interface ConstantVisitor
{
    public void visitIntegerConstant(           Clazz clazz, IntegerConstant            integerConstant);
    public void visitLongConstant(              Clazz clazz, LongConstant               longConstant);
    public void visitFloatConstant(             Clazz clazz, FloatConstant              floatConstant);
    public void visitDoubleConstant(            Clazz clazz, DoubleConstant             doubleConstant);
    public void visitPrimitiveArrayConstant(    Clazz clazz, PrimitiveArrayConstant     primitiveArrayConstant);
    public void visitStringConstant(            Clazz clazz, StringConstant             stringConstant);
    public void visitUtf8Constant(              Clazz clazz, Utf8Constant               utf8Constant);
    public void visitDynamicConstant(           Clazz clazz, DynamicConstant            dynamicConstant);
    public void visitInvokeDynamicConstant(     Clazz clazz, InvokeDynamicConstant      invokeDynamicConstant);
    public void visitMethodHandleConstant(      Clazz clazz, MethodHandleConstant       methodHandleConstant);
    public void visitFieldrefConstant(          Clazz clazz, FieldrefConstant           fieldrefConstant);
    public void visitInterfaceMethodrefConstant(Clazz clazz, InterfaceMethodrefConstant interfaceMethodrefConstant);
    public void visitMethodrefConstant(         Clazz clazz, MethodrefConstant          methodrefConstant);
    public void visitClassConstant(             Clazz clazz, ClassConstant              classConstant);
    public void visitMethodTypeConstant(        Clazz clazz, MethodTypeConstant         methodTypeConstant);
    public void visitNameAndTypeConstant(       Clazz clazz, NameAndTypeConstant        nameAndTypeConstant);
    public void visitModuleConstant(            Clazz clazz, ModuleConstant             moduleConstant);
    public void visitPackageConstant(           Clazz clazz, PackageConstant            packageConstant);
}
