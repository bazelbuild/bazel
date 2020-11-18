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
package proguard.optimize.peephole;

import proguard.classfile.Clazz;
import proguard.classfile.constant.*;
import proguard.classfile.constant.visitor.ConstantVisitor;
import proguard.classfile.util.*;

/**
 * This ConstantVisitor delegates to a given constant visitor, except for
 * constants that contain wildcards (indices larger than 0xffff).
 *
 * @see InstructionSequenceMatcher
 * @author Eric Lafortune
 */
public class WildcardConstantFilter
implements ConstantVisitor
{
    private static final int WILDCARD = InstructionSequenceMatcher.X;


    private final ConstantVisitor constantVisitor;

    private final MyWildcardChecker wildcardChecker = new MyWildcardChecker();


    /**
     * Creates a new WildcardClassReferenceInitializer that delegates to the
     * given constant visitor.
     */
    public WildcardConstantFilter(ConstantVisitor constantVisitor)
    {
        this.constantVisitor = constantVisitor;
    }


    // Implementations for ConstantVisitor.

    public void visitIntegerConstant(Clazz clazz, IntegerConstant integerConstant)
    {
        constantVisitor.visitIntegerConstant(clazz, integerConstant);
    }


    public void visitLongConstant(Clazz clazz, LongConstant longConstant)
    {
        constantVisitor.visitLongConstant(clazz, longConstant);
    }


    public void visitFloatConstant(Clazz clazz, FloatConstant floatConstant)
    {
        constantVisitor.visitFloatConstant(clazz, floatConstant);
    }


    public void visitDoubleConstant(Clazz clazz, DoubleConstant doubleConstant)
    {
        constantVisitor.visitDoubleConstant(clazz, doubleConstant);
    }


    public void visitPrimitiveArrayConstant(Clazz clazz, PrimitiveArrayConstant primitiveArrayConstant)
    {
        constantVisitor.visitPrimitiveArrayConstant(clazz, primitiveArrayConstant);
    }


    public void visitStringConstant(Clazz clazz, StringConstant stringConstant)
    {
        if (!containsWildcard(clazz, stringConstant))
        {
            constantVisitor.visitStringConstant(clazz, stringConstant);
        }
    }


    public void visitUtf8Constant(Clazz clazz, Utf8Constant utf8Constant)
    {
        constantVisitor.visitUtf8Constant(clazz, utf8Constant);
    }


    public void visitDynamicConstant(Clazz clazz, DynamicConstant dynamicConstant)
    {
        if (!containsWildcard(clazz, dynamicConstant))
        {
            constantVisitor.visitDynamicConstant(clazz, dynamicConstant);
        }
    }


    public void visitInvokeDynamicConstant(Clazz clazz, InvokeDynamicConstant invokeDynamicConstant)
    {
        if (!containsWildcard(clazz, invokeDynamicConstant))
        {
            constantVisitor.visitInvokeDynamicConstant(clazz, invokeDynamicConstant);
        }
    }


    public void visitMethodHandleConstant(Clazz clazz, MethodHandleConstant methodHandleConstant)
    {
        constantVisitor.visitMethodHandleConstant(clazz, methodHandleConstant);
    }


    public void visitFieldrefConstant(Clazz clazz, FieldrefConstant fieldrefConstant)
    {
        if (!containsWildcard(clazz, fieldrefConstant))
        {
            constantVisitor.visitFieldrefConstant(clazz, fieldrefConstant);
        }
    }


    public void visitInterfaceMethodrefConstant(Clazz clazz, InterfaceMethodrefConstant interfaceMethodrefConstant)
    {
        if (!containsWildcard(clazz, interfaceMethodrefConstant))
        {
            constantVisitor.visitInterfaceMethodrefConstant(clazz, interfaceMethodrefConstant);
        }
    }


    public void visitMethodrefConstant(Clazz clazz, MethodrefConstant methodrefConstant)
    {
        if (!containsWildcard(clazz, methodrefConstant))
        {
            constantVisitor.visitMethodrefConstant(clazz, methodrefConstant);
        }
    }


    public void visitClassConstant(Clazz clazz, ClassConstant classConstant)
    {
        if (!containsWildcard(clazz, classConstant))
        {
            constantVisitor.visitClassConstant(clazz, classConstant);
        }
    }


    public void visitMethodTypeConstant(Clazz clazz, MethodTypeConstant methodTypeConstant)
    {
        if (!containsWildcard(clazz, methodTypeConstant))
        {
            constantVisitor.visitMethodTypeConstant(clazz, methodTypeConstant);
        }
    }


    public void visitNameAndTypeConstant(Clazz clazz, NameAndTypeConstant nameAndTypeConstant)
    {
        if (!containsWildcard(clazz, nameAndTypeConstant))
        {
            constantVisitor.visitNameAndTypeConstant(clazz, nameAndTypeConstant);
        }
    }


    public void visitModuleConstant(Clazz clazz, ModuleConstant moduleConstant)
    {
        if (!containsWildcard(clazz, moduleConstant))
        {
            constantVisitor.visitModuleConstant(clazz, moduleConstant);
        }
    }


    public void visitPackageConstant(Clazz clazz, PackageConstant packageConstant)
    {
        if (!containsWildcard(clazz, packageConstant))
        {
            constantVisitor.visitPackageConstant(clazz, packageConstant);
        }
    }


    // Small utility methods.

    /**
     * Checks whether the given constant contains wildcard indices.
     */
    private boolean containsWildcard(Clazz clazz, Constant constant)
    {
        wildcardChecker.containsWildcard = false;
        constant.accept(clazz, wildcardChecker);
        return wildcardChecker.containsWildcard;
    }



    private static class MyWildcardChecker
    extends              SimplifiedVisitor
    implements           ConstantVisitor
    {
        public boolean containsWildcard;


        // Implementations for ConstantVisitor.

        public void visitAnyConstant(Clazz clazz, Constant constant) {}


        public void visitStringConstant(Clazz clazz, StringConstant stringConstant)
        {
            if (stringConstant.u2stringIndex >= WILDCARD)
            {
                containsWildcard = true;
            }
        }


        public void visitInvokeDynamicConstant(Clazz clazz, InvokeDynamicConstant invokeDynamicConstant)
        {
            if (invokeDynamicConstant.u2nameAndTypeIndex >= WILDCARD)
            {
                containsWildcard = true;
            }
            else
            {
                // Recursively check the referenced constant.
                clazz.constantPoolEntryAccept(invokeDynamicConstant.u2nameAndTypeIndex, this);
            }
        }


        public void visitAnyRefConstant(Clazz clazz, RefConstant refConstant)
        {
            if (refConstant.u2classIndex       >= WILDCARD ||
                refConstant.u2nameAndTypeIndex >= WILDCARD)
            {
                containsWildcard = true;
            }
            else
            {
                // Recursively check the referenced constants.
                clazz.constantPoolEntryAccept(refConstant.u2classIndex,       this);
                clazz.constantPoolEntryAccept(refConstant.u2nameAndTypeIndex, this);
            }
        }


        public void visitClassConstant(Clazz clazz, ClassConstant classConstant)
        {
            if (classConstant.u2nameIndex >= WILDCARD)
            {
                containsWildcard = true;
            }
        }


        public void visitMethodTypeConstant(Clazz clazz, MethodTypeConstant methodTypeConstant)
        {
            if (methodTypeConstant.u2descriptorIndex >= WILDCARD)
            {
                containsWildcard = true;
            }
        }


        public void visitNameAndTypeConstant(Clazz clazz, NameAndTypeConstant nameAndTypeConstant)
        {
            if (nameAndTypeConstant.u2nameIndex       >= WILDCARD ||
                nameAndTypeConstant.u2descriptorIndex >= WILDCARD)
            {
                containsWildcard = true;
            }
        }


        public void visitModuleConstant(Clazz clazz, ModuleConstant moduleConstant)
        {
            if (moduleConstant.u2nameIndex >= WILDCARD)
            {
                containsWildcard = true;
            }
        }


        public void visitPackageConstant(Clazz clazz, PackageConstant packageConstant)
        {
            if (packageConstant.u2nameIndex >= WILDCARD)
            {
                containsWildcard = true;
            }
        }
    }
}
