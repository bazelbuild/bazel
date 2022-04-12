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
package proguard.classfile.editor;

import proguard.classfile.*;
import proguard.classfile.attribute.*;
import proguard.classfile.constant.*;
import proguard.classfile.constant.visitor.ConstantVisitor;

/**
 * This ConstantVisitor adds all constants that it visits to the constant pool
 * of a given target class.
 *
 * Bootstrap methods attributes are automatically updated for invokedynamic
 * constants.
 *
 * @author Eric Lafortune
 */
public class ConstantAdder
implements   ConstantVisitor
{
    private final ConstantPoolEditor             constantPoolEditor;
    private final BootstrapMethodsAttributeAdder bootstrapMethodsAttributeAdder;

    private int constantIndex;


    /**
     * Creates a new ConstantAdder that will copy constants into the given
     * target class.
     */
    public ConstantAdder(ProgramClass targetClass)
    {
        constantPoolEditor             = new ConstantPoolEditor(targetClass);
        bootstrapMethodsAttributeAdder = new BootstrapMethodsAttributeAdder(targetClass);
    }


    /**
     * Adds a copy of the specified constant in the given class and returns
     * its index. If the specified index is 0, the returned value is 0 too.
     */
    public int addConstant(Clazz clazz, int constantIndex)
    {
        clazz.constantPoolEntryAccept(constantIndex, this);

        return this.constantIndex;
    }


    /**
     * Adds a copy of the given constant in the given class and returns
     * its index.
     */
    public int addConstant(Clazz clazz, Constant constant)
    {
        constant.accept(clazz, this);

        return this.constantIndex;
    }


    /**
     * Returns the index of the most recently created constant in the constant
     * pool of the target class.
     */
    public int getConstantIndex()
    {
        return constantIndex;
    }


    // Implementations for ConstantVisitor.

    public void visitIntegerConstant(Clazz clazz, IntegerConstant integerConstant)
    {
        constantIndex =
            constantPoolEditor.addIntegerConstant(integerConstant.getValue());
    }


    public void visitLongConstant(Clazz clazz, LongConstant longConstant)
    {
        constantIndex =
            constantPoolEditor.addLongConstant(longConstant.getValue());
    }


    public void visitFloatConstant(Clazz clazz, FloatConstant floatConstant)
    {
        constantIndex =
            constantPoolEditor.addFloatConstant(floatConstant.getValue());
    }


    public void visitDoubleConstant(Clazz clazz, DoubleConstant doubleConstant)
    {
        constantIndex =
            constantPoolEditor.addDoubleConstant(doubleConstant.getValue());
    }


    public void visitPrimitiveArrayConstant(Clazz clazz, PrimitiveArrayConstant primitiveArrayConstant)
    {
        constantIndex =
            constantPoolEditor.addPrimitiveArrayConstant(primitiveArrayConstant.getValues());
    }


    public void visitStringConstant(Clazz clazz, StringConstant stringConstant)
    {
        constantIndex =
            constantPoolEditor.addStringConstant(stringConstant.getString(clazz),
                                                 stringConstant.referencedClass,
                                                 stringConstant.referencedMember);
    }


    public void visitUtf8Constant(Clazz clazz, Utf8Constant utf8Constant)
    {
        constantIndex =
            constantPoolEditor.addUtf8Constant(utf8Constant.getString());
    }


    public void visitDynamicConstant(Clazz clazz, DynamicConstant dynamicConstant)
    {
        // Find the bootstrap methods attribute.
        AttributesEditor attributesEditor =
            new AttributesEditor((ProgramClass)clazz, false);

        BootstrapMethodsAttribute bootstrapMethodsAttribute =
            (BootstrapMethodsAttribute)attributesEditor.findAttribute(ClassConstants.ATTR_BootstrapMethods);

        // Add the name and type constant.
        clazz.constantPoolEntryAccept(dynamicConstant.u2nameAndTypeIndex, this);

        // Copy the referenced classes.
        Clazz[] referencedClasses     = dynamicConstant.referencedClasses;
        Clazz[] referencedClassesCopy = null;
        if (referencedClasses != null)
        {
            referencedClassesCopy = new Clazz[referencedClasses.length];
            System.arraycopy(referencedClasses, 0,
                             referencedClassesCopy, 0,
                             referencedClasses.length);
        }

        bootstrapMethodsAttribute.bootstrapMethodEntryAccept(clazz,
                                                             dynamicConstant.getBootstrapMethodAttributeIndex(),
                                                             bootstrapMethodsAttributeAdder);

        // Then add the actual invoke dynamic constant.
        constantIndex =
            constantPoolEditor.addDynamicConstant(bootstrapMethodsAttributeAdder.getBootstrapMethodIndex(),
                                                  constantIndex,
                                                  referencedClassesCopy);
    }


    public void visitInvokeDynamicConstant(Clazz clazz, InvokeDynamicConstant invokeDynamicConstant)
    {
        // Find the bootstrap methods attribute.
        AttributesEditor attributesEditor =
            new AttributesEditor((ProgramClass)clazz, false);

        BootstrapMethodsAttribute bootstrapMethodsAttribute =
            (BootstrapMethodsAttribute)attributesEditor.findAttribute(ClassConstants.ATTR_BootstrapMethods);

        // Add the name and type constant.
        clazz.constantPoolEntryAccept(invokeDynamicConstant.u2nameAndTypeIndex, this);

        // Copy the referenced classes.
        Clazz[] referencedClasses     = invokeDynamicConstant.referencedClasses;
        Clazz[] referencedClassesCopy = null;
        if (referencedClasses != null)
        {
            referencedClassesCopy = new Clazz[referencedClasses.length];
            System.arraycopy(referencedClasses, 0,
                             referencedClassesCopy, 0,
                             referencedClasses.length);
        }

        bootstrapMethodsAttribute.bootstrapMethodEntryAccept(clazz,
                                                             invokeDynamicConstant.getBootstrapMethodAttributeIndex(),
                                                             bootstrapMethodsAttributeAdder);

        // Then add the actual invoke dynamic constant.
        constantIndex =
            constantPoolEditor.addInvokeDynamicConstant(bootstrapMethodsAttributeAdder.getBootstrapMethodIndex(),
                                                        constantIndex,
                                                        referencedClassesCopy);
    }


    public void visitMethodHandleConstant(Clazz clazz, MethodHandleConstant methodHandleConstant)
    {
        // First add the field ref, interface method ref, or method ref
        // constant.
        clazz.constantPoolEntryAccept(methodHandleConstant.u2referenceIndex, this);

        // Then add the actual method handle constant.
        constantIndex =
            constantPoolEditor.addMethodHandleConstant(methodHandleConstant.getReferenceKind(),
                                                       constantIndex);
    }


    public void visitModuleConstant(Clazz clazz, ModuleConstant moduleConstant)
    {
        constantIndex =
            constantPoolEditor.addModuleConstant(moduleConstant.getName(clazz));
    }


    public void visitPackageConstant(Clazz clazz, PackageConstant packageConstant)
    {
        constantIndex =
            constantPoolEditor.addPackageConstant(packageConstant.getName(clazz));
    }


    public void visitFieldrefConstant(Clazz clazz, FieldrefConstant fieldrefConstant)
    {
        // First add the referenced class constant, with its own referenced class.
        clazz.constantPoolEntryAccept(fieldrefConstant.u2classIndex, this);

        // Then add the actual field reference constant, with its referenced
        // class and class member.
        constantIndex =
            constantPoolEditor.addFieldrefConstant(constantIndex,
                                                   fieldrefConstant.getName(clazz),
                                                   fieldrefConstant.getType(clazz),
                                                   fieldrefConstant.referencedClass,
                                                   fieldrefConstant.referencedMember);
    }


    public void visitInterfaceMethodrefConstant(Clazz clazz, InterfaceMethodrefConstant interfaceMethodrefConstant)
    {
        // First add the referenced class constant, with its own referenced class.
        clazz.constantPoolEntryAccept(interfaceMethodrefConstant.u2classIndex, this);

        // Then add the actual interface method reference constant, with its
        // referenced class and class member.
        constantIndex =
            constantPoolEditor.addInterfaceMethodrefConstant(constantIndex,
                                                             interfaceMethodrefConstant.getName(clazz),
                                                             interfaceMethodrefConstant.getType(clazz),
                                                             interfaceMethodrefConstant.referencedClass,
                                                             interfaceMethodrefConstant.referencedMember);
    }


    public void visitMethodrefConstant(Clazz clazz, MethodrefConstant methodrefConstant)
    {
        // First add the referenced class constant, with its own referenced class.
        clazz.constantPoolEntryAccept(methodrefConstant.u2classIndex, this);

        // Then add the actual method reference constant, with its referenced
        // class and class member.
        constantIndex =
            constantPoolEditor.addMethodrefConstant(constantIndex,
                                                    methodrefConstant.getName(clazz),
                                                    methodrefConstant.getType(clazz),
                                                    methodrefConstant.referencedClass,
                                                    methodrefConstant.referencedMember);
    }


    public void visitClassConstant(Clazz clazz, ClassConstant classConstant)
    {
        // Add the class constant, with its referenced class..
        constantIndex =
            constantPoolEditor.addClassConstant(classConstant.getName(clazz),
                                                classConstant.referencedClass);
    }


    public void visitMethodTypeConstant(Clazz clazz, MethodTypeConstant methodTypeConstant)
    {
        constantIndex =
            constantPoolEditor.addMethodTypeConstant(methodTypeConstant.getType(clazz),
                                                     methodTypeConstant.referencedClasses);
    }


    public void visitNameAndTypeConstant(Clazz clazz, NameAndTypeConstant nameAndTypeConstant)
    {
        constantIndex =
            constantPoolEditor.addNameAndTypeConstant(nameAndTypeConstant.getName(clazz),
                                                      nameAndTypeConstant.getType(clazz));
    }
}
