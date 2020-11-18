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
package proguard.classfile.io;

import proguard.classfile.*;
import proguard.classfile.constant.*;
import proguard.classfile.constant.visitor.ConstantVisitor;
import proguard.classfile.util.*;
import proguard.classfile.visitor.*;

import java.io.DataInput;

/**
 * This ClassVisitor fills out the LibraryClass objects that it visits with data
 * from the given DataInput object.
 *
 * @author Eric Lafortune
 */
public class LibraryClassReader
extends      SimplifiedVisitor
implements   ClassVisitor,
             MemberVisitor,
             ConstantVisitor
{
    private static final LibraryField[]  EMPTY_LIBRARY_FIELDS  = new LibraryField[0];
    private static final LibraryMethod[] EMPTY_LIBRARY_METHODS = new LibraryMethod[0];


    private final RuntimeDataInput dataInput;
    private final boolean          skipNonPublicClasses;
    private final boolean          skipNonPublicClassMembers;

    // A global array that acts as a parameter for the visitor methods.
    private Constant[]      constantPool;


    /**
     * Creates a new ProgramClassReader for reading from the given DataInput.
     */
    public LibraryClassReader(DataInput dataInput,
                              boolean   skipNonPublicClasses,
                              boolean   skipNonPublicClassMembers)
    {
        this.dataInput                 = new RuntimeDataInput(dataInput);
        this.skipNonPublicClasses      = skipNonPublicClasses;
        this.skipNonPublicClassMembers = skipNonPublicClassMembers;
    }


    // Implementations for ClassVisitor.

    public void visitProgramClass(ProgramClass libraryClass)
    {
    }


    public void visitLibraryClass(LibraryClass libraryClass)
    {
        // Read and check the magic number.
        int u4magic = dataInput.readInt();

        ClassUtil.checkMagicNumber(u4magic);

        // Read and check the version numbers.
        int u2minorVersion = dataInput.readUnsignedShort();
        int u2majorVersion = dataInput.readUnsignedShort();

        int u4version = ClassUtil.internalClassVersion(u2majorVersion,
                                                       u2minorVersion);

        ClassUtil.checkVersionNumbers(u4version);

        // Read the constant pool. Note that the first entry is not used.
        int u2constantPoolCount = dataInput.readUnsignedShort();

        // Create the constant pool array.
        constantPool = new Constant[u2constantPoolCount];

        for (int index = 1; index < u2constantPoolCount; index++)
        {
            Constant constant = createConstant();
            constant.accept(libraryClass, this);

            int tag = constant.getTag();
            if (tag == ClassConstants.CONSTANT_Class ||
                tag == ClassConstants.CONSTANT_Utf8)
            {
                constantPool[index] = constant;
            }

            // Long constants and double constants take up two entries in the
            // constant pool.
            if (tag == ClassConstants.CONSTANT_Long ||
                tag == ClassConstants.CONSTANT_Double)
            {
                index++;
            }
        }

        // Read the general class information.
        libraryClass.u2accessFlags = dataInput.readUnsignedShort();

        // We may stop parsing this library class if it's not public anyway.
        // E.g. only about 60% of all rt.jar classes need to be parsed.
        if (skipNonPublicClasses &&
            AccessUtil.accessLevel(libraryClass.getAccessFlags()) < AccessUtil.PUBLIC)
        {
            return;
        }

        // Read the class and super class indices.
        int u2thisClass  = dataInput.readUnsignedShort();
        int u2superClass = dataInput.readUnsignedShort();

        // Store their actual names.
        libraryClass.thisClassName  = getClassName(u2thisClass);
        libraryClass.superClassName = (u2superClass == 0) ? null :
                                      getClassName(u2superClass);

        // Read the interfaces
        int u2interfacesCount = dataInput.readUnsignedShort();

        libraryClass.interfaceNames = new String[u2interfacesCount];
        for (int index = 0; index < u2interfacesCount; index++)
        {
            // Store the actual interface name.
            int u2interface = dataInput.readUnsignedShort();
            libraryClass.interfaceNames[index] = getClassName(u2interface);
        }

        // Read the fields.
        int u2fieldsCount = dataInput.readUnsignedShort();

        // Create the fields array.
        LibraryField[] reusableFields = new LibraryField[u2fieldsCount];

        int visibleFieldsCount = 0;
        for (int index = 0; index < u2fieldsCount; index++)
        {
            LibraryField field = new LibraryField();
            this.visitLibraryMember(libraryClass, field);

            // Only store fields that are visible.
            if (AccessUtil.accessLevel(field.getAccessFlags()) >=
                (skipNonPublicClassMembers ? AccessUtil.PROTECTED :
                                             AccessUtil.PACKAGE_VISIBLE))
            {
                reusableFields[visibleFieldsCount++] = field;
            }
        }

        // Copy the visible fields (if any) into a fields array of the right size.
        if (visibleFieldsCount == 0)
        {
            libraryClass.fields = EMPTY_LIBRARY_FIELDS;
        }
        else
        {
            libraryClass.fields = new LibraryField[visibleFieldsCount];
            System.arraycopy(reusableFields, 0, libraryClass.fields, 0, visibleFieldsCount);
        }

        // Read the methods.
        int u2methodsCount = dataInput.readUnsignedShort();

        // Create the methods array.
        LibraryMethod[] reusableMethods = new LibraryMethod[u2methodsCount];

        int visibleMethodsCount = 0;
        for (int index = 0; index < u2methodsCount; index++)
        {
            LibraryMethod method = new LibraryMethod();
            this.visitLibraryMember(libraryClass, method);

            // Only store methods that are visible.
            if (AccessUtil.accessLevel(method.getAccessFlags()) >=
                (skipNonPublicClassMembers ? AccessUtil.PROTECTED :
                                             AccessUtil.PACKAGE_VISIBLE))
            {
                reusableMethods[visibleMethodsCount++] = method;
            }
        }

        // Copy the visible methods (if any) into a methods array of the right size.
        if (visibleMethodsCount == 0)
        {
            libraryClass.methods = EMPTY_LIBRARY_METHODS;
        }
        else
        {
            libraryClass.methods = new LibraryMethod[visibleMethodsCount];
            System.arraycopy(reusableMethods, 0, libraryClass.methods, 0, visibleMethodsCount);
        }

        // Skip the class attributes.
        skipAttributes();
    }


    // Implementations for MemberVisitor.

    public void visitProgramMember(ProgramClass libraryClass, ProgramMember libraryMember)
    {
    }


    public void visitLibraryMember(LibraryClass libraryClass, LibraryMember libraryMember)
    {
        // Read the general field information.
        libraryMember.u2accessFlags = dataInput.readUnsignedShort();
        libraryMember.name          = getString(dataInput.readUnsignedShort());
        libraryMember.descriptor    = getString(dataInput.readUnsignedShort());

        // Skip the field attributes.
        skipAttributes();
    }


    // Implementations for ConstantVisitor.

    public void visitIntegerConstant(Clazz clazz, IntegerConstant integerConstant)
    {
        dataInput.skipBytes(4);
    }


    public void visitLongConstant(Clazz clazz, LongConstant longConstant)
    {
        dataInput.skipBytes(8);
    }


    public void visitFloatConstant(Clazz clazz, FloatConstant floatConstant)
    {
        dataInput.skipBytes(4);
    }


    public void visitDoubleConstant(Clazz clazz, DoubleConstant doubleConstant)
    {
        dataInput.skipBytes(8);
    }


    public void visitPrimitiveArrayConstant(Clazz clazz, PrimitiveArrayConstant primitiveArrayConstant)
    {
        char u2primitiveType = dataInput.readChar();
        int  u4length        = dataInput.readInt();

        dataInput.skipBytes(primitiveSize(u2primitiveType) * u4length);
    }


    public void visitStringConstant(Clazz clazz, StringConstant stringConstant)
    {
        dataInput.skipBytes(2);
    }


    public void visitUtf8Constant(Clazz clazz, Utf8Constant utf8Constant)
    {
        int u2length = dataInput.readUnsignedShort();

        // Read the UTF-8 bytes.
        byte[] bytes = new byte[u2length];
        dataInput.readFully(bytes);
        utf8Constant.setBytes(bytes);
    }


    public void visitDynamicConstant(Clazz clazz, DynamicConstant dynamicConstant)
    {
        dataInput.skipBytes(4);
    }


    public void visitInvokeDynamicConstant(Clazz clazz, InvokeDynamicConstant invokeDynamicConstant)
    {
        dataInput.skipBytes(4);
    }


    public void visitMethodHandleConstant(Clazz clazz, MethodHandleConstant methodHandleConstant)
    {
        dataInput.skipBytes(3);
    }


    public void visitAnyRefConstant(Clazz clazz, RefConstant refConstant)
    {
        dataInput.skipBytes(4);
    }


    public void visitClassConstant(Clazz clazz, ClassConstant classConstant)
    {
        classConstant.u2nameIndex = dataInput.readUnsignedShort();
    }


    public void visitMethodTypeConstant(Clazz clazz, MethodTypeConstant methodTypeConstant)
    {
        dataInput.skipBytes(2);
    }


    public void visitNameAndTypeConstant(Clazz clazz, NameAndTypeConstant nameAndTypeConstant)
    {
        dataInput.skipBytes(4);
    }


    public void visitModuleConstant(Clazz clazz, ModuleConstant moduleConstant)
    {
        dataInput.skipBytes(2);
    }


    public void visitPackageConstant(Clazz clazz, PackageConstant packageConstant)
    {
        dataInput.skipBytes(2);
    }

    // Small utility methods.

    /**
     * Returns the class name of the ClassConstant at the specified index in the
     * reusable constant pool.
     */
    private String getClassName(int constantIndex)
    {
        ClassConstant classEntry = (ClassConstant)constantPool[constantIndex];

        return getString(classEntry.u2nameIndex);
    }


    /**
     * Returns the string of the Utf8Constant at the specified index in the
     * reusable constant pool.
     */
    private String getString(int constantIndex)
    {
        return ((Utf8Constant)constantPool[constantIndex]).getString();
    }


    private Constant createConstant()
    {
        int u1tag = dataInput.readUnsignedByte();

        switch (u1tag)
        {
            case ClassConstants.CONSTANT_Integer:            return new IntegerConstant();
            case ClassConstants.CONSTANT_Float:              return new FloatConstant();
            case ClassConstants.CONSTANT_Long:               return new LongConstant();
            case ClassConstants.CONSTANT_Double:             return new DoubleConstant();
            case ClassConstants.CONSTANT_String:             return new StringConstant();
            case ClassConstants.CONSTANT_Utf8:               return new Utf8Constant();
            case ClassConstants.CONSTANT_Dynamic:            return new DynamicConstant();
            case ClassConstants.CONSTANT_InvokeDynamic:      return new InvokeDynamicConstant();
            case ClassConstants.CONSTANT_MethodHandle:       return new MethodHandleConstant();
            case ClassConstants.CONSTANT_Fieldref:           return new FieldrefConstant();
            case ClassConstants.CONSTANT_Methodref:          return new MethodrefConstant();
            case ClassConstants.CONSTANT_InterfaceMethodref: return new InterfaceMethodrefConstant();
            case ClassConstants.CONSTANT_Class:              return new ClassConstant();
            case ClassConstants.CONSTANT_MethodType:         return new MethodTypeConstant();
            case ClassConstants.CONSTANT_NameAndType:        return new NameAndTypeConstant();
            case ClassConstants.CONSTANT_Module:             return new ModuleConstant();
            case ClassConstants.CONSTANT_Package:            return new PackageConstant();

            default: throw new RuntimeException("Unknown constant type ["+u1tag+"] in constant pool");
        }
    }


    private void skipAttributes()
    {
        int u2attributesCount = dataInput.readUnsignedShort();

        for (int index = 0; index < u2attributesCount; index++)
        {
            skipAttribute();
        }
    }


    private void skipAttribute()
    {
        dataInput.skipBytes(2);
        int u4attributeLength = dataInput.readInt();
        dataInput.skipBytes(u4attributeLength);
    }


    /**
     * Returns the size in bytes of the given primitive type.
     */
    private int primitiveSize(char primitiveType)
    {
        switch (primitiveType)
        {
            case ClassConstants.TYPE_BOOLEAN:
            case ClassConstants.TYPE_BYTE:    return 1;
            case ClassConstants.TYPE_CHAR:
            case ClassConstants.TYPE_SHORT:   return 2;
            case ClassConstants.TYPE_INT:
            case ClassConstants.TYPE_FLOAT:   return 4;
            case ClassConstants.TYPE_LONG:
            case ClassConstants.TYPE_DOUBLE:  return 8;
        }

        return 0;
    }
}
