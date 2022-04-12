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
package proguard.classfile;

import proguard.classfile.attribute.Attribute;
import proguard.classfile.attribute.visitor.AttributeVisitor;
import proguard.classfile.constant.*;
import proguard.classfile.constant.visitor.ConstantVisitor;
import proguard.classfile.util.ClassSubHierarchyInitializer;
import proguard.classfile.visitor.*;

/**
 * This Clazz is a complete representation of the data in a Java class.
 *
 * @author Eric Lafortune
 */
public class ProgramClass implements Clazz
{
    private static final int[]           EMPTY_INTERFACES = new int[0];
    private static final ProgramField[]  EMPTY_FIELDS     = new ProgramField[0];
    private static final ProgramMethod[] EMPTY_METHODS    = new ProgramMethod[0];
    private static final Attribute[]     EMPTY_ATTRIBUTES = new Attribute[0];


    //  public int             u4magic;
    public int             u4version;
    public int             u2constantPoolCount;
    public Constant[]      constantPool;
    public int             u2accessFlags;
    public int             u2thisClass;
    public int             u2superClass;
    public int             u2interfacesCount;
    public int[]           u2interfaces;
    public int             u2fieldsCount;
    public ProgramField[]  fields;
    public int             u2methodsCount;
    public ProgramMethod[] methods;
    public int             u2attributesCount;
    public Attribute[]     attributes;

    /**
     * An extra field pointing to the subclasses of this class.
     * This field is filled out by the {@link ClassSubHierarchyInitializer}.
     */
    public Clazz[] subClasses;

    /**
     * An extra field in which visitors can store information.
     */
    public Object visitorInfo;


    /**
     * Creates an uninitialized ProgramClass.
     */
    public ProgramClass() {}


    /**
     * Creates an initialized ProgramClass without fields, methods, attributes,
     * or subclasses.
     */
    public ProgramClass(int             u4version,
                        int             u2constantPoolCount,
                        Constant[]      constantPool,
                        int             u2accessFlags,
                        int             u2thisClass,
                        int             u2superClass)
    {
        this(u4version,
             u2constantPoolCount,
             constantPool,
             u2accessFlags,
             u2thisClass,
             u2superClass,
             0,
             EMPTY_INTERFACES,
             0,
             EMPTY_FIELDS,
             0,
             EMPTY_METHODS,
             0,
             EMPTY_ATTRIBUTES,
             null);
    }


    /**
     * Creates an initialized ProgramClass.
     */
    public ProgramClass(int             u4version,
                        int             u2constantPoolCount,
                        Constant[]      constantPool,
                        int             u2accessFlags,
                        int             u2thisClass,
                        int             u2superClass,
                        int             u2interfacesCount,
                        int[]           u2interfaces,
                        int             u2fieldsCount,
                        ProgramField[]  fields,
                        int             u2methodsCount,
                        ProgramMethod[] methods,
                        int             u2attributesCount,
                        Attribute[]     attributes,
                        Clazz[]         subClasses)
    {
        this.u4version           = u4version;
        this.u2constantPoolCount = u2constantPoolCount;
        this.constantPool        = constantPool;
        this.u2accessFlags       = u2accessFlags;
        this.u2thisClass         = u2thisClass;
        this.u2superClass        = u2superClass;
        this.u2interfacesCount   = u2interfacesCount;
        this.u2interfaces        = u2interfaces;
        this.u2fieldsCount       = u2fieldsCount;
        this.fields              = fields;
        this.u2methodsCount      = u2methodsCount;
        this.methods             = methods;
        this.u2attributesCount   = u2attributesCount;
        this.attributes          = attributes;
        this.subClasses          = subClasses;
    }


    /**
     * Returns the Constant at the given index in the constant pool.
     */
    public Constant getConstant(int constantIndex)
    {
        return constantPool[constantIndex];
    }


    // Implementations for Clazz.

    public int getAccessFlags()
    {
        return u2accessFlags;
    }

    public String getName()
    {
        return getClassName(u2thisClass);
    }

    public String getSuperName()
    {
        return u2superClass == 0 ? null : getClassName(u2superClass);
    }

    public int getInterfaceCount()
    {
        return u2interfacesCount;
    }

    public String getInterfaceName(int index)
    {
        return getClassName(u2interfaces[index]);
    }

    public int getTag(int constantIndex)
    {
        return constantPool[constantIndex].getTag();
    }

    public String getString(int constantIndex)
    {
        try
        {
            return ((Utf8Constant)constantPool[constantIndex]).getString();
        }
        catch (ClassCastException ex)
        {
            throw ((IllegalStateException)new IllegalStateException("Expected Utf8Constant at index ["+constantIndex+"] in class ["+getName()+"]").initCause(ex));
        }
    }

    public String getStringString(int constantIndex)
    {
        try
        {
            return ((StringConstant)constantPool[constantIndex]).getString(this);
        }
        catch (ClassCastException ex)
        {
            throw ((IllegalStateException)new IllegalStateException("Expected StringConstant at index ["+constantIndex+"] in class ["+getName()+"]").initCause(ex));
        }
    }

    public String getClassName(int constantIndex)
    {
        try
        {
            return ((ClassConstant)constantPool[constantIndex]).getName(this);
        }
        catch (ClassCastException ex)
        {
            throw ((IllegalStateException)new IllegalStateException("Expected ClassConstant at index ["+constantIndex+"]").initCause(ex));
        }
    }

    public String getName(int constantIndex)
    {
        try
        {
            return ((NameAndTypeConstant)constantPool[constantIndex]).getName(this);
        }
        catch (ClassCastException ex)
        {
            throw ((IllegalStateException)new IllegalStateException("Expected NameAndTypeConstant at index ["+constantIndex+"] in class ["+getName()+"]").initCause(ex));
        }
    }

    public String getType(int constantIndex)
    {
        try
        {
            return ((NameAndTypeConstant)constantPool[constantIndex]).getType(this);
        }
        catch (ClassCastException ex)
        {
            throw ((IllegalStateException)new IllegalStateException("Expected NameAndTypeConstant at index ["+constantIndex+"] in class ["+getName()+"]").initCause(ex));
        }
    }


    public String getRefClassName(int constantIndex)
    {
        try
        {
            return ((RefConstant)constantPool[constantIndex]).getClassName(this);
        }
        catch (ClassCastException ex)
        {
            throw ((IllegalStateException)new IllegalStateException("Expected RefConstant at index ["+constantIndex+"] in class ["+getName()+"]").initCause(ex));
        }
    }

    public String getRefName(int constantIndex)
    {
        try
        {
            return ((RefConstant)constantPool[constantIndex]).getName(this);
        }
        catch (ClassCastException ex)
        {
            throw ((IllegalStateException)new IllegalStateException("Expected RefConstant at index ["+constantIndex+"] in class ["+getName()+"]").initCause(ex));
        }
    }

    public String getRefType(int constantIndex)
    {
        try
        {
            return ((RefConstant)constantPool[constantIndex]).getType(this);
        }
        catch (ClassCastException ex)
        {
            throw ((IllegalStateException)new IllegalStateException("Expected RefConstant at index ["+constantIndex+"] in class ["+getName()+"]").initCause(ex));
        }
    }


    public void addSubClass(Clazz clazz)
    {
        if (subClasses == null)
        {
            subClasses = new Clazz[1];
        }
        else
        {
            // Copy the old elements into new larger array.
            Clazz[] newSubClasses = new Clazz[subClasses.length+1];
            System.arraycopy(subClasses, 0, newSubClasses, 0, subClasses.length);
            subClasses = newSubClasses;
        }

        subClasses[subClasses.length-1] = clazz;
    }


    public Clazz getSuperClass()
    {
        return u2superClass != 0 ?
            ((ClassConstant)constantPool[u2superClass]).referencedClass :
            null;
    }


    public Clazz getInterface(int index)
    {
        return ((ClassConstant)constantPool[u2interfaces[index]]).referencedClass;
    }


    public boolean extends_(Clazz clazz)
    {
        if (this.equals(clazz))
        {
            return true;
        }

        Clazz superClass = getSuperClass();
        return superClass != null &&
               superClass.extends_(clazz);
    }


    public boolean extends_(String className)
    {
        if (getName().equals(className))
        {
            return true;
        }

        Clazz superClass = getSuperClass();
        return superClass != null &&
               superClass.extends_(className);
    }


    public boolean extendsOrImplements(Clazz clazz)
    {
        if (this.equals(clazz))
        {
            return true;
        }

        Clazz superClass = getSuperClass();
        if (superClass != null &&
            superClass.extendsOrImplements(clazz))
        {
            return true;
        }

        for (int index = 0; index < u2interfacesCount; index++)
        {
            Clazz interfaceClass = getInterface(index);
            if (interfaceClass != null &&
                interfaceClass.extendsOrImplements(clazz))
            {
                return true;
            }
        }

        return false;
    }


    public boolean extendsOrImplements(String className)
    {
        if (getName().equals(className))
        {
            return true;
        }

        Clazz superClass = getSuperClass();
        if (superClass != null &&
            superClass.extendsOrImplements(className))
        {
            return true;
        }

        for (int index = 0; index < u2interfacesCount; index++)
        {
            Clazz interfaceClass = getInterface(index);
            if (interfaceClass != null &&
                interfaceClass.extendsOrImplements(className))
            {
                return true;
            }
        }

        return false;
    }


    public Field findField(String name, String descriptor)
    {
        for (int index = 0; index < u2fieldsCount; index++)
        {
            Field field = fields[index];
            if ((name       == null || field.getName(this).equals(name)) &&
                (descriptor == null || field.getDescriptor(this).equals(descriptor)))
            {
                return field;
            }
        }

        return null;
    }


    public Method findMethod(String name, String descriptor)
    {
        for (int index = 0; index < u2methodsCount; index++)
        {
            Method method = methods[index];
            if ((name       == null || method.getName(this).equals(name)) &&
                (descriptor == null || method.getDescriptor(this).equals(descriptor)))
            {
                return method;
            }
        }

        return null;
    }


    public void accept(ClassVisitor classVisitor)
    {
        classVisitor.visitProgramClass(this);
    }


    public void hierarchyAccept(boolean      visitThisClass,
                                boolean      visitSuperClass,
                                boolean      visitInterfaces,
                                boolean      visitSubclasses,
                                ClassVisitor classVisitor)
    {
        // First visit the current classfile.
        if (visitThisClass)
        {
            accept(classVisitor);
        }

        // Then visit its superclass, recursively.
        if (visitSuperClass)
        {
            Clazz superClass = getSuperClass();
            if (superClass != null)
            {
                superClass.hierarchyAccept(true,
                                           true,
                                           visitInterfaces,
                                           false,
                                           classVisitor);
            }
        }

        // Then visit its interfaces, recursively.
        if (visitInterfaces)
        {
            // Visit the interfaces of the superclasses, if we haven't done so yet.
            if (!visitSuperClass)
            {
                Clazz superClass = getSuperClass();
                if (superClass != null)
                {
                    superClass.hierarchyAccept(false,
                                               false,
                                               true,
                                               false,
                                               classVisitor);
                }
            }

            // Visit the interfaces.
            for (int index = 0; index < u2interfacesCount; index++)
            {
                Clazz interfaceClass = getInterface(index);
                if (interfaceClass != null)
                {
                    interfaceClass.hierarchyAccept(true,
                                                   false,
                                                   true,
                                                   false,
                                                   classVisitor);
                }
            }
        }

        // Then visit its subclasses, recursively.
        if (visitSubclasses)
        {
            if (subClasses != null)
            {
                for (int index = 0; index < subClasses.length; index++)
                {
                    Clazz subClass = subClasses[index];
                    subClass.hierarchyAccept(true,
                                             false,
                                             false,
                                             true,
                                             classVisitor);
                }
            }
        }
    }


    public void subclassesAccept(ClassVisitor classVisitor)
    {
        if (subClasses != null)
        {
            for (int index = 0; index < subClasses.length; index++)
            {
                subClasses[index].accept(classVisitor);
            }
        }
    }


    public void constantPoolEntriesAccept(ConstantVisitor constantVisitor)
    {
        for (int index = 1; index < u2constantPoolCount; index++)
        {
            if (constantPool[index] != null)
            {
                constantPool[index].accept(this, constantVisitor);
            }
        }
    }


    public void constantPoolEntryAccept(int index, ConstantVisitor constantVisitor)
    {
        constantPool[index].accept(this, constantVisitor);
    }


    public void thisClassConstantAccept(ConstantVisitor constantVisitor)
    {
        constantPool[u2thisClass].accept(this, constantVisitor);
    }


    public void superClassConstantAccept(ConstantVisitor constantVisitor)
    {
        if (u2superClass != 0)
        {
            constantPool[u2superClass].accept(this, constantVisitor);
        }
    }


    public void interfaceConstantsAccept(ConstantVisitor constantVisitor)
    {
        for (int index = 0; index < u2interfacesCount; index++)
        {
            constantPool[u2interfaces[index]].accept(this, constantVisitor);
        }
    }


    public void fieldsAccept(MemberVisitor memberVisitor)
    {
        for (int index = 0; index < u2fieldsCount; index++)
        {
            fields[index].accept(this, memberVisitor);
        }
    }


    public void fieldAccept(String name, String descriptor, MemberVisitor memberVisitor)
    {
        Field field = findField(name, descriptor);
        if (field != null)
        {
            field.accept(this, memberVisitor);
        }
    }


    public void methodsAccept(MemberVisitor memberVisitor)
    {
        for (int index = 0; index < u2methodsCount; index++)
        {
            methods[index].accept(this, memberVisitor);
        }
    }


    public void methodAccept(String name, String descriptor, MemberVisitor memberVisitor)
    {
        Method method = findMethod(name, descriptor);
        if (method != null)
        {
            method.accept(this, memberVisitor);
        }
    }


    public boolean mayHaveImplementations(Method method)
    {
        return
            (u2accessFlags & ClassConstants.ACC_FINAL) == 0 &&
            (method == null ||
             ((method.getAccessFlags() & (ClassConstants.ACC_PRIVATE |
                                          ClassConstants.ACC_STATIC  |
                                          ClassConstants.ACC_FINAL)) == 0 &&
              !method.getName(this).equals(ClassConstants.METHOD_NAME_INIT)));
    }


    public void attributesAccept(AttributeVisitor attributeVisitor)
    {
        for (int index = 0; index < u2attributesCount; index++)
        {
            attributes[index].accept(this, attributeVisitor);
        }
    }


    public void attributeAccept(String name, AttributeVisitor attributeVisitor)
    {
        for (int index = 0; index < u2attributesCount; index++)
        {
            Attribute attribute = attributes[index];
            if (attribute.getAttributeName(this).equals(name))
            {
                attribute.accept(this, attributeVisitor);
            }
        }
    }


    // Implementations for VisitorAccepter.

    public Object getVisitorInfo()
    {
        return visitorInfo;
    }

    public void setVisitorInfo(Object visitorInfo)
    {
        this.visitorInfo = visitorInfo;
    }


    // Implementations for Object.

    public String toString()
    {
        return "ProgramClass("+getName()+")";
    }
}
