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

import proguard.classfile.attribute.visitor.AttributeVisitor;
import proguard.classfile.constant.visitor.ConstantVisitor;
import proguard.classfile.visitor.*;


/**
 * This interface provides access to the representation of a Java class.
 *
 * @author Eric Lafortune
 */
public interface Clazz extends VisitorAccepter
{
    /**
     * Returns the access flags of this class.
     * @see ClassConstants
     */
    public int getAccessFlags();

    /**
     * Returns the full internal name of this class.
     */
    public String getName();

    /**
     * Returns the full internal name of the super class of this class, or
     * null if this class represents java.lang.Object.
     */
    public String getSuperName();

    /**
     * Returns the number of interfaces that this class implements.
     */
    public int getInterfaceCount();

    /**
     * Returns the full internal name of the interface at the given index of
     * this class.
     */
    public String getInterfaceName(int index);

    /**
     * Returns the tag value of the Constant at the specified index.
     */
    public int getTag(int constantIndex);

    /**
     * Returns the String value of the Utf8Constant at the specified index.
     */
    public String getString(int constantIndex);

    /**
     * Returns the String value of the StringConstant at the specified index.
     */
    public String getStringString(int constantIndex);

    /**
     * Returns the class name of ClassConstant at the specified index.
     */
    public String getClassName(int constantIndex);

    /**
     * Returns the name of the NameAndTypeConstant at the specified index.
     */
    public String getName(int constantIndex);

    /**
     * Returns the type of the NameAndTypeConstant at the specified index.
     */
    public String getType(int constantIndex);

    /**
     * Returns the class name of the RefConstant at the specified index.
     */
    public String getRefClassName(int constantIndex);

    /**
     * Returns the name of the RefConstant at the specified index.
     */
    public String getRefName(int constantIndex);

    /**
     * Returns the type of the RefConstant at the specified index.
     */
    public String getRefType(int constantIndex);


    // Methods pertaining to related classes.

    /**
     * Notifies this Clazz that it is being subclassed by another class.
     */
    public void addSubClass(Clazz clazz);

    /**
     * Returns the super class of this class.
     */
    public Clazz getSuperClass();

    /**
     * Returns the interface at the given index.
     */
    public Clazz getInterface(int index);

    /**
     * Returns whether this class extends the given class.
     * A class is always considered to extend itself.
     * Interfaces are considered to only extend the root Object class.
     */
    public boolean extends_(Clazz clazz);

    /**
     * Returns whether this class extends the specified class.
     * A class is always considered to extend itself.
     * Interfaces are considered to only extend the root Object class.
     */
    public boolean extends_(String className);

    /**
     * Returns whether this class implements the given class.
     * A class is always considered to implement itself.
     * Interfaces are considered to implement all their superinterfaces.
     */
    public boolean extendsOrImplements(Clazz clazz);

    /**
     * Returns whether this class implements the specified class.
     * A class is always considered to implement itself.
     * Interfaces are considered to implement all their superinterfaces.
     */
    public boolean extendsOrImplements(String className);


    // Methods for getting specific class members.

    /**
     * Returns the field with the given name and descriptor.
     */
    Field findField(String name, String descriptor);

    /**
     * Returns the method with the given name and descriptor.
     */
    Method findMethod(String name, String descriptor);


    // Methods for accepting various types of visitors.

    /**
     * Accepts the given class visitor.
     */
    public void accept(ClassVisitor classVisitor);

    /**
     * Accepts the given class visitor in the class hierarchy.
     * @param visitThisClass   specifies whether to visit this class.
     * @param visitSuperClass  specifies whether to visit the super classes.
     * @param visitInterfaces  specifies whether to visit the interfaces.
     * @param visitSubclasses  specifies whether to visit the subclasses.
     * @param classVisitor     the <code>ClassVisitor</code> that will
     *                         visit the class hierarchy.
     */
    public void hierarchyAccept(boolean      visitThisClass,
                                boolean      visitSuperClass,
                                boolean      visitInterfaces,
                                boolean      visitSubclasses,
                                ClassVisitor classVisitor);

    /**
     * Lets the given class visitor visit all known subclasses.
     * @param classVisitor the <code>ClassVisitor</code> that will visit the
     *                     subclasses.
     */
    public void subclassesAccept(ClassVisitor classVisitor);

    /**
     * Lets the given constant pool entry visitor visit all constant pool entries
     * of this class.
     */
    public void constantPoolEntriesAccept(ConstantVisitor constantVisitor);

    /**
     * Lets the given constant pool entry visitor visit the constant pool entry
     * at the specified index.
     */
    public void constantPoolEntryAccept(int index, ConstantVisitor constantVisitor);

    /**
     * Lets the given constant pool entry visitor visit the class constant pool
     * entry of this class.
     */
    public void thisClassConstantAccept(ConstantVisitor constantVisitor);

    /**
     * Lets the given constant pool entry visitor visit the class constant pool
     * entry of the super class of this class, if there is one.
     */
    public void superClassConstantAccept(ConstantVisitor constantVisitor);

    /**
     * Lets the given constant pool entry visitor visit the class constant pool
     * entries for all interfaces of this class.
     */
    public void interfaceConstantsAccept(ConstantVisitor constantVisitor);

    /**
     * Lets the given member info visitor visit all fields of this class.
     */
    public void fieldsAccept(MemberVisitor memberVisitor);

    /**
     * Lets the given member info visitor visit the specified field.
     */
    public void fieldAccept(String name, String descriptor, MemberVisitor memberVisitor);

    /**
     * Lets the given member info visitor visit all methods of this class.
     */
    public void methodsAccept(MemberVisitor memberVisitor);

    /**
     * Lets the given member info visitor visit the specified method.
     */
    public void methodAccept(String name, String descriptor, MemberVisitor memberVisitor);

    /**
     * Returns whether the given method may possibly have implementing or
     * overriding methods down the class hierarchy. This can only be true
     * if the class is not final, and the method is not private, static, or
     * final, or a constructor.
     * @param method the method that may have implementations.
     * @return whether it may have implementations.
     */
    public boolean mayHaveImplementations(Method method);

    /**
     * Lets the given attribute info visitor visit all attributes of this class.
     */
    public void attributesAccept(AttributeVisitor attributeVisitor);

    /**
     * Lets the given attribute info visitor visit the specified attribute.
     */
    public void attributeAccept(String name, AttributeVisitor attributeVisitor);
}
