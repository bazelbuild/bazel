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
package proguard.classfile.util;

import proguard.classfile.*;
import proguard.classfile.visitor.*;

/**
 * This utility class provides methods to find class members in a given class
 * or in its hierarchy.
 *
 * @author Eric Lafortune
 */
public class MemberFinder
extends      SimplifiedVisitor
implements   MemberVisitor
{
    private static class MemberFoundException extends RuntimeException {}
    private static final MemberFoundException MEMBER_FOUND = new MemberFoundException();

    private final boolean searchHierarchy;

    private Clazz  clazz;
    private Member member;


    /**
     * Creates a new MemberFinder that looks in the class hierarchy.
     */
    public MemberFinder()
    {
        this(true);
    }


    /**
     * Creates a new MemberFinder that looks in the class hierarchy if
     * specified.
     */
    public MemberFinder(boolean searchHierarchy)
    {
        this.searchHierarchy = searchHierarchy;
    }


    /**
     * Finds the field with the given name and descriptor in the given
     * class or its hierarchy. The name and descriptor may contain wildcards.
     */
    public Field findField(Clazz  clazz,
                           String name,
                           String descriptor)
    {
        return findField(null, clazz, name, descriptor);
    }


    /**
     * Finds the field with the given name and descriptor in the given
     * class or its hierarchy. The name and descriptor may contain wildcards.
     */
    public Field findField(Clazz  referencingClass,
                           Clazz  clazz,
                           String name,
                           String descriptor)
    {
        return (Field)findMember(referencingClass, clazz, name, descriptor, true);
    }


    /**
     * Finds the method with the given name and descriptor in the given
     * class or its hierarchy. The name and descriptor may contain wildcards.
     */
    public Method findMethod(Clazz  clazz,
                             String name,
                             String descriptor)
    {
        return findMethod(null, clazz, name, descriptor);
    }


    /**
     * Finds the method with the given name and descriptor in the given
     * class or its hierarchy. The name and descriptor may contain wildcards.
     */
    public Method findMethod(Clazz  referencingClass,
                             Clazz  clazz,
                             String name,
                             String descriptor)
    {
        return (Method)findMember(referencingClass, clazz, name, descriptor, false);
    }


    /**
     * Finds the class member with the given name and descriptor in the given
     * class or its hierarchy. The name and descriptor may contain wildcards.
     */
    public Member findMember(Clazz   clazz,
                             String  name,
                             String  descriptor,
                             boolean isField)
    {
        return findMember(null, clazz, name, descriptor, isField);
    }


    /**
     * Finds the class member with the given name and descriptor in the given
     * class or its hierarchy, referenced from the optional given class.
     * The name and descriptor may contain wildcards.
     */
    public Member findMember(Clazz   referencingClass,
                             Clazz   clazz,
                             String  name,
                             String  descriptor,
                             boolean isField)
    {
        try
        {
            boolean containsWildcards =
                (name       != null && (name.indexOf('*')       >= 0 || name.indexOf('?')       >= 0)) ||
                (descriptor != null && (descriptor.indexOf('*') >= 0 || descriptor.indexOf('?') >= 0));

            this.clazz  = null;
            this.member = null;

            // The class member may be in a different class, as of Java 1.2.
            // The class member may be private in a nest member, as of Java 11.
            // Check the accessibility from the referencing class, if any
            // (non-dummy), taking into account access flags and nests.
            MemberVisitor memberVisitor =
                referencingClass           != null &&
                referencingClass.getName() != null ?
                    new MemberClassAccessFilter(referencingClass, this) :
                    this;

            // We'll return with a MemberFoundException as soon as we've found
            // the class member.
            clazz.hierarchyAccept(true,
                                  searchHierarchy,
                                  searchHierarchy,
                                  false,
                                  containsWildcards ?
                    isField ?
                        new AllFieldVisitor(
                        new MemberNameFilter(name,
                        new MemberDescriptorFilter(descriptor,
                        memberVisitor))) :

                        new AllMethodVisitor(
                        new MemberNameFilter(name,
                        new MemberDescriptorFilter(descriptor,
                        memberVisitor))) :
                   isField ?
                        new NamedFieldVisitor(name, descriptor,
                        memberVisitor) :

                        new NamedMethodVisitor(name, descriptor,
                        memberVisitor));
        }
        catch (MemberFoundException ex)
        {
            // We've found the member we were looking for.
        }

        return member;
    }


    /**
     * Returns the corresponding class of the most recently found class
     * member.
     */
    public Clazz correspondingClass()
    {
        return clazz;
    }


    /**
     * Returns whether the given method is overridden anywhere down the class
     * hierarchy.
     */
    public boolean isOverriden(Clazz  clazz,
                               Method method)
    {
        String name       = method.getName(clazz);
        String descriptor = method.getDescriptor(clazz);

        // Go looking for the method down the class hierarchy.
        try
        {
            this.clazz  = null;
            this.member = null;

            clazz.hierarchyAccept(false, false, false, true,
                new NamedMethodVisitor(name, descriptor,
                new MemberAccessFilter(0, ClassConstants.ACC_PRIVATE, this)));
        }
        catch (MemberFoundException ex)
        {
            // We've found an overriding method.
            return true;
        }

        return false;
    }


    /**
     * Returns whether the given field is shadowed anywhere down the class
     * hierarchy.
     */
    public boolean isShadowed(Clazz clazz,
                              Field field)
    {
        String name       = field.getName(clazz);
        String descriptor = field.getDescriptor(clazz);

        // Go looking for the field down the class hierarchy.
        try
        {
            this.clazz  = null;
            this.member = null;
            clazz.hierarchyAccept(false, false, false, true,
                new NamedFieldVisitor(name, descriptor,
                new MemberAccessFilter(0, ClassConstants.ACC_PRIVATE, this)));
        }
        catch (MemberFoundException ex)
        {
            // We've found a shadowing field.
            return true;
        }

        return false;
    }


//    // Implementations for ClassVisitor.
//
//    private void visitAnyClass(Clazz clazz)
//    {
//        if (member == null)
//        {
//            member = isField ?
//                (Member)clazz.findField(name, descriptor) :
//                (Member)clazz.findMethod(name, descriptor);
//
//            if (member != null)
//            {
//                this.clazz = clazz;
//            }
//        }
//    }


    // Implementations for MemberVisitor.

    public void visitAnyMember(Clazz clazz, Member member)
    {
        this.clazz  = clazz;
        this.member = member;

        throw MEMBER_FOUND;
    }
}
