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
package proguard.obfuscate;

import proguard.classfile.*;
import proguard.classfile.util.*;
import proguard.classfile.visitor.MemberVisitor;

import java.util.Map;

/**
 * This MemberInfoVisitor solves obfuscation naming conflicts in all class
 * members that it visits. It avoids names from the given descriptor map,
 * delegating to the given obfuscator in order to get a new name if necessary.
 *
 * @author Eric Lafortune
 */
public class MemberNameConflictFixer implements MemberVisitor
{
    private final boolean          allowAggressiveOverloading;
    private final Map              descriptorMap;
    private final WarningPrinter   warningPrinter;
    private final MemberObfuscator memberObfuscator;


    /**
     * Creates a new MemberNameConflictFixer.
     * @param allowAggressiveOverloading a flag that specifies whether class
     *                                   members can be overloaded aggressively.
     * @param descriptorMap              the map of descriptors to
     *                                   [new name - old name] maps.
     * @param warningPrinter             an optional warning printer to which
     *                                   warnings about conflicting name
     *                                   mappings can be printed.
     * @param memberObfuscator           the obfuscator that can assign new
     *                                   names to members with conflicting
     *                                   names.
     */
    public MemberNameConflictFixer(boolean          allowAggressiveOverloading,
                                   Map              descriptorMap,
                                   WarningPrinter   warningPrinter,
                                   MemberObfuscator memberObfuscator)
    {
        this.allowAggressiveOverloading = allowAggressiveOverloading;
        this.descriptorMap              = descriptorMap;
        this.warningPrinter             = warningPrinter;
        this.memberObfuscator           = memberObfuscator;
    }




    // Implementations for MemberVisitor.

    public void visitProgramField(ProgramClass programClass, ProgramField programField)
    {
        visitMember(programClass, programField, true);
    }


    public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
    {
        // Special cases: <clinit> and <init> are always kept unchanged.
        // We can ignore them here.
        String name = programMethod.getName(programClass);
        if (ClassUtil.isInitializer(name))
        {
            return;
        }

        visitMember(programClass, programMethod, false);
    }


    public void visitLibraryField(LibraryClass libraryClass, LibraryField libraryField) {}
    public void visitLibraryMethod(LibraryClass libraryClass, LibraryMethod libraryMethod) {}


    /**
     * Obfuscates the given class member.
     * @param clazz   the class  of the given member.
     * @param member  the class member to be obfuscated.
     * @param isField specifies whether the class member is a field.
     */
    private void visitMember(Clazz   clazz,
                             Member  member,
                             boolean isField)
    {
        // Get the member's name and descriptor.
        String name       = member.getName(clazz);
        String descriptor = member.getDescriptor(clazz);

        // Check whether we're allowed to overload aggressively.
        if (!allowAggressiveOverloading)
        {
            // Trim the return argument from the descriptor if not.
            // Works for fields and methods alike.
            descriptor = descriptor.substring(0, descriptor.indexOf(')')+1);
        }

        // Get the name map.
        Map nameMap = MemberObfuscator.retrieveNameMap(descriptorMap, descriptor);

        // Get the member's new name.
        String newName = MemberObfuscator.newMemberName(member);

        // Get the expected old name for this new name.
        String previousName = (String)nameMap.get(newName);
        if (previousName != null &&
            !name.equals(previousName))
        {
            // There's a conflict! A member (with a given old name) in a
            // first namespace has received the same new name as this
            // member (with a different old name) in a second name space,
            // and now these two have to live together in this name space.
            if (MemberObfuscator.hasFixedNewMemberName(member) &&
                warningPrinter != null)
            {
                descriptor = member.getDescriptor(clazz);
                warningPrinter.print(clazz.getName(),
                                     "Warning: " + ClassUtil.externalClassName(clazz.getName()) +
                                                   (isField ?
                                                       ": field '" + ClassUtil.externalFullFieldDescription(0, name, descriptor) :
                                                       ": method '" + ClassUtil.externalFullMethodDescription(clazz.getName(), 0, name, descriptor)) +
                                     "' can't be mapped to '" + newName +
                                     "' because it would conflict with " +
                                     (isField ?
                                         "field '" :
                                         "method '" ) + previousName +
                                     "', which is already being mapped to '" + newName + "'");
            }

            // Clear the conflicting name.
            MemberObfuscator.setNewMemberName(member, null);

            // Assign a new name.
            member.accept(clazz, memberObfuscator);
        }
    }
}
